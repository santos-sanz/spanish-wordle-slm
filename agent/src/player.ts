import { Agent, type AgentTool } from "@earendil-works/pi-agent-core";
import { Type, contentText, type Model, type Models } from "@earendil-works/pi-ai";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { WordleBridge, type HistoryRow } from "./bridge.js";

export type Track = "pure" | "agent" | "oracle";

// LFM2.5 otherwise opens a hidden reasoning block and can spend the entire
// response budget before emitting the short JSON answer it was trained on.
const LFM_THINK_START_TOKEN_ID = "124901";
const WORD_TOKEN_IDS = JSON.parse(
  readFileSync(resolve("data/processed/word_token_ids.json"), "utf8"),
) as Record<string, number[]>;

const PURE_SYSTEM =
  "Modo PURE. Juegas Wordle en español. La palabra objetivo tiene cinco letras y dispones de seis " +
  "intentos. 0=gris, 1=amarillo, 2=verde. Respeta letras repetidas. Responde únicamente " +
  'con JSON válido: {"guess":"palabra"}.';

function systemPrompt(track: Track): string {
  if (track === "agent") {
    return `Modo AGENT. Juegas Wordle en español. La palabra objetivo tiene cinco letras y dispones de seis ` +
      `intentos. 0=gris, 1=amarillo, 2=verde. Respeta letras repetidas. Responde únicamente ` +
      `con JSON válido: {"guess":"palabra"}. Puedes usar get_candidates una vez por turno ` +
      `para consultar soluciones compatibles.`;
  }
  if (track === "oracle") {
    return `${PURE_SYSTEM} Debes usar best_guess para obtener la jugada del solver.`;
  }
  return PURE_SYSTEM;
}

function promptFor(history: HistoryRow[], turn: number): string {
  const rows = history.length
    ? history.map((row) => `${row.guess} -> ${row.feedback}`).join("\n")
    : "Sin intentos previos.";
  return `Turno ${turn}/6. Historial:\n${rows}\nElige el siguiente intento.`;
}

function finalText(messages: readonly unknown[]): string {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index] as any;
    if (message?.role !== "assistant") continue;
    if (typeof message.content === "string") return message.content;
    if (Array.isArray(message.content)) {
      return message.content
        .filter((block: any) => block?.type === "text")
        .map((block: any) => block.text ?? "")
        .join("");
    }
  }
  return "";
}

function parseGuess(text: string): string | null {
  const withoutThinking = text.replace(/<think>[\s\S]*?<\/think>/gi, " ");
  try {
    const start = withoutThinking.indexOf("{");
    const end = withoutThinking.lastIndexOf("}");
    if (start >= 0 && end > start) {
      const parsed = JSON.parse(withoutThinking.slice(start, end + 1));
      if (typeof parsed.guess === "string") return parsed.guess.toLowerCase();
    }
  } catch {
    // Fall through to the strict five-letter extraction.
  }
  const normalized = withoutThinking.toLowerCase();
  const strict = normalized.match(/(?<![a-zñ])[a-zñ]{5}(?![a-zñ])/u);
  if (strict) return strict[0];
  // Small models occasionally emit a valid five-letter Wordle word with a
  // plural/adjectival suffix (e.g. "cobros").  Recovering the first five
  // letters lets the bridge perform the normal validity check without giving
  // the player any target or candidate information.
  return normalized.match(/[a-zñ]{5}/u)?.[0] ?? null;
}

export type TurnResult = {
  guess: string | null;
  invalidActions: number;
  toolCalls: number;
  latencyMs: number;
  inputTokens: number;
  outputTokens: number;
  costUsd: number;
  effectiveProviders: string[];
  responseModels: string[];
  transcript: unknown[];
};

function telemetry(messages: readonly unknown[]) {
  const assistant = messages.filter(
    (message: any) => message?.role === "assistant" && message?.usage,
  ) as any[];
  return {
    inputTokens: assistant.reduce((sum, message) => sum + (message.usage.input ?? 0), 0),
    outputTokens: assistant.reduce((sum, message) => sum + (message.usage.output ?? 0), 0),
    costUsd: assistant.reduce((sum, message) => sum + (message.usage.cost?.total ?? 0), 0),
    effectiveProviders: [...new Set(assistant.map((message) => String(message.provider)))],
    responseModels: [
      ...new Set(assistant.map((message) => String(message.responseModel ?? message.model))),
    ],
  };
}

export async function chooseGuess(options: {
  models: Models;
  model: Model<any>;
  bridge: WordleBridge;
  history: HistoryRow[];
  turn: number;
  track: Track;
}): Promise<TurnResult> {
  const { models, model, bridge, history, turn, track } = options;
  const started = performance.now();
  if (track === "oracle") {
    const result = await bridge.bestGuess(history);
    return {
      guess: result.guess,
      invalidActions: 0,
      toolCalls: 1,
      latencyMs: performance.now() - started,
      inputTokens: 0,
      outputTokens: 0,
      costUsd: 0,
      effectiveProviders: [model.provider],
      responseModels: [model.id],
      transcript: [{ role: "tool", name: "best_guess", content: result }],
    };
  }
  let toolCalls = 0;
  let candidateCalls = 0;
  const tools: AgentTool<any>[] = [];
  if (track === "agent") {
    tools.push({
      name: "get_candidates",
      label: "Get candidates",
      description: "Return every Spanish Wordle answer compatible with the current history.",
      parameters: Type.Object({}, { additionalProperties: false }),
      executionMode: "sequential",
      execute: async () => {
        candidateCalls += 1;
        toolCalls += 1;
        if (candidateCalls > 1) throw new Error("get_candidates may be called once per turn");
        const result = await bridge.candidates(history);
        return {
          content: [{ type: "text", text: JSON.stringify(result) }],
          details: { count: result.count },
        };
      },
    });
  }

  const makeStreamFn = (
    temperature: number,
    seed: number,
    blockedWords: readonly string[] = [],
    repetitionPenalty = 0,
  ) =>
    (selectedModel: Model<any>, context: any, options: any) =>
      models.streamSimple(selectedModel, context, {
        ...options,
        temperature,
        // A Wordle action is a tiny JSON object. Keep the fixed 512-token
        // contract for OpenRouter, but cap local MLX generations so invalid
        // repair attempts cannot spend most of the benchmark on empty
        // reasoning tails.
        maxTokens: selectedModel.provider === "openrouter" ? 512 : 128,
        samplingParams:
          selectedModel.provider === "openrouter"
            ? { seed: 20260814, provider: { allow_fallbacks: false } }
            : {
                seed,
                ...(repetitionPenalty > 0
                  ? { repetition_penalty: repetitionPenalty, repetition_context_size: 256 }
                  : {}),
                logit_bias: {
                  [LFM_THINK_START_TOKEN_ID]: -100,
                  ...Object.fromEntries(
                    blockedWords
                      .flatMap((word) => WORD_TOKEN_IDS[word] ?? [])
                      .map((tokenId) => [String(tokenId), -100]),
                  ),
                },
                // mlx_lm 0.31.3 drops the CLI adapter while resolving its
                // default-model alias. Sending it per request makes the
                // loaded policy explicit and testable.
                ...(process.env.WORDLE_ADAPTER_PATH?.trim().toLowerCase() === "none"
                  ? {}
                  : {
                      adapters: resolve(
                        process.env.WORDLE_ADAPTER_PATH?.trim() || "adapters/selected",
                      ),
                    }),
              },
      });
  const streamFn = makeStreamFn(0, 20260814);
  const agent = new Agent({
    initialState: {
      systemPrompt: systemPrompt(track),
      model,
      thinkingLevel: "low",
      tools,
      messages: [],
    },
    prepareNextTurnWithContext: ({ context }) => {
      if (track !== "agent" || candidateCalls === 0 || !context.tools?.length) return undefined;
      return { context: { ...context, tools: [] } };
    },
    streamFn,
    toolExecution: "sequential",
  });
  await agent.prompt(promptFor(history, turn));
  if (agent.state.errorMessage) throw new Error(`Pi provider error: ${agent.state.errorMessage}`);
  let invalidActions = 0;
  let guess = parseGuess(finalText(agent.state.messages));
  const previousGuesses = new Set(history.map((row) => row.guess));
  const attemptedGuesses = new Set(previousGuesses);
  const transcriptMessages: unknown[] = [...agent.state.messages];
  while (invalidActions < 2) {
    const validation = guess ? await bridge.validateWord(guess) : { valid: false };
    if (validation.valid && guess && !attemptedGuesses.has(guess)) break;
    if (guess) attemptedGuesses.add(guess);
    invalidActions += 1;
    const unavailable = [...attemptedGuesses].join(", ") || "ninguna";
    const repairMessage =
      `${promptFor(history, turn)}\nRespuesta inválida o repetida. ` +
      `Palabras no disponibles: ${unavailable}. ` +
      'Devuelve únicamente JSON con una palabra válida nueva: {"guess":"palabra"}.';
    if (track === "pure") {
      // A fresh context prevents the adapter from copying its own invalid
      // previous answer.  It is still Pure: only the public history and the
      // unavailable guesses are supplied, never the candidate set or target.
      const repairAgent = new Agent({
        initialState: {
          systemPrompt: systemPrompt(track),
          model,
          thinkingLevel: "low",
          tools: [],
          messages: [],
        },
        streamFn:
          model.provider === "openrouter"
            ? streamFn
            : makeStreamFn(
                0.2,
                20260814 + invalidActions,
                [...attemptedGuesses],
                1.25,
              ),
        toolExecution: "sequential",
      });
      await repairAgent.prompt(repairMessage);
      if (repairAgent.state.errorMessage)
        throw new Error(`Pi provider error: ${repairAgent.state.errorMessage}`);
      transcriptMessages.push(...repairAgent.state.messages);
      guess = parseGuess(finalText(repairAgent.state.messages));
    } else {
      await agent.prompt(
        `Respuesta inválida o repetida. Palabras no disponibles: ${unavailable}. ` +
          'Devuelve únicamente JSON con una palabra válida nueva: {"guess":"palabra"}.',
      );
      if (agent.state.errorMessage)
        throw new Error(`Pi provider error: ${agent.state.errorMessage}`);
      transcriptMessages.push(...agent.state.messages.slice(-2));
      guess = parseGuess(finalText(agent.state.messages));
    }
  }
  if (
    !guess ||
    !(await bridge.validateWord(guess)).valid ||
    attemptedGuesses.has(guess)
  ) guess = null;
  const usage = telemetry(transcriptMessages);
  return {
    guess,
    invalidActions,
    toolCalls,
    latencyMs: performance.now() - started,
    ...usage,
    transcript: transcriptMessages,
  };
}
