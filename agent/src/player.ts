import { Agent, type AgentTool } from "@earendil-works/pi-agent-core";
import { Type, contentText, type Model, type Models } from "@earendil-works/pi-ai";
import { WordleBridge, type HistoryRow } from "./bridge.js";

export type Track = "pure" | "agent" | "oracle";

// LFM2.5 otherwise opens a hidden reasoning block and can spend the entire
// response budget before emitting the short JSON answer it was trained on.
const LFM_THINK_START_TOKEN_ID = "124901";

const PURE_SYSTEM =
  "Juegas Wordle en español. La palabra objetivo tiene cinco letras y dispones de seis " +
  "intentos. 0=gris, 1=amarillo, 2=verde. Respeta letras repetidas. Responde únicamente " +
  'con JSON válido: {"guess":"palabra"}.';

function systemPrompt(track: Track): string {
  if (track === "agent") {
    return `${PURE_SYSTEM} Puedes usar get_candidates una vez por turno para consultar soluciones compatibles.`;
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
  const match = withoutThinking.toLowerCase().match(/(?<![a-zñ])[a-zñ]{5}(?![a-zñ])/u);
  return match?.[0] ?? null;
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

  const agent = new Agent({
    initialState: {
      systemPrompt: systemPrompt(track),
      model,
      thinkingLevel: "low",
      tools,
      messages: [],
    },
    streamFn: (selectedModel, context, options) =>
      models.streamSimple(selectedModel, context, {
        ...options,
        temperature: 0,
        maxTokens: 512,
        samplingParams:
          selectedModel.provider === "openrouter"
            ? { seed: 20260814, provider: { allow_fallbacks: false } }
            : { seed: 20260814, logit_bias: { [LFM_THINK_START_TOKEN_ID]: -100 } },
      }),
    toolExecution: "sequential",
  });
  await agent.prompt(promptFor(history, turn));
  if (agent.state.errorMessage) throw new Error(`Pi provider error: ${agent.state.errorMessage}`);
  let invalidActions = 0;
  let guess = parseGuess(finalText(agent.state.messages));
  while (invalidActions < 2) {
    const validation = guess ? await bridge.validateWord(guess) : { valid: false };
    if (validation.valid) break;
    invalidActions += 1;
    await agent.prompt(
      'Respuesta inválida. Devuelve únicamente JSON con una palabra válida: {"guess":"palabra"}.',
    );
    if (agent.state.errorMessage) throw new Error(`Pi provider error: ${agent.state.errorMessage}`);
    guess = parseGuess(finalText(agent.state.messages));
  }
  if (!guess || !(await bridge.validateWord(guess)).valid) guess = null;
  const usage = telemetry(agent.state.messages);
  return {
    guess,
    invalidActions,
    toolCalls,
    latencyMs: performance.now() - started,
    ...usage,
    transcript: [...agent.state.messages],
  };
}
