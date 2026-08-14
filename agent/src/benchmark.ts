import "dotenv/config";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import { parseArgs } from "node:util";
import { WordleBridge, type HistoryRow } from "./bridge.js";
import { loadModel, type ModelTarget } from "./models.js";
import { chooseGuess, type Track } from "./player.js";

type GameResult = {
  target: string;
  solved: boolean;
  turns: number;
  scoredTurns: number;
  invalidActions: number;
  toolCalls: number;
  latencyMs: number;
  inputTokens: number;
  outputTokens: number;
  costUsd: number;
  effectiveProviders: string[];
  responseModels: string[];
  errors: string[];
  history: HistoryRow[];
};

async function loadTargets(split: string): Promise<string[]> {
  const value = JSON.parse(await readFile("data/processed/splits.json", "utf8"));
  if (!Array.isArray(value[split])) throw new Error(`unknown split: ${split}`);
  return value[split];
}

async function main() {
  const { values } = parseArgs({
    options: {
      provider: { type: "string", default: "slm" },
      track: { type: "string", default: "pure" },
      split: { type: "string", default: "test" },
      limit: { type: "string" },
    },
  });
  const provider = values.provider as ModelTarget;
  const track = values.track as Track;
  if (!(["slm", "deepseek"] as string[]).includes(provider)) throw new Error("invalid provider");
  if (!(["pure", "agent", "oracle"] as string[]).includes(track)) throw new Error("invalid track");
  const targets = (await loadTargets(values.split)).slice(
    0,
    values.limit ? Number(values.limit) : undefined,
  );
  const bridge = new WordleBridge();
  const { models, model } = await loadModel(provider);
  const games: GameResult[] = [];
  try {
    for (const [index, target] of targets.entries()) {
      const history: HistoryRow[] = [];
      let invalidActions = 0;
      let toolCalls = 0;
      let latencyMs = 0;
      let inputTokens = 0;
      let outputTokens = 0;
      let costUsd = 0;
      const effectiveProviders = new Set<string>();
      const responseModels = new Set<string>();
      const errors: string[] = [];
      let solved = false;
      for (let turn = 1; turn <= 6; turn += 1) {
        let result;
        try {
          result = await chooseGuess({ models, model, bridge, history, turn, track });
        } catch (error) {
          errors.push(error instanceof Error ? error.message : String(error));
          break;
        }
        invalidActions += result.invalidActions;
        toolCalls += result.toolCalls;
        latencyMs += result.latencyMs;
        inputTokens += result.inputTokens;
        outputTokens += result.outputTokens;
        costUsd += result.costUsd;
        result.effectiveProviders.forEach((value) => effectiveProviders.add(value));
        result.responseModels.forEach((value) => responseModels.add(value));
        if (!result.guess) break;
        const feedback = await bridge.feedback(target, result.guess);
        history.push({ guess: result.guess, feedback: feedback.feedback });
        if (feedback.feedback === "22222") {
          solved = true;
          break;
        }
      }
      games.push({
        target,
        solved,
        turns: history.length,
        scoredTurns: solved ? history.length : 7,
        invalidActions,
        toolCalls,
        latencyMs,
        inputTokens,
        outputTokens,
        costUsd,
        effectiveProviders: [...effectiveProviders],
        responseModels: [...responseModels],
        errors,
        history,
      });
      process.stderr.write(`[${index + 1}/${targets.length}] ${target}: ${solved ? history.length : "loss"}\n`);
    }
  } finally {
    bridge.close();
  }
  const summary = {
    provider,
    model: model.id,
    track,
    split: values.split,
    games: games.length,
    wins: games.filter((game) => game.solved).length,
    winRate: games.filter((game) => game.solved).length / games.length,
    meanScoredTurns: games.reduce((sum, game) => sum + game.scoredTurns, 0) / games.length,
    invalidActions: games.reduce((sum, game) => sum + game.invalidActions, 0),
    toolCalls: games.reduce((sum, game) => sum + game.toolCalls, 0),
    latencyMs: games.reduce((sum, game) => sum + game.latencyMs, 0),
    inputTokens: games.reduce((sum, game) => sum + game.inputTokens, 0),
    outputTokens: games.reduce((sum, game) => sum + game.outputTokens, 0),
    costUsd: games.reduce((sum, game) => sum + game.costUsd, 0),
    errors: games.reduce((sum, game) => sum + game.errors.length, 0),
    effectiveProviders: [...new Set(games.flatMap((game) => game.effectiveProviders))],
    responseModels: [...new Set(games.flatMap((game) => game.responseModels))],
  };
  await mkdir("artifacts/benchmark", { recursive: true });
  const output = resolve("artifacts/benchmark", `${provider}-${track}.json`);
  await writeFile(output, `${JSON.stringify({ summary, games }, null, 2)}\n`);
  console.log(JSON.stringify(summary, null, 2));
}

await main();
