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

type BenchmarkFile = {
  complete: boolean;
  targetCount: number;
  summary: Record<string, unknown>;
  games: GameResult[];
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
      resume: { type: "boolean", default: false },
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
  await mkdir("artifacts/benchmark", { recursive: true });
  const output = resolve("artifacts/benchmark", `${provider}-${track}.json`);
  let games: GameResult[] = [];
  if (values.resume) {
    try {
      const previous = JSON.parse(await readFile(output, "utf8")) as BenchmarkFile;
      const prefixMatches = previous.games.every(
        (game, index) => game.target === targets[index],
      );
      if (
        previous.summary.model !== model.id ||
        previous.summary.track !== track ||
        previous.summary.split !== values.split ||
        previous.targetCount !== targets.length ||
        !prefixMatches
      ) {
        throw new Error("existing benchmark checkpoint does not match this run");
      }
      games = previous.games;
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
  }

  const summaryFor = (currentGames: GameResult[]) => ({
    provider,
    model: model.id,
    track,
    split: values.split,
    games: currentGames.length,
    wins: currentGames.filter((game) => game.solved).length,
    winRate: currentGames.length
      ? currentGames.filter((game) => game.solved).length / currentGames.length
      : 0,
    meanScoredTurns: currentGames.length
      ? currentGames.reduce((sum, game) => sum + game.scoredTurns, 0) / currentGames.length
      : 0,
    invalidActions: currentGames.reduce((sum, game) => sum + game.invalidActions, 0),
    toolCalls: currentGames.reduce((sum, game) => sum + game.toolCalls, 0),
    latencyMs: currentGames.reduce((sum, game) => sum + game.latencyMs, 0),
    inputTokens: currentGames.reduce((sum, game) => sum + game.inputTokens, 0),
    outputTokens: currentGames.reduce((sum, game) => sum + game.outputTokens, 0),
    costUsd: currentGames.reduce((sum, game) => sum + game.costUsd, 0),
    errors: currentGames.reduce((sum, game) => sum + game.errors.length, 0),
    effectiveProviders: [...new Set(currentGames.flatMap((game) => game.effectiveProviders))],
    responseModels: [...new Set(currentGames.flatMap((game) => game.responseModels))],
  });

  const persist = async (complete: boolean) => {
    const result: BenchmarkFile = {
      complete,
      targetCount: targets.length,
      summary: summaryFor(games),
      games,
    };
    await writeFile(output, `${JSON.stringify(result, null, 2)}\n`);
  };

  const startIndex = games.length;
  try {
    for (const [relativeIndex, target] of targets.slice(startIndex).entries()) {
      const index = startIndex + relativeIndex;
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
      await persist(false);
      process.stderr.write(`[${index + 1}/${targets.length}] ${target}: ${solved ? history.length : "loss"}\n`);
    }
  } finally {
    bridge.close();
  }
  const summary = summaryFor(games);
  await persist(true);
  console.log(JSON.stringify(summary, null, 2));
}

await main();
