import { readFile, writeFile } from "node:fs/promises";
import { parseArgs } from "node:util";

type GameResult = {
  target: string;
  solved: boolean;
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
};

type BenchmarkFile = {
  complete: boolean;
  targetCount: number;
  targetOffset: number;
  summary: Record<string, unknown>;
  games: GameResult[];
};

const { values } = parseArgs({
  options: {
    prefix: { type: "string" },
    shards: { type: "string" },
    output: { type: "string" },
  },
});
if (!values.prefix || !values.output || !values.shards) {
  throw new Error("--prefix, --shards, and --output are required");
}
for (const value of [values.prefix, values.output]) {
  if (!/^[a-z0-9][a-z0-9-]*$/u.test(value)) throw new Error("invalid artifact name");
}
const shardCount = Number(values.shards);
if (!Number.isSafeInteger(shardCount) || shardCount < 2) throw new Error("invalid shard count");

const shards: BenchmarkFile[] = [];
for (let index = 0; index < shardCount; index += 1) {
  const path = `artifacts/benchmark/${values.prefix}-shard-${index}.json`;
  const shard = JSON.parse(await readFile(path, "utf8")) as BenchmarkFile;
  if (!shard.complete || shard.games.length !== shard.targetCount) {
    throw new Error(`incomplete benchmark shard: ${path}`);
  }
  shards.push(shard);
}
shards.sort((left, right) => left.targetOffset - right.targetOffset);
const reference = shards[0].summary;
let expectedOffset = 0;
for (const shard of shards) {
  if (shard.targetOffset !== expectedOffset) throw new Error("benchmark shards are not contiguous");
  for (const field of ["provider", "model", "track", "split"] as const) {
    if (shard.summary[field] !== reference[field]) throw new Error(`benchmark shard ${field} differs`);
  }
  expectedOffset += shard.targetCount;
}
const splitData = JSON.parse(await readFile("data/processed/splits.json", "utf8"));
const canonical = splitData[String(reference.split)] as string[];
const games = shards.flatMap((shard) => shard.games);
if (!Array.isArray(canonical) || games.length !== canonical.length) {
  throw new Error("benchmark shards do not cover the full split");
}
if (games.some((game, index) => game.target !== canonical[index])) {
  throw new Error("merged benchmark targets differ from canonical order");
}

const summary = {
  provider: reference.provider,
  model: reference.model,
  track: reference.track,
  split: reference.split,
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
const merged: BenchmarkFile = {
  complete: true,
  targetCount: games.length,
  targetOffset: 0,
  summary,
  games,
};
await writeFile(
  `artifacts/benchmark/${values.output}.json`,
  `${JSON.stringify(merged, null, 2)}\n`,
);
console.log(JSON.stringify(summary, null, 2));
