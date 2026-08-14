import { readFile, writeFile } from "node:fs/promises";

type Game = { target: string; solved: boolean; scoredTurns: number };
type Result = { summary: Record<string, unknown>; games: Game[] };

function quantile(values: number[], probability: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor((sorted.length - 1) * probability)];
}

function compare(slm: Result, deepseek: Result) {
  const deepseekByTarget = new Map(deepseek.games.map((game) => [game.target, game]));
  const pairs = slm.games.map((game) => [game, deepseekByTarget.get(game.target)] as const);
  if (pairs.some(([, rival]) => !rival)) throw new Error("benchmark target sets differ");
  let seed = 20260814;
  const random = () => {
    seed = (1664525 * seed + 1013904223) >>> 0;
    return seed / 2 ** 32;
  };
  const winDiffs: number[] = [];
  const turnAdvantages: number[] = [];
  for (let sample = 0; sample < 10_000; sample += 1) {
    let slmWins = 0;
    let rivalWins = 0;
    let turnAdvantage = 0;
    for (let index = 0; index < pairs.length; index += 1) {
      const [small, large] = pairs[Math.floor(random() * pairs.length)];
      slmWins += Number(small.solved);
      rivalWins += Number(large!.solved);
      turnAdvantage += large!.scoredTurns - small.scoredTurns;
    }
    winDiffs.push((slmWins - rivalWins) / pairs.length);
    turnAdvantages.push(turnAdvantage / pairs.length);
  }
  const observedWinDiff =
    (slm.games.filter((game) => game.solved).length - deepseek.games.filter((game) => game.solved).length) /
    pairs.length;
  const observedTurnAdvantage =
    pairs.reduce((sum, [small, large]) => sum + large!.scoredTurns - small.scoredTurns, 0) /
    pairs.length;
  const metric = observedWinDiff === 0 ? "mean_scored_turns" : "win_rate";
  const interval = metric === "win_rate" ? winDiffs : turnAdvantages;
  const ci95 = [quantile(interval, 0.025), quantile(interval, 0.975)];
  return {
    targets: pairs.length,
    observedWinRateDifference: observedWinDiff,
    observedTurnAdvantage: observedTurnAdvantage,
    decisiveMetric: metric,
    ci95,
    slmWins: ci95[0] > 0,
  };
}

async function load(name: string): Promise<Result> {
  return JSON.parse(await readFile(`artifacts/benchmark/${name}.json`, "utf8"));
}

const tracks = {} as Record<string, unknown>;
for (const track of ["pure", "agent"]) {
  tracks[track] = compare(await load(`slm-${track}`), await load(`deepseek-${track}`));
}
const success = (tracks.pure as any).slmWins && (tracks.agent as any).slmWins;
const summary = { success, criterion: "SLM must win with a positive paired 95% bootstrap CI in Pure and Agent", tracks };
await writeFile("artifacts/benchmark/summary.json", `${JSON.stringify(summary, null, 2)}\n`);
await writeFile(
  "artifacts/benchmark/summary.md",
  `# Spanish Wordle SLM benchmark\n\n**Experimental objective achieved:** ${success ? "yes" : "no"}\n\n\`\`\`json\n${JSON.stringify(summary, null, 2)}\n\`\`\`\n`,
);
console.log(JSON.stringify(summary, null, 2));
