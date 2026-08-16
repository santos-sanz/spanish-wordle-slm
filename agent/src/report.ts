import { readFile, writeFile } from "node:fs/promises";

type Game = { target: string; solved: boolean; scoredTurns: number };
type Result = { complete: boolean; targetCount: number; summary: Record<string, unknown>; games: Game[] };

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
  const result = JSON.parse(await readFile(`artifacts/benchmark/${name}.json`, "utf8")) as Result;
  if (!result.complete || result.games.length !== result.targetCount) {
    throw new Error(`${name} benchmark is incomplete (${result.games.length}/${result.targetCount})`);
  }
  return result;
}

const tracks = {} as Record<string, unknown>;
const results = {} as Record<string, Result>;
for (const track of ["pure", "agent"]) {
  results[`slm-${track}`] = await load(`slm-${track}`);
  results[`deepseek-${track}`] = await load(`deepseek-${track}`);
  tracks[track] = compare(results[`slm-${track}`], results[`deepseek-${track}`]);
}
const success = (tracks.pure as any).slmWins && (tracks.agent as any).slmWins;
const summary = { success, criterion: "SLM must win with a positive paired 95% bootstrap CI in Pure and Agent", tracks };
await writeFile("artifacts/benchmark/summary.json", `${JSON.stringify(summary, null, 2)}\n`);

const percent = (value: unknown) => `${(Number(value) * 100).toFixed(1)}%`;
const decimal = (value: unknown, digits = 2) => Number(value).toFixed(digits);
const integer = (value: unknown) => Number(value).toLocaleString("en-US");
const trackRows = ["pure", "agent"].flatMap((track) =>
  ["slm", "deepseek"].map((provider) => {
    const result = results[`${provider}-${track}`];
    const row = result.summary;
    return `| ${track === "pure" ? "Pure" : "Agent"} | ${provider === "slm" ? "Spanish Wordle SLM" : String(row.model)} | ${integer(row.wins)}/${integer(row.games)} | ${percent(row.winRate)} | ${decimal(row.meanScoredTurns)} | ${integer(row.invalidActions)} | ${integer(row.toolCalls)} | $${decimal(row.costUsd, 4)} |`;
  }),
).join("\n");
const comparisonRows = ["pure", "agent"].map((track) => {
  const result = tracks[track] as any;
  return `| ${track === "pure" ? "Pure" : "Agent"} | ${result.decisiveMetric} | ${percent(result.observedWinRateDifference)} | ${decimal(result.observedTurnAdvantage)} | [${percent(result.ci95[0])}, ${percent(result.ci95[1])}] | ${result.slmWins ? "SLM wins" : "not demonstrated"} |`;
}).join("\n");
const rivalModel = String(results["deepseek-pure"].summary.model);
const targetCount = results["slm-pure"].targetCount;
const report = `# Spanish Wordle SLM benchmark

## Technical summary

**Experimental objective achieved:** ${success ? "yes" : "no"}. The frozen MLX adapter was compared with \`${rivalModel}\` on the same ${targetCount} hidden Spanish Wordle targets. A track counts as won only when the paired 95% bootstrap interval is positive for the first metric that differs: win rate first, then mean turns with losses scored as seven.

## Findings

| Track | Model | Wins | Win rate | Mean scored turns | Invalid actions | Tool calls | Cost |
|---|---|---:|---:|---:|---:|---:|---:|
${trackRows}

| Track | Decisive metric | Win-rate difference | Turn advantage | Paired 95% CI | Decision |
|---|---|---:|---:|---:|---|
${comparisonRows}

![Competition dashboard](competition-dashboard.png)

![Cumulative evaluation progress](competition-progress.png)

## Scope and metric definitions

- **Pure:** the model receives only the guess/feedback history and must emit a valid five-letter guess.
- **Agent:** the same loop, with at most one \`get_candidates\` call per turn; no best-move metric is exposed.
- **Win rate:** solved within six turns. A loss contributes seven turns to the mean.
- **Paired decision:** 10,000 bootstrap resamples of the shared target set with seed \`20260814\`.

## Model and experiment

- Local policy: \`LiquidAI/LFM2.5-2.6B-MLX-6bit\` with the frozen selected LoRA adapter and no-thinking inference template.
- Benchmark policy: \`${rivalModel}\`, temperature 0, low reasoning, seed \`20260814\`, 512 output-token cap, and OpenRouter fallback disabled.
- Harness: \`@earendil-works/pi-agent-core@0.84.2\`; at most two invalid-output repairs per turn.
- Source results: \`slm-pure.json\`, \`deepseek-pure.json\`, \`slm-agent.json\`, and \`deepseek-agent.json\`.

## Limitations and robustness

- Statistical uncertainty is reported from paired target-level outcomes; it does not generalize beyond the fixed Spanish answer distribution without further evaluation.
- Network latency and OpenRouter cost are observational and may vary; game outcomes use deterministic decoding but hosted provider infrastructure can still change.
- Checkpoint and prompt decisions were made on validation only. The hidden test was opened after freezing the adapter and benchmark configuration.

## Next steps

${success ? "Publish the frozen adapter manifest and retain this benchmark as the immutable reference run." : "Retrain or revise using training and validation only, freeze a new adapter, then rerun the full hidden test without changing the victory criterion."}
`;
await writeFile(
  "artifacts/benchmark/summary.md",
  report,
);
const generatedAt = new Date().toISOString();
const competition = ["pure", "agent"].flatMap((track) =>
  ["slm", "deepseek"].map((provider) => {
    const row = results[`${provider}-${track}`].summary;
    return {
      track: track === "pure" ? "Pure" : "Agent",
      policy: provider === "slm" ? "Spanish Wordle SLM" : "Benchmark model",
      model: provider === "slm" ? "LFM2.5 2.6B + QLoRA" : String(row.model),
      games: Number(row.games),
      wins: Number(row.wins),
      winRate: Number(row.winRate),
      meanScoredTurns: Number(row.meanScoredTurns),
      invalidActions: Number(row.invalidActions),
      toolCalls: Number(row.toolCalls),
      latencySeconds: Number(row.latencyMs) / 1000,
      inputTokens: Number(row.inputTokens),
      outputTokens: Number(row.outputTokens),
      costUsd: Number(row.costUsd),
    };
  }),
);
const decisions = ["pure", "agent"].map((track) => {
  const decision = tracks[track] as any;
  return {
    track: track === "pure" ? "Pure" : "Agent",
    decisiveMetric: decision.decisiveMetric,
    observedWinRateDifference: decision.observedWinRateDifference,
    observedTurnAdvantage: decision.observedTurnAdvantage,
    ci95Low: decision.ci95[0],
    ci95High: decision.ci95[1],
    decision: decision.slmWins ? "SLM wins" : "Not demonstrated",
  };
});
const artifact = {
  surface: "report",
  manifest: {
    version: 1,
    surface: "report",
    title: "Spanish Wordle SLM competition",
    description: "Paired hidden-test comparison of a local MLX SLM and a larger OpenRouter model.",
    generatedAt,
    cards: [
      {
        id: "pure_slm",
        description: "Frozen SLM performance without tools.",
        dataset: "competition",
        sourceId: "benchmark_results",
        filter: { track: "Pure", policy: "Spanish Wordle SLM" },
        metrics: [
          { label: "Pure win rate", field: "winRate", format: "percent" },
          { label: "Mean turns", field: "meanScoredTurns", format: "number" },
        ],
      },
      {
        id: "agent_slm",
        description: "Frozen SLM performance with candidate lookup.",
        dataset: "competition",
        sourceId: "benchmark_results",
        filter: { track: "Agent", policy: "Spanish Wordle SLM" },
        metrics: [
          { label: "Agent win rate", field: "winRate", format: "percent" },
          { label: "Mean turns", field: "meanScoredTurns", format: "number" },
        ],
      },
    ],
    charts: [
      {
        id: "win_rate",
        title: "Win rate by competitive track",
        subtitle: "Higher is better; every policy receives the same hidden targets.",
        type: "bar",
        dataset: "competition",
        sourceId: "benchmark_results",
        valueFormat: "percent",
        encodings: {
          x: { field: "track", type: "nominal", label: "Track" },
          y: { field: "winRate", type: "quantitative", label: "Win rate" },
          color: { field: "policy", type: "nominal", label: "Policy" },
          tooltip: [
            { field: "wins", type: "quantitative", label: "Wins" },
            { field: "games", type: "quantitative", label: "Games" },
          ],
        },
      },
      {
        id: "mean_turns",
        title: "Mean scored turns",
        subtitle: "Lower is better; unsolved games count as seven turns.",
        type: "bar",
        dataset: "competition",
        sourceId: "benchmark_results",
        encodings: {
          x: { field: "track", type: "nominal", label: "Track" },
          y: { field: "meanScoredTurns", type: "quantitative", label: "Mean turns" },
          color: { field: "policy", type: "nominal", label: "Policy" },
        },
      },
    ],
    tables: [
      {
        id: "competition_table",
        title: "Complete competition summary",
        subtitle: "Outcomes, behavior, latency, tokens, and hosted-model cost.",
        dataset: "competition",
        sourceId: "benchmark_results",
        columns: [
          { field: "track", label: "Track", type: "text" },
          { field: "policy", label: "Policy", type: "text" },
          { field: "wins", label: "Wins", format: "number" },
          { field: "games", label: "Games", format: "number" },
          { field: "winRate", label: "Win rate", format: "percent" },
          { field: "meanScoredTurns", label: "Mean turns", format: "number" },
          { field: "invalidActions", label: "Invalid", format: "number" },
          { field: "toolCalls", label: "Tool calls", format: "number" },
          { field: "latencySeconds", label: "Latency (s)", format: "number" },
          { field: "costUsd", label: "Cost", format: "currency" },
        ],
      },
      {
        id: "decision_table",
        title: "Paired statistical decisions",
        subtitle: "The 95% interval must be strictly positive for the first metric that differs.",
        dataset: "decisions",
        sourceId: "paired_bootstrap",
        columns: [
          { field: "track", label: "Track", type: "text" },
          { field: "decisiveMetric", label: "Metric", type: "text" },
          { field: "observedWinRateDifference", label: "Win-rate diff", format: "percent" },
          { field: "observedTurnAdvantage", label: "Turn advantage", format: "number" },
          { field: "ci95Low", label: "CI low", format: "number" },
          { field: "ci95High", label: "CI high", format: "number" },
          { field: "decision", label: "Decision", type: "text" },
        ],
      },
    ],
    sources: [
      { id: "benchmark_results", label: "Immutable paired benchmark JSON", path: "artifacts/benchmark/*.json" },
      { id: "paired_bootstrap", label: "Seeded paired bootstrap implementation", path: "agent/src/report.ts" },
    ],
    blocks: [
      {
        id: "executive_summary",
        type: "markdown",
        body: `## Technical conclusion\n\n**Objective achieved: ${success ? "yes" : "no"}.** The adapter is judged on Pure and Agent only. Oracle is a non-competitive ceiling.`,
      },
      { id: "metrics", type: "metric-strip", cardIds: ["pure_slm", "agent_slm"] },
      { id: "win_chart", type: "chart", chartId: "win_rate" },
      { id: "turn_chart", type: "chart", chartId: "mean_turns" },
      { id: "decision_detail", type: "table", tableId: "decision_table" },
      {
        id: "methodology",
        type: "markdown",
        body: `## Methodology and controls\n\n- ${targetCount} paired hidden targets per track; deterministic order and seed 20260814.\n- Pure receives history only. Agent may call get_candidates once per turn.\n- At most two invalid-output repairs per turn; a loss scores seven turns.\n- The adapter, prompt, and harness are frozen before opening the hidden test.\n- OpenRouter fallback is disabled; the effective provider and response model are recorded.`,
      },
      { id: "full_results", type: "table", tableId: "competition_table" },
      {
        id: "limitations",
        type: "markdown",
        body: "## Limitations\n\nResults apply to the fixed Spanish Wordle distribution. Hosted latency and cost may vary. Statistical intervals quantify target-level uncertainty, not future provider drift.",
      },
    ],
  },
  snapshot: {
    version: 1,
    generatedAt,
    status: "complete",
    datasets: { competition, decisions },
    accessIssues: [],
  },
  sources: [
    { id: "benchmark_results", description: "Four complete paired benchmark result files." },
    { id: "paired_bootstrap", description: "10,000 resamples with deterministic seed 20260814." },
  ],
  package_info: {
    originUrl: "artifact://spanish-wordle-slm-competition",
    controls: { edit: false, refresh: false },
  },
};
await writeFile(
  "artifacts/benchmark/technical-report.artifact.json",
  `${JSON.stringify(artifact, null, 2)}\n`,
);
console.log(JSON.stringify(summary, null, 2));
