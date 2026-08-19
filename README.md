# Spanish Wordle SLM

A reproducible experiment that fine-tunes `LiquidAI/LFM2.5-2.6B-MLX-6bit`
with QLoRA on Apple Silicon and compares it first with
`deepseek/deepseek-v4-flash-0731` on Spanish Wordle. The Pro model remains
deferred until the SLM beats Flash under the frozen criterion.

The experiment has three tracks:

- **Pure**: the model sees only the game history.
- **Agent**: the model may request the compatible answer candidates.
- **Oracle**: the model may request the solver's best guess. This is a system
  ceiling and is not used for the competitive claim.

## Documentation and current evidence

The project has two complementary records:

- [Experimental retrospective](docs/experimental-retrospective.md): the
  end-to-end article covering objectives, architecture, training decisions,
  failed branches, quota failures, benchmark results and lessons learned.
- [Experimental learning log](docs/experimental-learning-log.md): the
  chronological audit log to read before changing a prompt, parser, dataset,
  adapter or benchmark.

The current clean Flash comparison uses the same 124 hidden targets in both
competitive tracks and has `errors=0`:

| Track | SLM | Flash | Statistical status |
|---|---:|---:|---|
| Pure | 2/124 | 0/124 | Not demonstrated: paired 95% CI `[0.0, 4.0]` pp |
| Agent | 115/124 | 46/124 | SLM wins: paired 95% CI `[46.0, 65.3]` pp |

The project therefore remains an honest `success=false`: Agent is a strong
result, but the two-track claim still requires Pure to clear the positive
paired interval. Pro, Tencent HY3 and GPT-5.6 Luna runs are not accepted as
comparisons because their remote smokes or full runs were blocked by routing,
provider errors or insufficient credit.

Visual evidence is kept next to the machine-readable artifacts:

- [Training loss curve](artifacts/training/loss-curve.png) ·
  [SVG](artifacts/training/loss-curve.svg)
- [Flash competition dashboard](artifacts/benchmark/competition-dashboard-flash.png) ·
  [SVG](artifacts/benchmark/competition-dashboard-flash.svg)
- [Flash cumulative progress](artifacts/benchmark/competition-progress-flash.png) ·
  [SVG](artifacts/benchmark/competition-progress-flash.svg)
- [Flash report](artifacts/benchmark/summary-flash.md)

## Reproduce

```bash
uv sync --extra dev
uv run wordle-slm prepare-data
uv run wordle-slm validate
uv run pytest
npm install
npm test
```

Download the pinned model and run a training smoke test:

```bash
uv run wordle-slm download-model
uv run wordle-slm train --smoke
```

Start the time-bounded training run (maximum cumulative wall time: 5h45m):

```bash
caffeinate -dimsu uv run wordle-slm train
```

Generate the loss dashboard once, or keep it refreshed while training:

```bash
uv run wordle-slm visualize-training
uv run wordle-slm visualize-training --watch --interval 15
```

The PNG, SVG, machine-readable status, and normalized metric history are written
under `artifacts/training/`.

After selecting the best SFT checkpoint, build deterministic Wordle preference
pairs and run the memory-efficient DPO phase:

```bash
uv run wordle-slm prepare-preferences
uv run wordle-slm train-preference --smoke
caffeinate -dimsu uv run wordle-slm train-preference
uv run wordle-slm visualize-preference --watch --interval 15
```

Continue the selected DPO policy to a larger total iteration target while the
original SFT adapter remains the fixed reference:

```bash
caffeinate -dimsu uv run wordle-slm train-preference \
  --resume --iterations 3000 --evaluate-every 100 --patience 8
```

The continuation uses all 1,960 training preference pairs, never reads the
hidden test split, caches fixed-reference log probabilities, respects the
cumulative 5 h 45 min training budget, and stops early after eight validation
checks without a material improvement. The optimizer state cannot be recovered
from the already completed 400-iteration run, so the continuation resumes the
policy weights with a fresh AdamW state at the same `1e-6` learning rate.

To restart DPO from the selected SFT checkpoint with a theoretical loss floor
of zero, omit `--resume` and disable preference label smoothing explicitly:

```bash
caffeinate -dimsu uv run wordle-slm train-preference \
  --iterations 3000 --evaluate-every 100 --patience 8 --label-smoothing 0
```

With `label_smoothing=0`, zero is an asymptotic infimum rather than a value that
finite model weights are guaranteed to attain. Validation checkpoint selection,
early stopping, and the cumulative wall-clock watchdog remain authoritative.

DPO uses the SFT checkpoint at iteration 2,900 as a fixed reference and directly
increases the probability margin of the solver's action over a weaker valid
action. It is an offline preference-optimization alternative to GRPO: MLX-LM
0.31.3 does not ship a native GRPO trainer, and this approach avoids keeping a
second 2.6B model in unified memory. SFT and DPO objectives are not numerically
comparable, so their curves are reported separately.

Serve the trained adapter:

```bash
uv run wordle-slm serve  # serves adapters/selected unless WORDLE_ADAPTER_PATH is set
```

The server selects `adapters/selected` by default and falls back to the
latest SFT adapter when no selected checkpoint exists. Set `WORDLE_ADAPTER_PATH`
to override it explicitly. It also uses the tracked no-thinking inference chat
template so generation matches the response-only format used during training.
The local request also suppresses LFM's `<think>` opener; otherwise reasoning
can consume the full output budget before the trained JSON answer is emitted.

Run one paired benchmark track and generate the statistical report:

```bash
uv run wordle-slm benchmark --provider slm --track pure --split test
uv run wordle-slm benchmark --provider deepseek --track pure --split test
uv run wordle-slm benchmark --provider slm --track oracle --split test --output-name oracle-ceiling
npm run report
uv run wordle-slm report
uv run wordle-slm visualize-benchmark
```

Long benchmarks persist after every game and can continue safely with
`--resume`; a resume is rejected if model, track, split, target count, or target
prefix differs from the checkpoint. Remote runs can be split with `--offset`,
`--limit`, and distinct `--output-name` values, then merged with the checked-in
`npm run merge-benchmark` command; the merger rejects gaps, overlaps, model
drift, and non-canonical target order.

Repeat the two benchmark commands for `agent`. The hidden test split must only
be used after the adapter, prompt, and generation configuration are frozen.
The experiment succeeds only if the paired 95% bootstrap interval favors the
SLM in both competitive tracks. Oracle is reported only as a ceiling.

For the remote baseline, copy `.env.example` to `.env` and set
`OPENROUTER_API_KEY`. The checked-in example pins the current rival,
`deepseek/deepseek-v4-flash-0731`, with fallbacks disabled by the harness. Do
not switch to `deepseek/deepseek-v4-pro-0813` until Flash has been beaten under
the frozen criterion. Never commit `.env`.

## Data and licensing

The processed word lists are derived from pinned public sources documented in
`data/processed/provenance.json`. The historical source contains 617 rows and
616 unique normalized answers because `apoyo` appears twice. The Liquid model
is governed by the LFM Open License v1.0; see the upstream model card before
redistributing derivatives. Base weights, adapters, caches, and credentials are
deliberately excluded from Git.
