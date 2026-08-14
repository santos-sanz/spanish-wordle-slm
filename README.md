# Spanish Wordle SLM

A reproducible experiment that fine-tunes `LiquidAI/LFM2.5-2.6B-MLX-6bit`
with QLoRA on Apple Silicon and compares it with
`deepseek/deepseek-v4-pro-0813` on Spanish Wordle.

The experiment has three tracks:

- **Pure**: the model sees only the game history.
- **Agent**: the model may request the compatible answer candidates.
- **Oracle**: the model may request the solver's best guess. This is a system
  ceiling and is not used for the competitive claim.

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

Serve the trained adapter:

```bash
uv run wordle-slm serve
```

Run one paired benchmark track and generate the statistical report:

```bash
uv run wordle-slm benchmark --provider slm --track pure --split test
uv run wordle-slm benchmark --provider deepseek --track pure --split test
npm run report
uv run wordle-slm report
```

Repeat the two benchmark commands for `agent`. The hidden test split must only
be used after the adapter, prompt, and generation configuration are frozen.
The experiment succeeds only if the paired 95% bootstrap interval favors the
SLM in both competitive tracks. Oracle is reported only as a ceiling.

For the remote baseline, copy `.env.example` to `.env` and set
`OPENROUTER_API_KEY`. Never commit `.env`.

## Data and licensing

The processed word lists are derived from pinned public sources documented in
`data/processed/provenance.json`. The historical source contains 617 rows and
616 unique normalized answers because `apoyo` appears twice. The Liquid model
is governed by the LFM Open License v1.0; see the upstream model card before
redistributing derivatives. Base weights, adapters, caches, and credentials are
deliberately excluded from Git.
