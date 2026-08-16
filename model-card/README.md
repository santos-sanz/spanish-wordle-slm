---
library_name: mlx
license: other
license_name: lfm1.0
license_link: https://huggingface.co/LiquidAI/LFM2.5-2.6B-MLX-6bit/blob/main/LICENSE
language:
- es
pipeline_tag: text-generation
base_model: LiquidAI/LFM2.5-2.6B-MLX-6bit
tags:
- mlx
- lora
- qlora
- wordle
- spanish
---

# Spanish Wordle LFM2.5 2.6B QLoRA

Task-specific MLX LoRA adapter trained to play five-letter Spanish Wordle in
Pure, Agent, and Oracle modes. The adapter does not contain the 2.6B base
weights and must be loaded together with
`LiquidAI/LFM2.5-2.6B-MLX-6bit` at revision
`95f71f1c30e3247bc7f042c6fd64d7ca60258780`.

## Training

- Base checkpoint: 6-bit MLX export of LiquidAI LFM2.5 2.6B
- LoRA: rank 16, scale 32, dropout 0.05, final 16 layers
- Optimizer: AdamW, learning rate `3e-5` for the main SFT phase
- Batch size: 1 with gradient accumulation 8
- Maximum sequence length: 512
- Seed: `20260814`
- Dataset modes: approximately 60% Pure, 30% Agent, 10% Oracle
- Training targets: 431 of 616 unique historical Spanish Wordle answers
- Validation targets: 61; hidden test targets: 124

The main SFT run used 5,000 optimizer iterations with gradient accumulation of
8. A subsequent Pure-only cleanup phase used 150 iterations at `1e-5` on
8,757 training and 1,261 validation records. Repair examples keep the invalid
guess as runtime context and supervise only the valid replacement, avoiding a
repeated-guess label in the loss. The selected local adapter is the final
checkpoint of that cleanup phase (global iteration 5,355).

The word lists and reproducible training/evaluation code are maintained in the
private repository `santos-sanz/spanish-wordle-slm`. Test targets were excluded
from training trajectory generation.

## Evaluation status

The frozen comparison uses `deepseek/deepseek-v4-flash-0731` through
OpenRouter, temperature 0, seed `20260814`, reasoning low, no provider
fallback, and a 512-token cap. Pure has no tools; Agent may call
`get_candidates` once per turn; Oracle is a solver ceiling and does not count
toward the competitive claim. The final paired bootstrap report in the private
repository is authoritative. Success requires a positive 95% interval for the
SLM in both Pure and Agent; otherwise the run is reported as an experiment that
did not meet the target.

The hidden test produced 2/124 wins for the SLM versus 1/124 for the benchmark
in Pure (paired 95% CI for the win-rate difference: `[-1.6%, +4.0%]`), and
115/124 versus 8/124 in Agent (CI: `[+79.0%, +91.9%]`). The adapter therefore
demonstrates a statistically clear Agent-track win, but this run does not meet
the two-track success claim because Pure is inconclusive.

## Use with MLX-LM

```bash
python -m mlx_lm.generate \
  --model LiquidAI/LFM2.5-2.6B-MLX-6bit \
  --adapter-path santos-sanz/spanish-wordle-lfm2.5-2.6b-qlora \
  --prompt 'Juega Wordle en español.'
```

## License and attribution

This adapter is a derivative of LiquidAI LFM2.5 and is distributed under the
[LFM 1.0 License](https://huggingface.co/LiquidAI/LFM2.5-2.6B-MLX-6bit/blob/main/LICENSE).
Review that license before use or redistribution. LiquidAI is the author of the
base model; this project only provides task-specific adapter weights.
