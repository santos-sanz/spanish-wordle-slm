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

# Spanish Wordle LFM2.5 2.6B QLoRA + DPO

Task-specific MLX LoRA adapter trained to play five-letter Spanish Wordle in
Pure, Agent, and Oracle modes. The adapter does not contain the 2.6B base
weights and must be loaded together with
`LiquidAI/LFM2.5-2.6B-MLX-6bit` at revision
`95f71f1c30e3247bc7f042c6fd64d7ca60258780`.

## Training

- Base checkpoint: 6-bit MLX export of LiquidAI LFM2.5 2.6B
- LoRA: rank 8, scale 16, dropout 0.05, final 8 layers
- Optimizer: AdamW, learning rate `2e-5`
- Batch size: 1 with gradient accumulation 8
- Maximum sequence length: 512
- Seed: `20260814`
- Dataset modes: approximately 60% Pure, 30% Agent, 10% Oracle
- Training targets: 431 of 616 unique historical Spanish Wordle answers
- Validation targets: 61; hidden test targets: 124

The selected supervised checkpoint (iteration 2,900, validation cross-entropy
`0.219`) was then optimized with offline Direct Preference Optimization (DPO):

- 1,960 train and 225 validation preference pairs generated only from the
  train/validation Wordle splits
- Chosen action: deterministic solver action; rejected action: a valid action
  with a strictly worse solver ordering
- DPO beta: `0.2`; label smoothing: `0.05`; AdamW learning rate: `1e-6`
- 400 iterations; best checkpoint: iteration 400
- Validation DPO objective: `0.690` before optimization, `0.244` after
- Final validation preference accuracy: `98.75%`; reward margin: `+2.394`
- Peak memory: `3.22 GB`; elapsed DPO time: 21 minutes

The DPO objective is not numerically comparable with the supervised
cross-entropy loss. It measures whether the adapter assigns a larger relative
likelihood to the stronger Wordle action than to the weaker valid action.

The word lists and reproducible training/evaluation code are maintained in the
private repository `santos-sanz/spanish-wordle-slm`. Test targets were excluded
from training trajectory generation.

## Evaluation status

Competitive results remain pending until the adapter, prompts, and generation
configuration are frozen. Success requires a positive paired 95% bootstrap
interval against `deepseek/deepseek-v4-pro-0813` in both Pure and Agent tracks.
Oracle is a solver ceiling and does not count toward that claim.

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
