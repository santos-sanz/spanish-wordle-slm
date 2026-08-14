from __future__ import annotations

import hashlib
import json
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.tuner.utils import linear_to_lora_layers, print_trainable_parameters

from .data import ROOT
from .preferences import PREFERENCE_DIR, generate_preference_data
from .training import ADAPTER_DIR, MAX_WALL_SECONDS, MODEL_DIR, RUN_DIR

PREFERENCE_RUN_DIR = ROOT / "artifacts" / "runs" / "preference"
REFERENCE_ADAPTER = ADAPTER_DIR / "selected"
DPO_ADAPTER = ADAPTER_DIR / "dpo"


@dataclass(frozen=True)
class PreparedPair:
    chosen: list[int]
    rejected: list[int]
    chosen_start: int
    rejected_start: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _common_prefix(left: list[int], right: list[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def _prepare_pair(tokenizer: Any, record: dict[str, Any]) -> PreparedPair:
    prompt = tokenizer.apply_chat_template(
        record["prompt"], tokenize=True, add_generation_prompt=True
    )

    def completion(value: str) -> tuple[list[int], int]:
        messages = [*record["prompt"], {"role": "assistant", "content": value}]
        tokens = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        start = _common_prefix(prompt, tokens)
        if start >= len(tokens) - 1:
            raise ValueError("chat template did not produce supervised completion tokens")
        if len(tokens) > 512:
            raise ValueError("preference sequence exceeds 512 tokens")
        return tokens, start

    chosen, chosen_start = completion(record["chosen"])
    rejected, rejected_start = completion(record["rejected"])
    return PreparedPair(chosen, rejected, chosen_start, rejected_start)


def _batch(pair: PreparedPair, pad_token: int) -> tuple[mx.array, mx.array]:
    length = max(len(pair.chosen), len(pair.rejected))
    tokens = np.full((2, length), pad_token, dtype=np.int32)
    masks = np.zeros((2, length - 1), dtype=np.float32)
    for row, (values, start) in enumerate(
        ((pair.chosen, pair.chosen_start), (pair.rejected, pair.rejected_start))
    ):
        tokens[row, : len(values)] = values
        masks[row, max(0, start - 1) : len(values) - 1] = 1.0
    return mx.array(tokens), mx.array(masks)


def _completion_logps(model: nn.Module, tokens: mx.array, mask: mx.array) -> mx.array:
    logits = model(tokens[:, :-1])
    targets = tokens[:, 1:]
    token_logps = -nn.losses.cross_entropy(logits, targets, reduction="none")
    return (token_logps * mask).sum(axis=-1)


def _dpo_loss(
    model: nn.Module,
    tokens: mx.array,
    mask: mx.array,
    reference_logps: mx.array,
    beta: float,
    label_smoothing: float,
) -> tuple[mx.array, tuple[mx.array, mx.array, mx.array]]:
    policy_logps = _completion_logps(model, tokens, mask)
    policy_logratio = policy_logps[0] - policy_logps[1]
    reference_logratio = reference_logps[0] - reference_logps[1]
    preference_logit = beta * (policy_logratio - reference_logratio)
    positive_loss = mx.logaddexp(mx.array(0.0), -preference_logit)
    negative_loss = mx.logaddexp(mx.array(0.0), preference_logit)
    loss = (1 - label_smoothing) * positive_loss + label_smoothing * negative_loss
    chosen_reward = beta * (policy_logps[0] - reference_logps[0])
    rejected_reward = beta * (policy_logps[1] - reference_logps[1])
    margin = chosen_reward - rejected_reward
    accuracy = (margin > 0).astype(mx.float32)
    return loss, (margin, accuracy, policy_logratio)


def _load_records(split: str, limit: int) -> list[dict[str, Any]]:
    path = PREFERENCE_DIR / f"{split}.jsonl"
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    random.Random(20260814).shuffle(records)
    return records[:limit]


def _load_policy() -> tuple[nn.Module, Any, dict[str, Any]]:
    if not REFERENCE_ADAPTER.joinpath("adapters.safetensors").exists():
        raise RuntimeError("selected SFT adapter is missing")
    adapter_config = json.loads(
        REFERENCE_ADAPTER.joinpath("adapter_config.json").read_text(encoding="utf-8")
    )
    model, tokenizer = load(str(MODEL_DIR))
    model.freeze()
    linear_to_lora_layers(
        model,
        int(adapter_config["num_layers"]),
        adapter_config["lora_parameters"],
    )
    model.load_weights(str(REFERENCE_ADAPTER / "adapters.safetensors"), strict=False)
    mx.eval(model.parameters())
    return model, tokenizer, adapter_config


def _reference_logps(
    model: nn.Module,
    pairs: list[PreparedPair],
    pad_token: int,
    *,
    deadline: float | None = None,
) -> list[tuple[float, float]]:
    model.eval()
    values: list[tuple[float, float]] = []
    for index, pair in enumerate(pairs, start=1):
        if deadline is not None and time.monotonic() >= deadline - 300:
            raise RuntimeError("preference reference cache exhausted the training budget")
        tokens, mask = _batch(pair, pad_token)
        logps = _completion_logps(model, tokens, mask)
        mx.eval(logps)
        values.append((float(logps[0].item()), float(logps[1].item())))
        if index % 100 == 0:
            print(f"Reference log probabilities: {index}/{len(pairs)}", flush=True)
        mx.clear_cache()
    return values


def _records_sha256(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def _cached_reference_logps(
    model: nn.Module,
    pairs: list[PreparedPair],
    records: list[dict[str, Any]],
    pad_token: int,
    run_name: str,
    deadline: float,
) -> list[tuple[float, float]]:
    key = hashlib.sha256(
        (
            _sha256(REFERENCE_ADAPTER / "adapters.safetensors")
            + _records_sha256(records)
        ).encode()
    ).hexdigest()[:16]
    cache = PREFERENCE_RUN_DIR / f"reference-{run_name}-{key}.npz"
    if cache.exists():
        values = np.load(cache)["values"]
        if values.shape == (len(pairs), 2):
            print(f"Loaded {len(pairs)} cached reference log probabilities", flush=True)
            return [(float(row[0]), float(row[1])) for row in values]
    references = _reference_logps(model, pairs, pad_token, deadline=deadline)
    np.savez_compressed(cache, values=np.asarray(references, dtype=np.float32))
    return references


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _training_elapsed_seconds() -> float:
    state = RUN_DIR / "state.json"
    if not state.exists():
        return 0.0
    return float(json.loads(state.read_text(encoding="utf-8"))["elapsed_seconds"])


def _evaluate(
    model: nn.Module,
    pairs: list[PreparedPair],
    references: list[tuple[float, float]],
    pad_token: int,
    beta: float,
    label_smoothing: float,
) -> tuple[float, float, float]:
    model.eval()
    losses: list[float] = []
    margins: list[float] = []
    accuracies: list[float] = []
    for pair, reference in zip(pairs, references, strict=True):
        tokens, mask = _batch(pair, pad_token)
        loss, (margin, accuracy, _) = _dpo_loss(
            model,
            tokens,
            mask,
            mx.array(reference),
            beta,
            label_smoothing,
        )
        mx.eval(loss, margin, accuracy)
        losses.append(float(loss.item()))
        margins.append(float(margin.item()))
        accuracies.append(float(accuracy.item()))
    model.train()
    return float(np.mean(losses)), float(np.mean(margins)), float(np.mean(accuracies))


def _save_adapter(model: nn.Module, output: Path, config: dict[str, Any], iteration: int) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.trainable_parameters()))
    current = output / "adapters.safetensors"
    checkpoint = output / f"{iteration:07d}_adapters.safetensors"
    mx.save_safetensors(str(current), weights)
    mx.save_safetensors(str(checkpoint), weights)
    output.joinpath("adapter_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )
    return checkpoint


def train_dpo(
    *,
    smoke: bool = False,
    target_iterations: int | None = None,
    resume: bool = False,
    patience: int = 8,
    evaluate_every: int | None = None,
) -> dict[str, Any]:
    if not PREFERENCE_DIR.joinpath("train.jsonl").exists():
        generate_preference_data()
    default_iterations = 5 if smoke else 400
    iterations = target_iterations or default_iterations
    if iterations <= 0:
        raise ValueError("target iterations must be positive")
    train_limit = 12 if smoke else 10_000
    valid_limit = 8 if smoke else 80
    beta = 0.2
    label_smoothing = 0.05
    learning_rate = 1e-6
    report_every = 1 if smoke else 10
    evaluate_every = evaluate_every or (5 if smoke else 50)
    if evaluate_every <= 0:
        raise ValueError("evaluation interval must be positive")
    save_every = evaluate_every
    output = ADAPTER_DIR / ("dpo-smoke" if smoke else "dpo")
    selected_output = ADAPTER_DIR / ("dpo-smoke-selected" if smoke else "dpo-selected")
    PREFERENCE_RUN_DIR.mkdir(parents=True, exist_ok=True)
    run_name = "dpo-smoke" if smoke else "dpo"
    log_path = PREFERENCE_RUN_DIR / f"{run_name}.log"
    metrics_path = PREFERENCE_RUN_DIR / f"{run_name}.metrics.jsonl"
    result_path = PREFERENCE_RUN_DIR / f"{run_name}.json"
    state_path = PREFERENCE_RUN_DIR / f"{run_name}.state.json"
    previous_result: dict[str, Any] = {}
    previous_elapsed = 0.0
    start_iteration = 0
    if resume:
        if not result_path.exists() or not selected_output.joinpath(
            "adapters.safetensors"
        ).exists():
            raise RuntimeError("a selected DPO adapter is required to resume")
        previous_result = json.loads(result_path.read_text(encoding="utf-8"))
        start_iteration = int(previous_result["iterations"])
        previous_elapsed = float(previous_result.get("elapsed_seconds", 0.0))
        if iterations <= start_iteration:
            raise ValueError(
                f"target iterations ({iterations}) must exceed resume iteration "
                f"({start_iteration})"
            )
    else:
        shutil.rmtree(output, ignore_errors=True)

    remaining_budget = (
        MAX_WALL_SECONDS - _training_elapsed_seconds() - previous_elapsed
        if not smoke
        else 1800.0
    )
    if remaining_budget < 600:
        raise RuntimeError("less than ten minutes remain in the cumulative training budget")
    stage_started = time.monotonic()
    deadline = stage_started + remaining_budget
    state = {
        "status": "running",
        "resume": resume,
        "start_iteration": start_iteration,
        "iterations_planned": iterations,
        "patience": patience,
        "evaluate_every": evaluate_every,
        "remaining_budget_seconds_at_start": remaining_budget,
    }
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    model, tokenizer, adapter_config = _load_policy()
    print_trainable_parameters(model)
    pad_token = int(tokenizer.pad_token_id or tokenizer.eos_token_id)
    train_records = _load_records("train", train_limit)
    valid_records = _load_records("valid", valid_limit)
    train_pairs = [_prepare_pair(tokenizer, record) for record in train_records]
    valid_pairs = [_prepare_pair(tokenizer, record) for record in valid_records]
    all_pairs = train_pairs + valid_pairs
    references = _cached_reference_logps(
        model,
        all_pairs,
        train_records + valid_records,
        pad_token,
        run_name,
        deadline,
    )
    train_references = references[: len(train_pairs)]
    valid_references = references[len(train_pairs) :]
    if resume:
        model.load_weights(
            str(selected_output / "adapters.safetensors"), strict=False
        )
        mx.eval(model.parameters())
        print(f"Resumed policy weights from iteration {start_iteration}", flush=True)

    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=0.01)
    loss_and_grad = nn.value_and_grad(model, _dpo_loss)
    rng = random.Random(20260814)
    order = list(range(len(train_pairs)))
    metrics = _load_metrics(metrics_path) if resume else []
    best: tuple[float, int, Path] | None = None
    for row in metrics:
        if row.get("metric") != "dpo_validation":
            continue
        checkpoint = output / f"{int(row['iteration']):07d}_adapters.safetensors"
        candidate = (float(row["loss"]), int(row["iteration"]), checkpoint)
        if checkpoint.exists() and (best is None or candidate[0] < best[0]):
            best = candidate
    stale_evaluations = 0
    if best is not None:
        stale_evaluations = sum(
            1
            for row in metrics
            if row.get("metric") == "dpo_validation"
            and int(row["iteration"]) > best[1]
        )
    completed_iteration = start_iteration
    stop_reason = "target_reached"
    model.train()
    completed_epochs = start_iteration // len(order)
    for _ in range(completed_epochs + 1):
        rng.shuffle(order)
    with log_path.open("a" if resume else "w", encoding="utf-8") as log:
        for iteration in range(start_iteration + 1, iterations + 1):
            if iteration > start_iteration + 1 and (iteration - 1) % len(order) == 0:
                rng.shuffle(order)
            pair_index = order[(iteration - 1) % len(order)]
            tokens, mask = _batch(train_pairs[pair_index], pad_token)
            reference = mx.array(train_references[pair_index])
            (loss, (margin, accuracy, policy_logratio)), gradients = loss_and_grad(
                model,
                tokens,
                mask,
                reference,
                beta,
                label_smoothing,
            )
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss, margin, accuracy, policy_logratio)
            if iteration % report_every == 0:
                row = {
                    "iteration": iteration,
                    "metric": "dpo_train",
                    "loss": float(loss.item()),
                    "reward_margin": float(margin.item()),
                    "reward_accuracy": float(accuracy.item()),
                    "policy_logratio": float(policy_logratio.item()),
                    "peak_memory_gb": mx.get_peak_memory() / 1e9,
                }
                metrics.append(row)
                print(json.dumps(row), file=log, flush=True)
            if iteration == 1 or iteration % evaluate_every == 0 or iteration == iterations:
                val_loss, val_margin, val_accuracy = _evaluate(
                    model,
                    valid_pairs,
                    valid_references,
                    pad_token,
                    beta,
                    label_smoothing,
                )
                row = {
                    "iteration": iteration,
                    "metric": "dpo_validation",
                    "loss": val_loss,
                    "reward_margin": val_margin,
                    "reward_accuracy": val_accuracy,
                }
                metrics.append(row)
                print(json.dumps(row), file=log, flush=True)
            if iteration % save_every == 0 or iteration == iterations:
                config = {
                    **adapter_config,
                    "method": "dpo",
                    "adapter_path": "adapters/dpo-selected",
                    "config": "configs/lora-reference.yaml",
                    "data": "data/training",
                    "model": "LiquidAI/LFM2.5-2.6B-MLX-6bit",
                    "reference_adapter": "adapters/selected",
                    "beta": beta,
                    "label_smoothing": label_smoothing,
                    "learning_rate": learning_rate,
                    "preference_seed": 20260814,
                    "preference_manifest": "data/preferences/manifest.json",
                }
                config.pop("resume_adapter_file", None)
                checkpoint = _save_adapter(model, output, config, iteration)
                validation_rows = [
                    row for row in metrics if row["metric"] == "dpo_validation"
                ]
                latest_validation = validation_rows[-1]
                candidate = (float(latest_validation["loss"]), iteration, checkpoint)
                if best is None or candidate[0] < best[0] - 1e-4:
                    best = candidate
                    stale_evaluations = 0
                else:
                    stale_evaluations += 1
                state.update(
                    {
                        "completed_iteration": iteration,
                        "best_iteration": best[1],
                        "best_validation_loss": best[0],
                        "stale_evaluations": stale_evaluations,
                        "elapsed_seconds": previous_elapsed
                        + time.monotonic()
                        - stage_started,
                    }
                )
                state_path.write_text(
                    json.dumps(state, indent=2) + "\n", encoding="utf-8"
                )
                if patience > 0 and stale_evaluations >= patience:
                    stop_reason = "early_stopping"
                elif time.monotonic() >= deadline - 300:
                    stop_reason = "budget_exhausted"
                if stop_reason != "target_reached":
                    completed_iteration = iteration
                    break
            completed_iteration = iteration
            mx.clear_cache()

    metrics_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in metrics),
        encoding="utf-8",
    )
    if best is None:
        raise RuntimeError("DPO training did not produce a checkpoint")
    shutil.rmtree(selected_output, ignore_errors=True)
    selected_output.mkdir(parents=True)
    shutil.copy2(best[2], selected_output / "adapters.safetensors")
    shutil.copy2(output / "adapter_config.json", selected_output / "adapter_config.json")
    result = {
        "method": "dpo",
        "mode": "smoke" if smoke else "full",
        "iterations": completed_iteration,
        "target_iterations": iterations,
        "resumed_from_iteration": start_iteration if resume else None,
        "optimizer_state_resumed": False,
        "stop_reason": stop_reason,
        "elapsed_seconds": previous_elapsed + time.monotonic() - stage_started,
        "stage_elapsed_seconds": time.monotonic() - stage_started,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "best_validation_loss": best[0],
        "best_iteration": best[1],
        "reference_adapter_sha256": _sha256(
            REFERENCE_ADAPTER / "adapters.safetensors"
        ),
        "adapter_path": str(output.relative_to(ROOT)),
        "selected_adapter_path": str(selected_output.relative_to(ROOT)),
        "train_pairs": len(train_pairs),
        "validation_pairs": len(valid_pairs),
    }
    result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    state.update(
        {
            "status": stop_reason,
            "completed_iteration": completed_iteration,
            "best_iteration": best[1],
            "best_validation_loss": best[0],
            "elapsed_seconds": result["elapsed_seconds"],
        }
    )
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return result
