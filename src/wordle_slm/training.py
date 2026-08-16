from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from huggingface_hub import snapshot_download

from .data import ROOT, TRAINING_DIR

MODEL_ID = "LiquidAI/LFM2.5-2.6B-MLX-6bit"
MODEL_REVISION = "95f71f1c30e3247bc7f042c6fd64d7ca60258780"
MODEL_DIR = ROOT / "models" / "LFM2.5-2.6B-MLX-6bit"
ADAPTER_DIR = ROOT / "adapters"
INFERENCE_CHAT_TEMPLATE = ROOT / "configs" / "chat_template_no_think.jinja"
RUN_DIR = ROOT / "artifacts" / "runs" / "training"
MAX_WALL_SECONDS = 5 * 3600 + 45 * 60


@dataclass(frozen=True)
class RunResult:
    elapsed_seconds: float
    timed_out: bool
    last_iteration: int
    peak_memory_gb: float


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model() -> dict[str, Any]:
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=MODEL_DIR,
    )
    weight = MODEL_DIR / "model.safetensors"
    if not weight.exists() or weight.stat().st_size != 2_191_851_544:
        raise RuntimeError("downloaded model weight has an unexpected size")
    manifest = {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "weight_file": weight.name,
        "weight_size": weight.stat().st_size,
        "weight_sha256": _sha256(weight),
    }
    MODEL_DIR.joinpath("local-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    tracked = ROOT / "artifacts" / "model-manifest.json"
    tracked.parent.mkdir(parents=True, exist_ok=True)
    tracked.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _base_config(*, adapter_path: Path, iters: int, seq_length: int, layers: int) -> dict[str, Any]:
    return {
        "model": str(MODEL_DIR),
        "train": True,
        "fine_tune_type": "lora",
        "optimizer": "adamw",
        "data": str(TRAINING_DIR),
        "seed": 20260814,
        "num_layers": layers,
        "batch_size": 1,
        "iters": iters,
        "val_batches": 20,
        "learning_rate": 2e-5,
        "steps_per_report": 5,
        "steps_per_eval": 50,
        "grad_accumulation_steps": 8,
        "adapter_path": str(adapter_path),
        "save_every": 50,
        "max_seq_length": seq_length,
        "mask_prompt": True,
        "grad_checkpoint": True,
        "lora_parameters": {"rank": 8, "scale": 16.0, "dropout": 0.05},
    }


def _parse_metrics(log_text: str, name: str) -> tuple[list[dict[str, Any]], int, float]:
    metrics: list[dict[str, Any]] = []
    last_iteration = 0
    peak_memory = 0.0
    for line in log_text.splitlines():
        validation = re.search(r"Iter (\d+): Val loss ([0-9.]+)", line)
        if validation:
            iteration = int(validation.group(1))
            last_iteration = max(last_iteration, iteration)
            metrics.append(
                {
                    "run": name,
                    "iteration": iteration,
                    "metric": "validation_loss",
                    "value": float(validation.group(2)),
                }
            )
        training = re.search(
            r"Iter (\d+): Train loss ([0-9.]+).*Peak mem ([0-9.]+) GB", line
        )
        if training:
            iteration = int(training.group(1))
            memory = float(training.group(3))
            last_iteration = max(last_iteration, iteration)
            peak_memory = max(peak_memory, memory)
            metrics.append(
                {
                    "run": name,
                    "iteration": iteration,
                    "metric": "train_loss",
                    "value": float(training.group(2)),
                    "peak_memory_gb": memory,
                }
            )
    return metrics, last_iteration, peak_memory


def _run_lora(config: dict[str, Any], name: str, timeout: float | None = None) -> RunResult:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    config_path = RUN_DIR / f"{name}.yaml"
    log_path = RUN_DIR / f"{name}.log"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    started = time.monotonic()
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "mlx_lm", "lora", "--config", str(config_path)],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            return_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return_code = process.returncode
    elapsed = time.monotonic() - started
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    metrics, last_iteration, peak_memory = _parse_metrics(log_text, name)
    metrics_path = RUN_DIR / f"{name}.metrics.jsonl"
    metrics_path.write_text(
        "".join(json.dumps(metric, sort_keys=True) + "\n" for metric in metrics),
        encoding="utf-8",
    )
    if return_code != 0 and not timed_out:
        tail = log_text[-4000:]
        raise RuntimeError(f"MLX LoRA failed ({return_code}):\n{tail}")
    if re.search(r"(?:train|val) loss nan", log_text, flags=re.IGNORECASE):
        raise RuntimeError("MLX LoRA produced a NaN loss; refusing to accept the run")
    return RunResult(elapsed, timed_out, last_iteration, peak_memory)


def _select_lowest_loss_checkpoint(run_name: str, adapter_path: Path) -> dict[str, Any]:
    metrics_path = RUN_DIR / f"{run_name}.metrics.jsonl"
    validation = [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line and json.loads(line)["metric"] == "validation_loss"
    ]
    available: list[tuple[float, int, Path]] = []
    for metric in validation:
        iteration = int(metric["iteration"])
        checkpoint = adapter_path / f"{iteration:07d}_adapters.safetensors"
        if checkpoint.exists():
            available.append((float(metric["value"]), iteration, checkpoint))
    if available:
        loss, iteration, checkpoint = min(available)
    else:
        loss = float(validation[-1]["value"]) if validation else float("nan")
        iteration = int(validation[-1]["iteration"]) if validation else 0
        checkpoint = adapter_path / "adapters.safetensors"
    selected = ADAPTER_DIR / "selected"
    selected.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint, selected / "adapters.safetensors")
    shutil.copy2(adapter_path / "adapter_config.json", selected / "adapter_config.json")
    return {
        "selection": "minimum_validation_loss",
        "iteration": iteration,
        "validation_loss": loss,
        "source": str(checkpoint),
        "adapter_path": str(selected),
    }


def train(*, smoke: bool = False) -> dict[str, Any]:
    if not MODEL_DIR.joinpath("model.safetensors").exists():
        raise RuntimeError("model is not downloaded; run download-model first")
    if not TRAINING_DIR.joinpath("train.jsonl").exists():
        raise RuntimeError("training data is missing; run prepare-data first")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    if smoke:
        path = ADAPTER_DIR / "smoke"
        shutil.rmtree(path, ignore_errors=True)
        result = _run_lora(
            _base_config(adapter_path=path, iters=20, seq_length=512, layers=8), "smoke"
        )
        if result.timed_out:
            raise RuntimeError("the smoke test exceeded its time limit")
        return {
            "mode": "smoke",
            "elapsed_seconds": result.elapsed_seconds,
            "peak_memory_gb": result.peak_memory_gb,
            "adapter_path": str(path),
        }

    state_path = RUN_DIR / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"elapsed_seconds": 0.0}
    remaining = MAX_WALL_SECONDS - float(state["elapsed_seconds"])
    if remaining < 600:
        raise RuntimeError("less than ten minutes remain in the cumulative training budget")

    calibration_path = ADAPTER_DIR / "calibration"
    if "sequence_length" not in state:
        calibration: RunResult | None = None
        for candidate_seq, candidate_layers in ((512, 8), (384, 8), (384, 4)):
            shutil.rmtree(calibration_path, ignore_errors=True)
            try:
                candidate = _run_lora(
                    _base_config(
                        adapter_path=calibration_path,
                        iters=20,
                        seq_length=candidate_seq,
                        layers=candidate_layers,
                    ),
                    f"calibration-{candidate_seq}-{candidate_layers}",
                    timeout=min(remaining - 300, 1800),
                )
                if candidate.timed_out or candidate.peak_memory_gb > 14.0:
                    continue
                calibration = candidate
                state.update(
                    {
                        "sequence_length": candidate_seq,
                        "layers": candidate_layers,
                        "seconds_per_iteration": candidate.elapsed_seconds / 20,
                        "peak_memory_gb": candidate.peak_memory_gb,
                    }
                )
                break
            except RuntimeError as error:
                if "memory" not in str(error).lower() and "metal" not in str(error).lower():
                    raise
        if calibration is None:
            raise RuntimeError("all calibration configurations failed")
        state["elapsed_seconds"] = float(state["elapsed_seconds"]) + calibration.elapsed_seconds
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    seq_length = int(state["sequence_length"])
    layers = int(state["layers"])
    seconds_per_iter = float(state["seconds_per_iteration"])
    remaining = MAX_WALL_SECONDS - float(state["elapsed_seconds"])
    target_seconds = max(300.0, remaining - 300.0)
    state.setdefault("iterations_planned", max(50, min(3000, int(target_seconds / seconds_per_iter))))
    completed_iterations = int(state.get("completed_iterations", 0))
    iters = int(state["iterations_planned"]) - completed_iterations
    if iters <= 0:
        state["completed"] = True
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        return state
    final_path = ADAPTER_DIR / "final"
    if completed_iterations == 0:
        shutil.rmtree(final_path, ignore_errors=True)
    config = _base_config(
        adapter_path=final_path,
        iters=iters,
        seq_length=seq_length,
        layers=layers,
    )
    resume_file = final_path / "adapters.safetensors"
    config["resume_adapter_file"] = str(
        resume_file if resume_file.exists() else calibration_path / "adapters.safetensors"
    )
    run_index = int(state.get("run_index", 0)) + 1
    run_name = f"full-{run_index:02d}"
    result = _run_lora(config, run_name, timeout=target_seconds)
    state["elapsed_seconds"] = float(state["elapsed_seconds"]) + result.elapsed_seconds
    persisted_iterations = (
        (result.last_iteration // 50) * 50 if result.timed_out else iters
    )
    state["completed_iterations"] = completed_iterations + persisted_iterations
    state.update(
        {
            "completed": not result.timed_out,
            "run_index": run_index,
            "adapter_path": str(final_path),
            "last_run_timed_out": result.timed_out,
        }
    )
    if not result.timed_out:
        state["selected_checkpoint"] = _select_lowest_loss_checkpoint(run_name, final_path)
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return state


def serve() -> None:
    configured_adapter = os.environ.get("WORDLE_ADAPTER_PATH")
    adapter = (
        Path(configured_adapter).expanduser()
        if configured_adapter
        else ADAPTER_DIR / "dpo-selected"
    )
    if not adapter.is_absolute():
        adapter = ROOT / adapter
    if not configured_adapter and not adapter.joinpath("adapters.safetensors").exists():
        adapter = ADAPTER_DIR / "final"
    if not adapter.joinpath("adapters.safetensors").exists():
        raise RuntimeError(f"adapter is not available: {adapter}")
    os.execvp(
        sys.executable,
        [
            sys.executable,
            "-m",
            "mlx_lm",
            "server",
            "--model",
            str(MODEL_DIR),
            "--adapter-path",
            str(adapter),
            "--chat-template",
            INFERENCE_CHAT_TEMPLATE.read_text(encoding="utf-8"),
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
        ],
    )
