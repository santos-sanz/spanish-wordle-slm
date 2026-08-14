from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from typing import Any

import matplotlib
import numpy as np

from .data import ROOT
from .preference_training import PREFERENCE_RUN_DIR

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUTPUT_DIR = ROOT / "artifacts" / "preference"


def _load_metrics(smoke: bool = False) -> list[dict[str, Any]]:
    name = "dpo-smoke" if smoke else "dpo"
    log_path = PREFERENCE_RUN_DIR / f"{name}.log"
    metrics_path = PREFERENCE_RUN_DIR / f"{name}.metrics.jsonl"
    path = log_path if log_path.exists() else metrics_path
    if not path.exists():
        raise RuntimeError(f"preference metrics are not available: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if value.get("metric") in {"dpo_train", "dpo_validation"}:
            rows.append(value)
    if not rows:
        raise RuntimeError("preference training has not emitted metrics yet")
    return rows


def _ewma(values: list[float], alpha: float = 0.2) -> np.ndarray:
    if not values:
        return np.array([])
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1 - alpha) * result[-1])
    return np.asarray(result)


def _card(fig: plt.Figure, x: float, label: str, value: str, accent: str) -> None:
    box = FancyBboxPatch(
        (x, 0.82),
        0.205,
        0.09,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=fig.transFigure,
        facecolor="#F7F8FA",
        edgecolor="#E2E6EA",
        linewidth=0.8,
    )
    fig.add_artist(box)
    fig.text(x + 0.015, 0.875, value, fontsize=17, fontweight="bold", color=accent)
    fig.text(x + 0.015, 0.838, label, fontsize=9, color="#68707A")


def render_preference_dashboard(*, smoke: bool = False) -> dict[str, Any]:
    rows = _load_metrics(smoke)
    train = [row for row in rows if row["metric"] == "dpo_train"]
    validation = [row for row in rows if row["metric"] == "dpo_validation"]
    name = "dpo-smoke" if smoke else "dpo"
    state_path = PREFERENCE_RUN_DIR / f"{name}.state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {}
    )
    planned = int(state.get("iterations_planned", 5 if smoke else 400))
    run_status = str(state.get("status", "unknown"))
    current = max(int(row["iteration"]) for row in rows)
    progress = min(current / planned, 1.0)
    best = min(validation, key=lambda row: float(row["loss"]))
    latest_validation = validation[-1]
    train_iterations = [int(row["iteration"]) for row in train]
    train_losses = [float(row["loss"]) for row in train]
    smoothed = _ewma(train_losses)
    generated_at = datetime.now(UTC).isoformat()
    peak_memory = max((float(row.get("peak_memory_gb", 0)) for row in train), default=0)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#C9CFD6",
            "axes.labelcolor": "#3B424A",
            "xtick.color": "#68707A",
            "ytick.color": "#68707A",
        }
    )
    fig = plt.figure(figsize=(14, 8.5), facecolor="#FFFFFF")
    fig.text(
        0.065,
        0.965,
        "Spanish Wordle SLM — Preference optimization",
        fontsize=22,
        fontweight="bold",
        color="#20252B",
    )
    fig.text(
        0.065,
        0.935,
        "DPO after SFT checkpoint 2,900 · optimal Wordle action preferred over a weaker valid action",
        fontsize=10.5,
        color="#68707A",
    )
    _card(fig, 0.065, "PROGRESS", f"{current:,} / {planned:,}  ({progress:.1%})", "#155EEF")
    _card(
        fig,
        0.29,
        "BEST VALIDATION OBJECTIVE",
        f"{float(best['loss']):.3f}  @ {int(best['iteration']):,}",
        "#A15C00",
    )
    _card(
        fig,
        0.515,
        "VALIDATION PREFERENCE ACCURACY",
        f"{float(latest_validation['reward_accuracy']):.1%}",
        "#344054",
    )
    _card(
        fig,
        0.74,
        "VALIDATION REWARD MARGIN",
        f"{float(latest_validation['reward_margin']):+.3f}",
        "#344054",
    )

    axis = fig.add_axes((0.08, 0.18, 0.86, 0.57))
    axis.set_facecolor("#FFFFFF")
    axis.grid(axis="y", color="#E8EBEF", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.axhline(
        math.log(2),
        color="#8A929C",
        linestyle=":",
        linewidth=1.1,
        label="Unchanged-policy baseline · ln(2)",
    )
    axis.plot(
        train_iterations,
        train_losses,
        color="#87AFFF",
        linewidth=0.9,
        alpha=0.45,
        marker="o",
        markersize=2.8,
        label="DPO train objective (raw)",
    )
    axis.plot(
        train_iterations,
        smoothed,
        color="#155EEF",
        linewidth=2.4,
        label="DPO train objective (EWMA)",
    )
    axis.plot(
        [int(row["iteration"]) for row in validation],
        [float(row["loss"]) for row in validation],
        color="#C97900",
        linewidth=1.7,
        linestyle="--",
        marker="s",
        markersize=5.5,
        markerfacecolor="#FFFFFF",
        markeredgewidth=1.5,
        label="DPO validation objective",
    )
    axis.annotate(
        f"best {float(best['loss']):.3f}",
        xy=(int(best["iteration"]), float(best["loss"])),
        xytext=(10, 18),
        textcoords="offset points",
        fontsize=9,
        color="#7A4500",
        arrowprops={"arrowstyle": "-", "color": "#A15C00", "linewidth": 0.9},
    )
    axis.set_ylim(bottom=0, top=max(0.76, max(train_losses) * 1.08))
    axis.set_xlim(left=0, right=max(current * 1.05, planned * 0.05))
    axis.set_xlabel("Preference optimizer iteration", fontsize=10, labelpad=10)
    axis.set_ylabel("DPO objective (lower is better)", fontsize=10, labelpad=10)
    axis.legend(loc="upper right", frameon=False, ncols=2, fontsize=9)

    progress_axis = fig.add_axes((0.08, 0.09, 0.86, 0.035))
    progress_axis.barh([0], [1], color="#EDF0F3", height=0.55)
    progress_axis.barh([0], [progress], color="#155EEF", height=0.55)
    progress_axis.set_xlim(0, 1)
    progress_axis.axis("off")
    fig.text(0.08, 0.065, "0", fontsize=8, color="#7B8490")
    fig.text(0.94, 0.065, f"{planned:,}", fontsize=8, color="#7B8490", ha="right")
    fig.text(
        0.5,
        0.035,
        f"Source: deterministic Wordle preference pairs · updated {generated_at[:19].replace('T', ' ')} UTC",
        fontsize=8,
        color="#8A929C",
        ha="center",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = "dpo-smoke" if smoke else "dpo"
    png_path = OUTPUT_DIR / f"{prefix}-curves.png"
    svg_path = OUTPUT_DIR / f"{prefix}-curves.svg"
    fig.savefig(png_path, dpi=170, facecolor=fig.get_facecolor())
    fig.savefig(svg_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    summary = {
        "generated_at": generated_at,
        "iteration": current,
        "iterations_planned": planned,
        "run_status": run_status,
        "best_validation_loss": float(best["loss"]),
        "best_validation_iteration": int(best["iteration"]),
        "validation_reward_accuracy": float(latest_validation["reward_accuracy"]),
        "validation_reward_margin": float(latest_validation["reward_margin"]),
        "peak_memory_gb": peak_memory,
        "png": str(png_path.relative_to(ROOT)),
        "svg": str(svg_path.relative_to(ROOT)),
    }
    OUTPUT_DIR.joinpath(f"{prefix}-status.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def watch_preference_dashboard(interval_seconds: float = 15.0) -> None:
    try:
        while True:
            summary = render_preference_dashboard()
            print(
                f"iteration={summary['iteration']}/{summary['iterations_planned']} "
                f"best={summary['best_validation_loss']:.3f} "
                f"accuracy={summary['validation_reward_accuracy']:.1%}",
                flush=True,
            )
            if summary["run_status"] in {
                "target_reached",
                "early_stopping",
                "budget_exhausted",
                "failed",
            }:
                return
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        return
