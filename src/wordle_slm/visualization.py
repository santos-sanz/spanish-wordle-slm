from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import matplotlib
import numpy as np

from .data import ROOT
from .training import RUN_DIR

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUTPUT_DIR = ROOT / "artifacts" / "training"
TRAIN_RE = re.compile(
    r"Iter (\d+): Train loss ([0-9.]+), Learning Rate ([0-9.eE+-]+), "
    r"It/sec ([0-9.]+), Tokens/sec ([0-9.]+), Trained Tokens (\d+), "
    r"Peak mem ([0-9.]+) GB"
)
VAL_RE = re.compile(r"Iter (\d+): Val loss ([0-9.]+), Val took ([0-9.]+)s")
SAVE_RE = re.compile(r"Iter (\d+): Saved adapter weights")


@dataclass(frozen=True)
class TrainPoint:
    iteration: int
    loss: float
    learning_rate: float
    iterations_per_second: float
    tokens_per_second: float
    trained_tokens: int
    peak_memory_gb: float


@dataclass(frozen=True)
class ValidationPoint:
    iteration: int
    loss: float
    elapsed_seconds: float


@dataclass(frozen=True)
class TrainingSeries:
    train: list[TrainPoint]
    validation: list[ValidationPoint]
    checkpoints: list[int]
    completed: bool


def parse_training_log(text: str, *, iteration_offset: int = 0) -> TrainingSeries:
    train: list[TrainPoint] = []
    validation: list[ValidationPoint] = []
    checkpoints: list[int] = []
    for line in text.replace("\r", "\n").splitlines():
        match = TRAIN_RE.search(line)
        if match:
            train.append(
                TrainPoint(
                    iteration=int(match.group(1)) + iteration_offset,
                    loss=float(match.group(2)),
                    learning_rate=float(match.group(3)),
                    iterations_per_second=float(match.group(4)),
                    tokens_per_second=float(match.group(5)),
                    trained_tokens=int(match.group(6)),
                    peak_memory_gb=float(match.group(7)),
                )
            )
            continue
        match = VAL_RE.search(line)
        if match:
            validation.append(
                ValidationPoint(
                    iteration=int(match.group(1)) + iteration_offset,
                    loss=float(match.group(2)),
                    elapsed_seconds=float(match.group(3)),
                )
            )
            continue
        match = SAVE_RE.search(line)
        if match:
            checkpoints.append(int(match.group(1)) + iteration_offset)
    return TrainingSeries(train, validation, checkpoints, "Saved final weights" in text)


def load_training_series() -> TrainingSeries:
    paths = sorted(RUN_DIR.glob("full-*.log"))
    planned = _planned_iterations()
    offsets: list[int] = []
    for index, path in enumerate(paths):
        if index == 0:
            offsets.append(0)
            continue
        config = path.with_suffix(".yaml")
        match = (
            re.search(r"^iters:\s*(\d+)$", config.read_text(encoding="utf-8"), re.MULTILINE)
            if config.exists()
            else None
        )
        offsets.append(max(0, planned - int(match.group(1))) if match else offsets[-1])

    train: list[TrainPoint] = []
    validation: list[ValidationPoint] = []
    checkpoints: list[int] = []
    for index, (path, offset) in enumerate(zip(paths, offsets, strict=True)):
        parsed = parse_training_log(
            path.read_text(encoding="utf-8", errors="replace"), iteration_offset=offset
        )
        next_offset = offsets[index + 1] if index + 1 < len(offsets) else None
        train.extend(point for point in parsed.train if next_offset is None or point.iteration <= next_offset)
        validation.extend(
            point for point in parsed.validation if next_offset is None or point.iteration <= next_offset
        )
        checkpoints.extend(
            point for point in parsed.checkpoints if next_offset is None or point <= next_offset
        )
    state_path = RUN_DIR / "state.json"
    completed = (
        bool(json.loads(state_path.read_text(encoding="utf-8")).get("completed", False))
        if state_path.exists()
        else bool(paths and parse_training_log(paths[-1].read_text(encoding="utf-8")).completed)
    )
    return TrainingSeries(train, validation, checkpoints, completed)


def _ewma(values: list[float], alpha: float = 0.25) -> np.ndarray:
    if not values:
        return np.array([])
    smoothed = [values[0]]
    for value in values[1:]:
        smoothed.append(alpha * value + (1 - alpha) * smoothed[-1])
    return np.asarray(smoothed)


def _duration(seconds: object) -> str:
    if not isinstance(seconds, int | float) or not math.isfinite(seconds):
        return "—"
    minutes = max(0, round(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m" if hours else f"{minutes} min"


def _planned_iterations() -> int:
    state = RUN_DIR / "state.json"
    if state.exists():
        return int(
            json.loads(state.read_text(encoding="utf-8")).get("iterations_planned", 3000)
        )
    config = RUN_DIR / "full-01.yaml"
    if config.exists():
        match = re.search(
            r"^iters:\s*(\d+)$", config.read_text(encoding="utf-8"), re.MULTILINE
        )
        if match:
            return int(match.group(1))
    return 3000


def summarize(series: TrainingSeries) -> dict[str, object]:
    planned = _planned_iterations()
    current_iteration = max((point.iteration for point in series.train), default=0)
    recent_speed = [point.iterations_per_second for point in series.train[-12:]]
    speed = float(np.median(recent_speed)) if recent_speed else 0.0
    eta_seconds = (planned - current_iteration) / speed if speed > 0 else None
    best_validation = min(series.validation, key=lambda point: point.loss) if series.validation else None
    smoothed = _ewma([point.loss for point in series.train])
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "completed" if series.completed else "training",
        "iteration": current_iteration,
        "iterations_planned": planned,
        "progress": current_iteration / planned if planned else 0,
        "latest_train_loss": series.train[-1].loss if series.train else None,
        "smoothed_train_loss": float(smoothed[-1]) if len(smoothed) else None,
        "latest_validation_loss": series.validation[-1].loss if series.validation else None,
        "best_validation_loss": best_validation.loss if best_validation else None,
        "best_validation_iteration": best_validation.iteration if best_validation else None,
        "peak_memory_gb": max((point.peak_memory_gb for point in series.train), default=0),
        "iterations_per_second": speed,
        "eta_seconds": eta_seconds,
        "checkpoints": len(series.checkpoints),
    }


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


def render_training_dashboard() -> dict[str, object]:
    series = load_training_series()
    if not series.train:
        raise RuntimeError("no training metrics found in artifacts/runs/training/full-*.log")
    summary = summarize(series)
    train_iterations = [point.iteration for point in series.train]
    train_loss = [point.loss for point in series.train]
    smoothed = _ewma(train_loss)
    validation_iterations = [point.iteration for point in series.validation]
    validation_loss = [point.loss for point in series.validation]

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
        "Spanish Wordle SLM — Training",
        fontsize=22,
        fontweight="bold",
        color="#20252B",
    )
    fig.text(
        0.065,
        0.935,
        "QLoRA · LFM2.5 2.6B 6-bit · loss by optimizer iteration · live MLX log",
        fontsize=10.5,
        color="#68707A",
    )

    progress = float(summary["progress"])
    _card(
        fig,
        0.065,
        "PROGRESS",
        f"{int(summary['iteration']):,} / {int(summary['iterations_planned']):,}  ({progress:.1%})",
        "#155EEF",
    )
    _card(
        fig,
        0.29,
        "BEST VALIDATION LOSS",
        f"{float(summary['best_validation_loss']):.3f}  @ {int(summary['best_validation_iteration']):,}",
        "#A15C00",
    )
    _card(
        fig,
        0.515,
        "PEAK UNIFIED MEMORY",
        f"{float(summary['peak_memory_gb']):.3f} GB",
        "#344054",
    )
    _card(
        fig,
        0.74,
        "ESTIMATED TIME LEFT",
        _duration(summary["eta_seconds"]),
        "#344054",
    )

    axis = fig.add_axes((0.08, 0.19, 0.86, 0.56))
    axis.set_facecolor("#FFFFFF")
    axis.grid(axis="y", color="#E8EBEF", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    axis.plot(
        train_iterations,
        train_loss,
        color="#87AFFF",
        linewidth=0.9,
        alpha=0.45,
        marker="o",
        markersize=2.8,
        label="Train loss (raw)",
    )
    axis.plot(
        train_iterations,
        smoothed,
        color="#155EEF",
        linewidth=2.4,
        label="Train loss (EWMA)",
    )
    axis.plot(
        validation_iterations,
        validation_loss,
        color="#C97900",
        linewidth=1.7,
        linestyle="--",
        marker="s",
        markersize=5.5,
        markerfacecolor="#FFFFFF",
        markeredgewidth=1.5,
        label="Validation loss",
    )
    if series.checkpoints:
        axis.scatter(
            series.checkpoints,
            np.zeros(len(series.checkpoints)),
            marker="|",
            s=80,
            linewidths=1.2,
            color="#7B8490",
            label="Saved checkpoint",
            zorder=5,
        )
    best = min(series.validation, key=lambda point: point.loss)
    axis.annotate(
        f"best {best.loss:.3f}",
        xy=(best.iteration, best.loss),
        xytext=(10, 18),
        textcoords="offset points",
        fontsize=9,
        color="#7A4500",
        arrowprops={"arrowstyle": "-", "color": "#A15C00", "linewidth": 0.9},
    )
    axis.set_ylim(bottom=0)
    axis.set_xlim(left=0, right=max(train_iterations[-1] * 1.05, 50))
    axis.axvline(train_iterations[-1], color="#9BA3AC", linewidth=0.9, linestyle=":")
    axis.annotate(
        f"current · {train_iterations[-1]:,}",
        xy=(train_iterations[-1], smoothed[-1]),
        xytext=(-8, -18),
        textcoords="offset points",
        fontsize=8.5,
        color="#68707A",
        ha="right",
    )
    axis.set_xlabel("Optimizer iteration", fontsize=10, labelpad=10)
    axis.set_ylabel("Cross-entropy loss", fontsize=10, labelpad=10)
    axis.legend(loc="upper right", frameon=False, ncols=2, fontsize=9)

    bar_axis = fig.add_axes((0.08, 0.09, 0.86, 0.035))
    bar_axis.barh([0], [1], color="#EDF0F3", height=0.55)
    bar_axis.barh([0], [progress], color="#155EEF", height=0.55)
    bar_axis.set_xlim(0, 1)
    bar_axis.axis("off")
    fig.text(0.08, 0.065, "0", fontsize=8, color="#7B8490")
    fig.text(
        0.94,
        0.065,
        f"{int(summary['iterations_planned']):,}",
        fontsize=8,
        color="#7B8490",
        ha="right",
    )
    fig.text(
        0.5,
        0.035,
        f"Source: MLX training log · updated {str(summary['generated_at'])[:19].replace('T', ' ')} UTC",
        fontsize=8,
        color="#8A929C",
        ha="center",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "loss-curve.png"
    svg_path = OUTPUT_DIR / "loss-curve.svg"
    json_path = OUTPUT_DIR / "status.json"
    png_temp = OUTPUT_DIR / ".loss-curve.png.tmp"
    svg_temp = OUTPUT_DIR / ".loss-curve.svg.tmp"
    fig.savefig(png_temp, format="png", dpi=170, facecolor=fig.get_facecolor())
    fig.savefig(svg_temp, format="svg", facecolor=fig.get_facecolor())
    plt.close(fig)
    png_temp.replace(png_path)
    svg_temp.replace(svg_path)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    raw_path = OUTPUT_DIR / "metrics.jsonl"
    rows = [{"series": "train", **asdict(point)} for point in series.train] + [
        {"series": "validation", **asdict(point)} for point in series.validation
    ]
    raw_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    return {**summary, "png": str(png_path), "svg": str(svg_path)}


def watch_training_dashboard(interval_seconds: float = 15.0) -> None:
    try:
        while True:
            try:
                summary = render_training_dashboard()
            except RuntimeError as error:
                if "no training metrics found" not in str(error):
                    raise
                print("waiting for the first full-training metric", flush=True)
                time.sleep(interval_seconds)
                continue
            print(
                f"iteration={summary['iteration']}/{summary['iterations_planned']} "
                f"val={summary['latest_validation_loss']} best={summary['best_validation_loss']} "
                f"eta={_duration(summary['eta_seconds'])}",
                flush=True,
            )
            if summary["status"] == "completed":
                return
            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        return
