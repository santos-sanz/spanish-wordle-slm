from __future__ import annotations

import csv
import json
from typing import Any

import matplotlib
import numpy as np

from .data import ROOT

matplotlib.use("Agg")
from matplotlib import pyplot as plt

BENCHMARK_DIR = ROOT / "artifacts" / "benchmark"
SLM_COLOR = "#155EEF"
RIVAL_COLOR = "#C97A00"
INK = "#20252B"
MUTED = "#68707A"
GRID = "#E8EBEF"


def _load(name: str) -> dict[str, Any]:
    path = BENCHMARK_DIR / f"{name}.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if not result.get("complete") or len(result["games"]) != result.get("targetCount"):
        raise RuntimeError(f"incomplete benchmark: {name}")
    return result


def _paired_win_interval(
    slm_games: list[dict[str, Any]],
    rival_games: list[dict[str, Any]],
    *,
    samples: int = 10_000,
) -> tuple[float, float, float]:
    rival_by_target = {game["target"]: game for game in rival_games}
    pairs = [(game, rival_by_target.get(game["target"])) for game in slm_games]
    if not pairs or any(rival is None for _, rival in pairs):
        raise RuntimeError("benchmark target sets differ")
    differences = np.asarray(
        [int(small["solved"]) - int(rival["solved"]) for small, rival in pairs],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260814)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    bootstrap = differences[indices].mean(axis=1)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return float(differences.mean()), float(low), float(high)


def _save_figure(fig: plt.Figure, stem: str) -> dict[str, str]:
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    png = BENCHMARK_DIR / f"{stem}.png"
    svg = BENCHMARK_DIR / f"{stem}.svg"
    fig.savefig(png, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(svg, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"png": str(png), "svg": str(svg)}


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#C9CFD6")
    axis.tick_params(colors=MUTED)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)


def _dashboard(results: dict[str, dict[str, Any]]) -> dict[str, str]:
    tracks = ["pure", "agent"]
    slm = [results[f"slm-{track}"] for track in tracks]
    rival = [results[f"deepseek-{track}"] for track in tracks]
    target_count = int(slm[0]["targetCount"])
    rival_model = str(rival[0]["summary"]["model"])

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="white")
    fig.subplots_adjust(top=0.84, hspace=0.36, wspace=0.24)
    fig.suptitle(
        "Spanish Wordle Model Competition",
        x=0.06,
        ha="left",
        fontsize=23,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.91,
        f"Paired hidden test · {target_count} targets · SLM vs {rival_model} · losses score as 7 turns",
        fontsize=10.5,
        color=MUTED,
    )

    x = np.arange(len(tracks))
    width = 0.34
    axis = axes[0, 0]
    slm_rates = [float(item["summary"]["winRate"]) for item in slm]
    rival_rates = [float(item["summary"]["winRate"]) for item in rival]
    bars_a = axis.bar(x - width / 2, slm_rates, width, color=SLM_COLOR, label="Spanish Wordle SLM")
    bars_b = axis.bar(x + width / 2, rival_rates, width, color=RIVAL_COLOR, label="Benchmark model")
    axis.set_title("Win rate by track", loc="left", fontweight="bold", color=INK)
    axis.set_ylabel("Games won")
    axis.set_ylim(0, 1.08)
    axis.set_xticks(x, ["Pure", "Agent"])
    axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.legend(frameon=False, loc="upper left")
    for bars, items in ((bars_a, slm), (bars_b, rival)):
        for bar, item in zip(bars, items, strict=True):
            wins = int(item["summary"]["wins"])
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.025,
                f"{wins}/{target_count}",
                ha="center",
                fontsize=9,
                color=INK,
            )
    _style_axis(axis)

    axis = axes[0, 1]
    slm_turns = [float(item["summary"]["meanScoredTurns"]) for item in slm]
    rival_turns = [float(item["summary"]["meanScoredTurns"]) for item in rival]
    bars_a = axis.bar(x - width / 2, slm_turns, width, color=SLM_COLOR)
    bars_b = axis.bar(x + width / 2, rival_turns, width, color=RIVAL_COLOR)
    axis.set_title("Mean scored turns", loc="left", fontweight="bold", color=INK)
    axis.set_ylabel("Turns; lower is better")
    axis.set_ylim(0, 7.5)
    axis.set_xticks(x, ["Pure", "Agent"])
    for bars in (bars_a, bars_b):
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.12,
                f"{bar.get_height():.2f}",
                ha="center",
                fontsize=9,
                color=INK,
            )
    _style_axis(axis)

    axis = axes[1, 0]
    rows: list[tuple[str, dict[str, Any]]] = []
    for track in tracks:
        rows.extend(
            [
                (f"{track.title()} · SLM", results[f"slm-{track}"]),
                (f"{track.title()} · benchmark", results[f"deepseek-{track}"]),
            ]
        )
    early, late, losses = [], [], []
    for _, result in rows:
        games = result["games"]
        early.append(sum(game["solved"] and game["turns"] <= 3 for game in games) / len(games))
        late.append(sum(game["solved"] and game["turns"] > 3 for game in games) / len(games))
        losses.append(sum(not game["solved"] for game in games) / len(games))
    y = np.arange(len(rows))
    axis.barh(y, early, color="#155EEF", label="Solved in 1–3")
    axis.barh(y, late, left=early, color="#9CB9FF", edgecolor="#155EEF", label="Solved in 4–6")
    axis.barh(
        y,
        losses,
        left=np.asarray(early) + np.asarray(late),
        color="#E4E7EC",
        edgecolor="#98A2B3",
        label="Loss",
    )
    axis.set_title("Outcome composition", loc="left", fontweight="bold", color=INK)
    axis.set_xlabel("Share of targets")
    axis.set_xlim(0, 1)
    axis.set_yticks(y, [label for label, _ in rows])
    axis.invert_yaxis()
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    axis.legend(frameon=False, ncols=3, loc="lower left", bbox_to_anchor=(0, 1.01))
    _style_axis(axis)

    axis = axes[1, 1]
    intervals = [
        _paired_win_interval(
            results[f"slm-{track}"]["games"], results[f"deepseek-{track}"]["games"]
        )
        for track in tracks
    ]
    observed = np.asarray([item[0] for item in intervals])
    lows = np.asarray([item[1] for item in intervals])
    highs = np.asarray([item[2] for item in intervals])
    axis.axvline(0, color="#344054", linewidth=1.1, linestyle="--")
    axis.errorbar(
        observed,
        np.arange(2),
        xerr=np.vstack((observed - lows, highs - observed)),
        fmt="o",
        markersize=8,
        color=SLM_COLOR,
        ecolor=SLM_COLOR,
        capsize=5,
        linewidth=2,
    )
    axis.set_title("Paired win-rate advantage", loc="left", fontweight="bold", color=INK)
    axis.set_xlabel("SLM win rate minus benchmark win rate; 95% bootstrap CI")
    axis.set_yticks(np.arange(2), ["Pure", "Agent"])
    axis.set_xlim(min(-0.1, float(lows.min()) - 0.05), max(0.1, float(highs.max()) + 0.05))
    axis.xaxis.set_major_formatter(lambda value, _: f"{value:+.0%}")
    for index, (point, low, high) in enumerate(intervals):
        axis.text(
            high + 0.01,
            index,
            f"{point:+.1%} [{low:+.1%}, {high:+.1%}]",
            va="center",
            fontsize=9,
            color=INK,
        )
    _style_axis(axis)
    return _save_figure(fig, "competition-dashboard")


def _progress(results: dict[str, dict[str, Any]]) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), facecolor="white", sharey=True)
    fig.subplots_adjust(top=0.79, wspace=0.12)
    fig.suptitle(
        "Cumulative Win Rate During Evaluation",
        x=0.06,
        ha="left",
        fontsize=21,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.06,
        0.87,
        "Deterministic target order; endpoint values are the final reported rates",
        fontsize=10,
        color=MUTED,
    )
    for axis, track in zip(axes, ("pure", "agent"), strict=True):
        for name, color, label in (
            (f"slm-{track}", SLM_COLOR, "Spanish Wordle SLM"),
            (f"deepseek-{track}", RIVAL_COLOR, "Benchmark model"),
        ):
            wins = np.asarray([int(game["solved"]) for game in results[name]["games"]])
            cumulative = np.cumsum(wins) / np.arange(1, len(wins) + 1)
            axis.plot(
                np.arange(1, len(wins) + 1), cumulative, color=color, linewidth=2.2, label=label
            )
            axis.scatter([len(wins)], [cumulative[-1]], color=color, s=35, zorder=3)
        axis.set_title(track.title(), loc="left", fontweight="bold", color=INK)
        axis.set_xlabel("Evaluated targets")
        axis.set_xlim(1, len(results[f"slm-{track}"]["games"]))
        axis.set_ylim(0, 1.02)
        axis.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
        _style_axis(axis)
    axes[0].set_ylabel("Cumulative win rate")
    axes[0].legend(frameon=False, loc="lower right")
    return _save_figure(fig, "competition-progress")


def _write_csvs(results: dict[str, dict[str, Any]]) -> dict[str, str]:
    summary_path = BENCHMARK_DIR / "competition-summary.csv"
    game_path = BENCHMARK_DIR / "competition-games.csv"
    summary_fields = [
        "provider",
        "model",
        "track",
        "games",
        "wins",
        "winRate",
        "meanScoredTurns",
        "invalidActions",
        "toolCalls",
        "latencyMs",
        "inputTokens",
        "outputTokens",
        "costUsd",
        "errors",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for result in results.values():
            writer.writerow({field: result["summary"].get(field) for field in summary_fields})
    game_fields = [
        "provider",
        "model",
        "track",
        "target",
        "solved",
        "turns",
        "scoredTurns",
        "invalidActions",
        "toolCalls",
        "latencyMs",
        "inputTokens",
        "outputTokens",
        "costUsd",
    ]
    with game_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=game_fields)
        writer.writeheader()
        for result in results.values():
            for game in result["games"]:
                writer.writerow(
                    {
                        "provider": result["summary"]["provider"],
                        "model": result["summary"]["model"],
                        "track": result["summary"]["track"],
                        **{field: game.get(field) for field in game_fields[3:]},
                    }
                )
    return {"summary_csv": str(summary_path), "games_csv": str(game_path)}


def render_benchmark_visuals() -> dict[str, Any]:
    names = ["slm-pure", "deepseek-pure", "slm-agent", "deepseek-agent"]
    results = {name: _load(name) for name in names}
    target_sets = [{game["target"] for game in result["games"]} for result in results.values()]
    if any(targets != target_sets[0] for targets in target_sets[1:]):
        raise RuntimeError("benchmark target sets differ")
    dashboard = _dashboard(results)
    progress = _progress(results)
    csvs = _write_csvs(results)
    chart_map = {
        "surface": "static PNG and SVG supporting a technical HTML report",
        "palette": "hard two-root cap: blue SLM, gold benchmark, neutral references",
        "charts": [
            {
                "artifact": dashboard,
                "questions": [
                    "win-rate comparison",
                    "mean scored turns",
                    "outcome composition",
                    "paired uncertainty",
                ],
            },
            {"artifact": progress, "questions": ["cumulative evaluation stability"]},
        ],
        "data": csvs,
    }
    chart_map_path = BENCHMARK_DIR / "chart-map.json"
    chart_map_path.write_text(json.dumps(chart_map, indent=2) + "\n", encoding="utf-8")
    return {"dashboard": dashboard, "progress": progress, **csvs, "chart_map": str(chart_map_path)}
