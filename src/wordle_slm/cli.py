from __future__ import annotations

import argparse
import json
import subprocess

from .data import ROOT, prepare_data
from .dataset import generate_training_data
from .preference_training import train_dpo
from .preference_visualization import (
    render_preference_dashboard,
    watch_preference_dashboard,
)
from .preferences import generate_preference_data
from .training import download_model, serve, train
from .validation import validate_all
from .visualization import render_training_dashboard, watch_training_dashboard


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(prog="wordle-slm")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare-data")
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--skip-oracle", action="store_true")
    subparsers.add_parser("download-model")
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--smoke", action="store_true")
    subparsers.add_parser("prepare-preferences")
    preference_parser = subparsers.add_parser("train-preference")
    preference_parser.add_argument("--smoke", action="store_true")
    preference_visualization_parser = subparsers.add_parser("visualize-preference")
    preference_visualization_parser.add_argument("--smoke", action="store_true")
    preference_visualization_parser.add_argument("--watch", action="store_true")
    preference_visualization_parser.add_argument("--interval", type=float, default=15.0)
    visualization_parser = subparsers.add_parser("visualize-training")
    visualization_parser.add_argument("--watch", action="store_true")
    visualization_parser.add_argument("--interval", type=float, default=15.0)
    subparsers.add_parser("serve")
    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--provider", choices=("slm", "deepseek"), default="slm")
    benchmark_parser.add_argument("--track", choices=("pure", "agent", "oracle"), default="pure")
    benchmark_parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    benchmark_parser.add_argument("--limit", type=int)
    subparsers.add_parser("report")
    arguments = parser.parse_args()

    if arguments.command == "prepare-data":
        provenance = prepare_data()
        training = generate_training_data()
        _print({"provenance": provenance, "training_records": training})
    elif arguments.command == "validate":
        _print(validate_all(oracle=not arguments.skip_oracle))
    elif arguments.command == "download-model":
        _print(download_model())
    elif arguments.command == "train":
        _print(train(smoke=arguments.smoke))
    elif arguments.command == "prepare-preferences":
        _print(generate_preference_data())
    elif arguments.command == "train-preference":
        _print(train_dpo(smoke=arguments.smoke))
    elif arguments.command == "visualize-preference":
        if arguments.watch:
            watch_preference_dashboard(arguments.interval)
        else:
            _print(render_preference_dashboard(smoke=arguments.smoke))
    elif arguments.command == "visualize-training":
        if arguments.watch:
            watch_training_dashboard(arguments.interval)
        else:
            _print(render_training_dashboard())
    elif arguments.command == "serve":
        serve()
    elif arguments.command == "benchmark":
        command = [
            "npm",
            "run",
            "benchmark",
            "--",
            "--provider",
            arguments.provider,
            "--track",
            arguments.track,
            "--split",
            arguments.split,
        ]
        if arguments.limit is not None:
            command.extend(["--limit", str(arguments.limit)])
        subprocess.run(command, cwd=ROOT, check=True)
    elif arguments.command == "report":
        report = ROOT / "artifacts" / "benchmark" / "summary.json"
        if not report.exists():
            raise SystemExit("benchmark summary does not exist")
        _print(json.loads(report.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
