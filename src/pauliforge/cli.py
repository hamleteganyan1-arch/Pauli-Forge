"""Command-line research workflows; learned metrics stay explicitly empirical."""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .training import (
    MODES,
    ExperimentConfig,
    _atomic_json,
    evaluate_directory,
    export_checkpoint,
    train_directory,
)


def _add_training_options(parser):
    parser.add_argument("data_dir")
    parser.add_argument("--config", help="ExperimentConfig JSON")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="cpu, cuda[:index], or auto")
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--derivative-weight", type=float)
    parser.add_argument("--domain", default=None, help="FrozenDE JSON; HCNO default if omitted")


def _settings(args):
    values = {}
    if args.config:
        with Path(args.config).open(encoding="utf-8") as stream:
            values = json.load(stream)
    for name in (
        "epochs",
        "lr",
        "seed",
        "device",
        "mode",
        "validation_fraction",
        "derivative_weight",
    ):
        value = getattr(args, name, None)
        if value is not None:
            values[name] = value
    return ExperimentConfig(**values)


def build_parser():
    parser = argparse.ArgumentParser(description="PauliForge density-only research workflows")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="fit kinetic Pauli residual or a separate XC model")
    _add_training_options(train)
    train.add_argument("--output", default="pauliforge.pt")
    evaluate = sub.add_parser("evaluate", help="empirical evaluation with family-leakage checks")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("data_dir")
    evaluate.add_argument("--device", default="cpu")
    evaluate.add_argument(
        "--allow-training-families",
        action="store_true",
        help="explicitly request fit metrics rather than independent test metrics",
    )
    evaluate.add_argument("--output")
    export = sub.add_parser("export", help="portable JSON weights/config/domain export")
    export.add_argument("checkpoint")
    export.add_argument("--output", required=True)
    sweep = sub.add_parser("sweep", help="reproducible seed/learning-rate experiment grid")
    _add_training_options(sweep)
    sweep.add_argument("--seeds", nargs="+", type=int, default=[17, 23])
    sweep.add_argument("--learning-rates", nargs="+", type=float, default=[1e-3])
    sweep.add_argument("--output-dir", default="results/sweep")
    demo = sub.add_parser(
        "demo", help="synthetic trial-density calculation, not a physical benchmark"
    )
    demo.add_argument("--seed", type=int, default=17)
    demo.add_argument("--output")
    console = sub.add_parser("console", help="launch optional Physics Research Console")
    console.add_argument("--port", type=int, default=8501)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "train":
            cfg = _settings(args)
            train_directory(args.data_dir, output=args.output, config=cfg, domain_json=args.domain)
            print(json.dumps({"checkpoint": args.output, "mode": cfg.mode, "status": "EMPIRICAL"}))
        elif args.command == "evaluate":
            print(
                json.dumps(
                    evaluate_directory(
                        args.checkpoint,
                        args.data_dir,
                        device=args.device,
                        allow_training_families=args.allow_training_families,
                        output=args.output,
                    ),
                    indent=2,
                )
            )
        elif args.command == "export":
            report = export_checkpoint(args.checkpoint, args.output)
            print(
                json.dumps(
                    {
                        "output": args.output,
                        "format": report["format"],
                        "domain_hash": report["domain_hash"],
                    }
                )
            )
        elif args.command == "sweep":
            base = asdict(_settings(args))
            runs = []
            for index, (seed, lr) in enumerate(itertools.product(args.seeds, args.learning_rates)):
                config = ExperimentConfig(**dict(base, seed=seed, lr=lr))
                output = Path(args.output_dir) / f"run-{index:03d}" / "model.pt"
                train_directory(
                    args.data_dir, config=config, domain_json=args.domain, output=output
                )
                runs.append({"checkpoint": str(output), "experiment": asdict(config)})
            _atomic_json(
                Path(args.output_dir) / "manifest.json", {"runs": runs, "status": "EMPIRICAL"}
            )
            print(json.dumps({"runs": len(runs), "output_dir": args.output_dir}))
        elif args.command == "demo":
            from .ui import make_demo_payload, summarize_payload

            report = summarize_payload(make_demo_payload(args.seed))
            if args.output:
                _atomic_json(args.output, report)
            print(json.dumps(report, indent=2))
        elif args.command == "console":
            return subprocess.call(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    str(Path(__file__).with_name("ui.py")),
                    "--server.headless=true",
                    "--server.address=127.0.0.1",
                    f"--server.port={args.port}",
                ]
            )
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"pauliforge: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
