"""CLI for evaluation."""
from __future__ import annotations

import argparse

from xdeepfm.train.pipeline import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained checkpoint")
    parser.add_argument("--experiment", required=True, help="Path to experiment YAML")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_checkpoint(args.experiment, args.checkpoint)
    print(metrics)  # noqa: T201


if __name__ == "__main__":  # pragma: no cover
    main()
