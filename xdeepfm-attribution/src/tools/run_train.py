"""CLI for training."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root/src is importable when running as a script
CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xdeepfm.train.pipeline import train_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run xDeepFM training experiment")
    parser.add_argument("--experiment", required=True, help="Path to experiment YAML")
    parser.add_argument(
        "--timestamp-output",
        action="store_true",
        help="在输出目录名后追加时间戳，便于区分多次运行（若配置文件中未开启）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # 若未显式指定，则交给配置文件中的 use_timestamp 决定
    ts_flag = True if args.timestamp_output else None
    ckpt_path = train_experiment(args.experiment, use_timestamp=ts_flag)
    print(f"Training complete. Best checkpoint stored at {ckpt_path}")  # noqa: T201


if __name__ == "__main__":  # pragma: no cover
    main()
