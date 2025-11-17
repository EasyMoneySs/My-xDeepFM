"""CLI entrypoint for preprocessing."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root/src is importable when running as a script
CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from xdeepfm.data.preprocess import run_preprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run preprocessing pipeline")
    parser.add_argument("--config", required=True, help="Path to data YAML config")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if metadata exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_preprocess(args.config, force=args.force)


if __name__ == "__main__":  # pragma: no cover
    main()
