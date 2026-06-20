from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parking_intel.pipeline import build_prototype


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the parking intelligence prototype artifacts.")
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "prototype_config.json"),
        help="Path to the prototype configuration JSON file.",
    )
    args = parser.parse_args()
    summary = build_prototype(Path(args.config))
    print("Prototype build complete")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
