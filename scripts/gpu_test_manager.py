#!/usr/bin/env python
"""Compatibility entry point for Opifex GPU verification."""

from __future__ import annotations

import argparse
import sys

from scripts.verify_opifex_gpu import main as verify_gpu


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run the full GPU verifier. This is the default action.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the GPU test manager."""
    parse_args(argv)
    verify_gpu()
    return 0


if __name__ == "__main__":
    sys.exit(main())
