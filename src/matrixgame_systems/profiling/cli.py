from __future__ import annotations

import argparse
import sys

from . import command_runner, matrixgame_runner


def main() -> None:
    parser = argparse.ArgumentParser(prog="mgs-profile")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    subparsers.add_parser("matrixgame", help="Profile the pinned Matrix-Game 3 pipeline")
    subparsers.add_parser("command", help="Profile an arbitrary command and GPU telemetry")
    args, remaining = parser.parse_known_args()
    if args.mode == "matrixgame":
        sys.argv = [sys.argv[0], *remaining]
        matrixgame_runner.main()
    else:
        command_runner.main(remaining)


if __name__ == "__main__":
    main()
