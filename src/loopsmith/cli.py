"""Command-line entry point.

Usage:
    loopsmith "add input validation to user.py"
"""

import argparse
import sys

from loopsmith import __version__
from loopsmith.config import ConfigError, get_api_key, load_env_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loopsmith",
        description="A coding agent that works on the files in this folder.",
    )
    parser.add_argument("task", help="what you want the agent to do")
    parser.add_argument(
        "--version", action="version", version=f"loopsmith {__version__}"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns the process exit code (0 means success)."""
    args = build_parser().parse_args(argv)

    load_env_file()
    try:
        get_api_key()
    except ConfigError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    # The agent loop arrives in a later step. For now, confirm the setup.
    print(f"Setup OK. Task received: {args.task}")
    print("The agent loop is not built yet.")
    return 0
