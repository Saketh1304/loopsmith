"""Command-line entry point.

Usage:
    loopsmith "explain what a virtual environment is"
    loopsmith --list-models
"""

import argparse
import sys

from loopsmith import __version__
from loopsmith.config import ConfigError, get_api_key, get_model_name, load_env_file
from loopsmith.llm import Message, ModelClient, ModelError
from loopsmith.llm.gemini import GeminiClient

SYSTEM_PROMPT = "You are loopsmith, a concise and careful coding assistant."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="loopsmith",
        description="A coding agent that works on the files in this folder.",
    )
    parser.add_argument("task", nargs="?", help="what you want the agent to do")
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="show the models your API key can use, then exit",
    )
    parser.add_argument(
        "--version", action="version", version=f"loopsmith {__version__}"
    )
    return parser


def main(argv: list[str] | None = None, client: ModelClient | None = None) -> int:
    """Run the CLI. Returns the process exit code (0 means success).

    `client` exists so tests can pass in a fake model. In normal use it is
    None and a real GeminiClient is created from the configuration.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.task and not args.list_models:
        parser.error("give a task, or use --list-models")

    load_env_file()
    try:
        if client is None:
            client = GeminiClient(get_api_key(), get_model_name())

        if args.list_models:
            for name in client.list_models():
                print(name)
            return 0

        reply = client.generate([Message("user", args.task)], system=SYSTEM_PROMPT)
        print(reply.text)
        return 0
    except (ConfigError, ModelError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
