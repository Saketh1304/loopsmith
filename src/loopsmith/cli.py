"""Command-line entry point.

Usage:
    loopsmith "what does this project do?"
    loopsmith --list-models
"""

import argparse
import sys
from pathlib import Path

from loopsmith import __version__
from loopsmith.agent import DEFAULT_MAX_STEPS, Agent
from loopsmith.config import ConfigError, get_api_key, get_model_name, load_env_file
from loopsmith.llm import ModelClient, ModelError, ToolCall, ToolResult
from loopsmith.llm.gemini import GeminiClient
from loopsmith.tools import Workspace, default_tools

SYSTEM_PROMPT = (
    "You are loopsmith, a careful coding assistant working inside a project "
    "folder. Use the tools to look at the project before answering: do not "
    "guess what files contain. Paths are relative to the project folder. "
    "When you have what you need, answer concisely."
)


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
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"stop after this many model calls (default {DEFAULT_MAX_STEPS})",
    )
    parser.add_argument(
        "--version", action="version", version=f"loopsmith {__version__}"
    )
    return parser


def print_tool_call(call: ToolCall, result: ToolResult) -> None:
    """Show each tool call as it happens.

    This goes to stderr so that stdout holds only the final answer, which
    keeps the output usable in scripts.
    """
    args = ", ".join(f"{key}={value!r}" for key, value in call.args.items())
    status = "error" if result.is_error else f"{len(result.content)} chars"
    print(f"  -> {call.name}({args})  [{status}]", file=sys.stderr)


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

        agent = Agent(
            client,
            tools=default_tools(Workspace(Path.cwd())),
            system=SYSTEM_PROMPT,
            max_steps=args.max_steps,
            on_tool=print_tool_call,
        )
        result = agent.run(args.task)
        print(result.text)
        return 0 if result.finished else 1
    except (ConfigError, ModelError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
