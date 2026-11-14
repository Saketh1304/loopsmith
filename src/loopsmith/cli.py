"""Command-line entry point.

Usage:
    loopsmith "what does this project do?"
    loopsmith "add a docstring to utils.py" --yes
    loopsmith --list-models
"""

import argparse
import platform
import sys
from pathlib import Path

from loopsmith import __version__
from loopsmith.agent import DEFAULT_MAX_STEPS, Agent, Approver
from loopsmith.config import ConfigError, get_api_key, get_model_name, load_env_file
from loopsmith.llm import ModelClient, ModelError, ToolCall, ToolResult
from loopsmith.llm.gemini import GeminiClient
from loopsmith.llm.retry import RetryingClient
from loopsmith.tools import Workspace, default_tools


def system_prompt() -> str:
    """Standing instructions sent with every model call."""
    return (
        "You are loopsmith, a careful coding agent working inside a project "
        "folder. All paths are relative to that folder.\n"
        "- Look before you act: list and read files instead of guessing what "
        "they contain.\n"
        "- Read a file before editing it. Use edit_file for small changes and "
        "write_file for new files.\n"
        "- After changing code, run the project's tests if it has any.\n"
        f"- Commands run on {platform.system()} through the system shell. They "
        "cannot ask for keyboard input and are stopped after a time limit.\n"
        "- The user approves each change and command. If one is declined, do "
        "not repeat it.\n"
        "When the task is done, reply with a short summary of what you did."
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
        "-y",
        "--yes",
        action="store_true",
        help="approve every file change and command without asking",
    )
    parser.add_argument(
        "--version", action="version", version=f"loopsmith {__version__}"
    )
    return parser


# Progress messages go to stderr so that stdout holds only the final
# answer, which keeps the output usable in scripts.


def print_tool_call(call: ToolCall, result: ToolResult) -> None:
    """Show each tool call after it has run."""
    shown = []
    for key, value in call.args.items():
        text = repr(value)
        shown.append(f"{key}={text if len(text) <= 60 else text[:57] + '...'}")
    status = "error" if result.is_error else f"{len(result.content)} chars"
    print(f"  -> {call.name}({', '.join(shown)})  [{status}]", file=sys.stderr)


def print_retry(error: ModelError, delay: float, attempt: int) -> None:
    print(
        f"  .. model call failed (HTTP {error.status}); "
        f"retrying in {delay:.0f}s (attempt {attempt})",
        file=sys.stderr,
    )


def approve_everything(call: ToolCall, preview: str) -> bool:
    return True


def ask_on_terminal(call: ToolCall, preview: str) -> bool:
    """Show what is about to happen and ask the user to allow it."""
    print(f"\n{preview}", file=sys.stderr)
    if not sys.stdin.isatty():
        # No person is there to answer (a script or a pipe). Refuse, since
        # running unreviewed actions should need an explicit --yes.
        print("Not allowed: no terminal to ask on. Use --yes to allow.", file=sys.stderr)
        return False
    print("Allow? [y/N] ", end="", file=sys.stderr, flush=True)
    answer = sys.stdin.readline().strip().lower()
    return answer in ("y", "yes")


def main(
    argv: list[str] | None = None,
    client: ModelClient | None = None,
    approve: Approver | None = None,
) -> int:
    """Run the CLI. Returns the process exit code (0 means success).

    `client` and `approve` exist so tests can pass in a fake model and a
    scripted approver. In normal use both are None: a real GeminiClient is
    created, and the user is asked on the terminal.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.task and not args.list_models:
        parser.error("give a task, or use --list-models")

    load_env_file()
    try:
        injected = client is not None
        if client is None:
            client = GeminiClient(get_api_key(), get_model_name())

        if args.list_models:
            for name in client.list_models():
                print(name)
            return 0

        if not injected:
            client = RetryingClient(client, on_retry=print_retry)
        if approve is None:
            approve = approve_everything if args.yes else ask_on_terminal

        agent = Agent(
            client,
            tools=default_tools(Workspace(Path.cwd())),
            system=system_prompt(),
            max_steps=args.max_steps,
            on_tool=print_tool_call,
            approve=approve,
        )
        result = agent.run(args.task)
        print(result.text)
        return 0 if result.finished else 1
    except (ConfigError, ModelError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
