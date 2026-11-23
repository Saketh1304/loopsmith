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
from loopsmith.agent import DEFAULT_MAX_STEPS, Agent, AgentResult, Approver
from loopsmith.config import ConfigError, get_api_key, get_model_name, load_env_file
from loopsmith.context import DEFAULT_CONTEXT_BUDGET, SummaryReport, TrimReport
from loopsmith.llm import ModelClient, ModelError, ToolCall, ToolResult, Usage
from loopsmith.llm.gemini import GeminiClient
from loopsmith.llm.retry import RetryingClient
from loopsmith.subagent import SubAgents
from loopsmith.tools import Workspace, default_tools, research_tools


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
        "- For a question that needs reading several files, use delegate. A "
        "sub-agent does the reading and gives you only its findings, which "
        "keeps this conversation short. Read a file yourself when you are "
        "about to edit it.\n"
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
        "--context-budget",
        type=int,
        default=DEFAULT_CONTEXT_BUDGET,
        help="trim old tool output, then summarise old history, once a request "
        f"passes this many tokens (default {DEFAULT_CONTEXT_BUDGET}; 0 turns both off)",
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


def print_tool_call(call: ToolCall, result: ToolResult, indent: str = "  ") -> None:
    """Show each tool call after it has run."""
    shown = []
    for key, value in call.args.items():
        text = repr(value)
        shown.append(f"{key}={text if len(text) <= 60 else text[:57] + '...'}")
    status = "error" if result.is_error else f"{len(result.content)} chars"
    print(f"{indent}-> {call.name}({', '.join(shown)})  [{status}]", file=sys.stderr)


# A sub-agent's activity is shown further in, above the `delegate` call
# that started it.
SUB_INDENT = "      "


def print_sub_tool_call(call: ToolCall, result: ToolResult) -> None:
    print_tool_call(call, result, indent=SUB_INDENT)


def print_sub_step_usage(step: int, usage: Usage) -> None:
    print(
        f"{SUB_INDENT}.. sub-agent step {step}: {usage.input_tokens:,} tokens in, "
        f"{usage.output_tokens:,} out",
        file=sys.stderr,
    )


def print_retry(error: ModelError, delay: float, attempt: int) -> None:
    print(
        f"  .. model call failed (HTTP {error.status}); "
        f"retrying in {delay:.0f}s (attempt {attempt})",
        file=sys.stderr,
    )


def print_step_usage(step: int, usage: Usage) -> None:
    """Show how big each request was, so the history can be seen growing."""
    print(
        f"  .. step {step}: {usage.input_tokens:,} tokens in, "
        f"{usage.output_tokens:,} out",
        file=sys.stderr,
    )


def print_trim(report: TrimReport) -> None:
    noun = "result" if report.results_trimmed == 1 else "results"
    print(
        f"  .. trimmed {report.results_trimmed} old tool {noun}, "
        f"about {report.tokens_freed:,} tokens",
        file=sys.stderr,
    )


def print_summary(report: SummaryReport) -> None:
    if report.error is not None:
        print(
            f"  .. could not summarise the history ({report.error}); carrying on without",
            file=sys.stderr,
        )
        return
    cost = ""
    if report.usage is not None:
        cost = (
            f" (one extra model call: {report.usage.input_tokens:,} tokens in, "
            f"{report.usage.output_tokens:,} out)"
        )
    print(
        f"  .. summarised {report.messages_replaced} old messages{cost}",
        file=sys.stderr,
    )


def print_total_usage(result: AgentResult, subagents: SubAgents) -> None:
    """Show the token totals for the run. Silent if none were reported.

    The totals include what sub-agents used. "last request" is the main
    agent's own history, which is what sub-agents keep small.
    """
    usage = result.usage + subagents.usage
    if usage == Usage():
        return
    line = (
        f"  tokens: {usage.input_tokens:,} in ({usage.cached_tokens:,} cached), "
        f"{usage.output_tokens:,} out; last request {result.context_tokens:,}"
    )
    if subagents.runs:
        noun = "run" if subagents.runs == 1 else "runs"
        line += (
            f"; {subagents.runs} sub-agent {noun} used "
            f"{subagents.usage.input_tokens:,} of the tokens in"
        )
    print(line, file=sys.stderr)


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

        workspace = Workspace(Path.cwd())
        subagents = SubAgents(
            client,
            research_tools(workspace),
            context_budget=args.context_budget,
            on_tool=print_sub_tool_call,
            on_usage=print_sub_step_usage,
        )
        agent = Agent(
            client,
            tools=[*default_tools(workspace), subagents.tool()],
            system=system_prompt(),
            max_steps=args.max_steps,
            on_tool=print_tool_call,
            approve=approve,
            on_usage=print_step_usage,
            context_budget=args.context_budget,
            on_trim=print_trim,
            on_summary=print_summary,
        )
        result = agent.run(args.task)
        print_total_usage(result, subagents)
        print(result.text)
        return 0 if result.finished else 1
    except (ConfigError, ModelError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
