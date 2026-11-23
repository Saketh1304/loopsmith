"""The agent loop: the core of the project.

    1. Send the conversation and the list of tools to the model.
    2. If the model answers with text only, the task is done.
    3. If it asks for tools, run them and add the results to the
       conversation.
    4. Go back to step 1.

The model never runs anything itself. It only asks; this code decides
what happens.
"""

from dataclasses import dataclass
from typing import Callable

from loopsmith.context import (
    DEFAULT_CONTEXT_BUDGET,
    SUMMARY_SYSTEM_PROMPT,
    SummaryReport,
    TrimReport,
    estimate_history_tokens,
    split_history,
    summary_request,
    task_message,
    trim_history,
    trim_target,
    worth_summarising,
)
from loopsmith.llm.base import (
    Message,
    ModelClient,
    ModelError,
    ToolCall,
    ToolResult,
    Usage,
)
from loopsmith.tools.base import Tool, ToolError

DEFAULT_MAX_STEPS = 20

DECLINED_MESSAGE = (
    "The user declined this action, so it was not carried out. Do not try "
    "the same action again. Continue another way, or stop and explain."
)

# Called once for each tool call, after it has run.
ToolObserver = Callable[[ToolCall, ToolResult], None]

# Asked before a tool that changes things runs. Receives the call and a
# description of what it will do; returns True to allow it.
Approver = Callable[[ToolCall, str], bool]

# Called after each model call that reported token counts, with the step
# number and the counts for that one call.
UsageObserver = Callable[[int, Usage], None]

# Called each time old tool output was trimmed from the history.
TrimObserver = Callable[[TrimReport], None]

# Called each time the agent summarised old history, or tried and failed.
SummaryObserver = Callable[[SummaryReport], None]


@dataclass(frozen=True)
class AgentResult:
    text: str
    steps: int
    finished: bool  # False when the step limit stopped the run
    usage: Usage = Usage()  # total over every model call in the run
    # Input tokens of the last model call: how big the history had grown.
    context_tokens: int = 0


class Agent:
    def __init__(
        self,
        client: ModelClient,
        tools: list[Tool],
        system: str | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        on_tool: ToolObserver | None = None,
        approve: Approver | None = None,
        on_usage: UsageObserver | None = None,
        context_budget: int | None = DEFAULT_CONTEXT_BUDGET,
        on_trim: TrimObserver | None = None,
        on_summary: SummaryObserver | None = None,
    ):
        self._client = client
        self._tools = {tool.name: tool for tool in tools}
        self._system = system
        self._max_steps = max_steps
        self._on_tool = on_tool
        self._approve = approve
        self._on_usage = on_usage
        # None or 0 turns trimming and summarising off.
        self._context_budget = context_budget
        self._on_trim = on_trim
        self._on_summary = on_summary

    def run(self, task: str) -> AgentResult:
        messages = [task_message(task)]
        specs = [tool.spec for tool in self._tools.values()]
        total = Usage()
        context_tokens = 0
        summary = ""  # the latest summary of old history, if one was made

        for step in range(1, self._max_steps + 1):
            reply = self._client.generate(messages, system=self._system, tools=specs)
            messages.append(reply.as_message())

            # Every call sends the whole history again, so the total grows
            # faster than the history does. The latest call's input count
            # is the size of the history itself.
            if reply.usage is not None:
                total = total + reply.usage
                context_tokens = reply.usage.input_tokens
                if self._on_usage is not None:
                    self._on_usage(step, reply.usage)

            if not reply.tool_calls:
                return AgentResult(
                    text=reply.text,
                    steps=step,
                    finished=True,
                    usage=total,
                    context_tokens=context_tokens,
                )

            results = tuple(self._run_tool(call) for call in reply.tool_calls)
            messages.append(Message("tool", tool_results=results))

            # Shrink only on a fresh measurement. With no token count from
            # the provider the history size is unknown, and it is left
            # alone.
            if reply.usage is not None and self._context_budget:
                # The provider's count is for the request just sent. The
                # next one also carries this reply and the results that just
                # arrived, so their estimated size is added before deciding.
                # Without it, one big file read slips a whole request past
                # the budget before trimming notices (decision 61).
                tokens = reply.usage.input_tokens + estimate_history_tokens(messages[-2:])
                messages, tokens = self._trim_if_over_budget(messages, tokens)
                if tokens > self._context_budget:
                    # Trimming was not enough. Summarising costs one more
                    # model call, and that call is counted in the total.
                    messages, summary, cost = self._summarise(messages, task, summary)
                    total = total + cost

        return AgentResult(
            text=f"Stopped after {self._max_steps} steps without finishing.",
            steps=self._max_steps,
            finished=False,
            usage=total,
            context_tokens=context_tokens,
        )

    def _trim_if_over_budget(
        self, messages: list[Message], tokens: int
    ) -> tuple[list[Message], int]:
        """Trim old tool output once the last request passed the budget.

        Returns the history and its estimated size after trimming.
        """
        budget = self._context_budget
        if not budget or tokens <= budget:
            return messages, tokens
        trimmed, report = trim_history(messages, tokens - trim_target(budget))
        if report.results_trimmed and self._on_trim is not None:
            self._on_trim(report)
        return trimmed, tokens - report.tokens_freed

    def _summarise(
        self, messages: list[Message], task: str, summary: str
    ) -> tuple[list[Message], str, Usage]:
        """Replace the older part of the history with a summary.

        Returns the history, the summary now in force, and what the extra
        model call cost. Summarising only saves tokens; the run does not
        depend on it. So if the call fails, the history is returned as it
        was and the run goes on.
        """
        old, recent = split_history(messages)
        if not worth_summarising(old, self._context_budget or 0):
            return messages, summary, Usage()

        try:
            reply = self._client.generate(
                summary_request(task, summary, old), system=SUMMARY_SYSTEM_PROMPT
            )
        except ModelError as error:
            self._report_summary(SummaryReport(error=str(error)))
            return messages, summary, Usage()

        cost = reply.usage or Usage()
        new_summary = reply.text.strip()
        if not new_summary:
            self._report_summary(
                SummaryReport(usage=reply.usage, error="the model returned an empty summary")
            )
            return messages, summary, cost

        self._report_summary(SummaryReport(messages_replaced=len(old), usage=reply.usage))
        return [task_message(task, new_summary), *recent], new_summary, cost

    def _report_summary(self, report: SummaryReport) -> None:
        if self._on_summary is not None:
            self._on_summary(report)

    def _run_tool(self, call: ToolCall) -> ToolResult:
        """Run one tool call. Never raises: failures become error results.

        A failed tool call goes back to the model as an error message, so
        it can correct itself (fix a path, pick another tool) instead of
        the whole run crashing.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            known = ", ".join(sorted(self._tools)) or "none"
            result = self._error(call, f"Unknown tool '{call.name}'. Available tools: {known}.")
        else:
            try:
                if tool.needs_approval and not self._is_approved(tool, call):
                    result = self._error(call, DECLINED_MESSAGE)
                else:
                    result = ToolResult(call.name, tool.run(call.args), id=call.id)
            except ToolError as error:
                result = self._error(call, str(error))
            except Exception as error:  # a bug in a tool must not kill the loop
                result = self._error(call, f"Tool crashed: {type(error).__name__}: {error}")

        if self._on_tool is not None:
            self._on_tool(call, result)
        return result

    def _is_approved(self, tool: Tool, call: ToolCall) -> bool:
        """Ask the approver. With no approver configured, the answer is no.

        Building the preview also validates the arguments, so a bad call
        raises ToolError here and the user is never asked about it.
        """
        preview = tool.preview(call.args)
        if self._approve is None:
            return False
        return self._approve(call, preview)

    @staticmethod
    def _error(call: ToolCall, message: str) -> ToolResult:
        return ToolResult(call.name, message, is_error=True, id=call.id)
