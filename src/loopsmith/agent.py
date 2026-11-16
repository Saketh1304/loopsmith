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

from loopsmith.llm.base import Message, ModelClient, ToolCall, ToolResult, Usage
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
    ):
        self._client = client
        self._tools = {tool.name: tool for tool in tools}
        self._system = system
        self._max_steps = max_steps
        self._on_tool = on_tool
        self._approve = approve
        self._on_usage = on_usage

    def run(self, task: str) -> AgentResult:
        messages = [Message("user", task)]
        specs = [tool.spec for tool in self._tools.values()]
        total = Usage()
        context_tokens = 0

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

        return AgentResult(
            text=f"Stopped after {self._max_steps} steps without finishing.",
            steps=self._max_steps,
            finished=False,
            usage=total,
            context_tokens=context_tokens,
        )

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
