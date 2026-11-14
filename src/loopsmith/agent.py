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

from loopsmith.llm.base import Message, ModelClient, ToolCall, ToolResult
from loopsmith.tools.base import Tool, ToolError

DEFAULT_MAX_STEPS = 20

# Called once for each tool call, after it has run.
ToolObserver = Callable[[ToolCall, ToolResult], None]


@dataclass(frozen=True)
class AgentResult:
    text: str
    steps: int
    finished: bool  # False when the step limit stopped the run


class Agent:
    def __init__(
        self,
        client: ModelClient,
        tools: list[Tool],
        system: str | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        on_tool: ToolObserver | None = None,
    ):
        self._client = client
        self._tools = {tool.name: tool for tool in tools}
        self._system = system
        self._max_steps = max_steps
        self._on_tool = on_tool

    def run(self, task: str) -> AgentResult:
        messages = [Message("user", task)]
        specs = [tool.spec for tool in self._tools.values()]

        for step in range(1, self._max_steps + 1):
            reply = self._client.generate(messages, system=self._system, tools=specs)
            messages.append(reply.as_message())

            if not reply.tool_calls:
                return AgentResult(text=reply.text, steps=step, finished=True)

            results = tuple(self._run_tool(call) for call in reply.tool_calls)
            messages.append(Message("tool", tool_results=results))

        return AgentResult(
            text=f"Stopped after {self._max_steps} steps without finishing.",
            steps=self._max_steps,
            finished=False,
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
                result = ToolResult(call.name, tool.run(call.args), id=call.id)
            except ToolError as error:
                result = self._error(call, str(error))
            except Exception as error:  # a bug in a tool must not kill the loop
                result = self._error(call, f"Tool crashed: {type(error).__name__}: {error}")

        if self._on_tool is not None:
            self._on_tool(call, result)
        return result

    @staticmethod
    def _error(call: ToolCall, message: str) -> ToolResult:
        return ToolResult(call.name, message, is_error=True, id=call.id)
