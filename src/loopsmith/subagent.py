"""Sub-agents: handing a research question to a second agent.

Exploring a codebase fills the history with file contents that are only
needed once. A sub-agent does that exploring in a history of its own and
hands back only its findings, so the main agent's history gains one
short answer instead of ten files.

To the main agent this is an ordinary tool called `delegate`. The agent
loop does not know that sub-agents exist.
"""

from loopsmith.agent import Agent, ToolObserver, UsageObserver
from loopsmith.context import DEFAULT_CONTEXT_BUDGET
from loopsmith.llm.base import ModelClient, ModelError, Usage
from loopsmith.tools.base import Tool, ToolError, require_string

# A sub-agent gets fewer steps than the main agent. A research question
# that needs more than this should be split into smaller ones.
SUB_AGENT_MAX_STEPS = 10

SUB_AGENT_SYSTEM_PROMPT = (
    "You are a research sub-agent of loopsmith, a coding agent. Another "
    "agent has handed you one question about the project in this folder. "
    "All paths are relative to that folder.\n"
    "- You can list and read files. You cannot change files or run "
    "commands, so do not try.\n"
    "- Look instead of guessing: read the files that matter.\n"
    "- Your final reply is the only thing the other agent will see. It does "
    "not see the files you read. Give the answer with the evidence: file "
    "paths, function names, and short quotes of the lines that matter.\n"
    "- If you cannot find the answer, say so and say where you looked."
)

DELEGATE_DESCRIPTION = (
    "Hand a research question to a sub-agent. It explores the project in a "
    "fresh history of its own and returns only its findings, which keeps "
    "this conversation short. Use it for questions that need reading "
    "several files, such as where something is defined or how a part of "
    "the code works. It can list and read files; it cannot change files or "
    "run commands. It does not see this conversation, so give it a "
    "complete question with all the details it needs."
)


class SubAgents:
    """Starts sub-agents and keeps count of what they used.

    The object is needed, and not just a function, because it holds a
    running total. In Java this would be a small class with two fields;
    it is the same here.
    """

    def __init__(
        self,
        client: ModelClient,
        tools: list[Tool],
        max_steps: int = SUB_AGENT_MAX_STEPS,
        context_budget: int | None = DEFAULT_CONTEXT_BUDGET,
        on_tool: ToolObserver | None = None,
        on_usage: UsageObserver | None = None,
    ):
        # A sub-agent has nobody to ask for approval, so it is only ever
        # given tools that change nothing. This is checked here, once,
        # and not left to whoever builds the tool list.
        changing = [tool.name for tool in tools if tool.needs_approval]
        if changing:
            raise ValueError(
                f"Sub-agents are read-only and cannot be given: {', '.join(changing)}."
            )
        self._client = client
        self._tools = tools
        self._max_steps = max_steps
        self._context_budget = context_budget
        self._on_tool = on_tool
        self._on_usage = on_usage
        self.usage = Usage()  # total over every sub-agent run
        self.runs = 0

    def run(self, task: str) -> str:
        """Run one sub-agent on `task` and return its final answer.

        Raises ToolError when there is no usable answer. The main agent
        then gets an error result and decides what to do next, as with
        any other tool.
        """
        # The sub-agent's own tool list has no `delegate` in it, so a
        # sub-agent can never start another one.
        agent = Agent(
            self._client,
            self._tools,
            system=SUB_AGENT_SYSTEM_PROMPT,
            max_steps=self._max_steps,
            on_tool=self._on_tool,
            on_usage=self._on_usage,
            context_budget=self._context_budget,
        )
        self.runs += 1
        try:
            result = agent.run(task)
        except ModelError as error:
            raise ToolError(f"The sub-agent could not reach the model: {error}") from None

        self.usage = self.usage + result.usage
        if not result.finished:
            raise ToolError(
                f"The sub-agent used all {self._max_steps} of its steps without "
                "finishing. Ask a smaller question, or do the work yourself."
            )
        if not result.text.strip():
            raise ToolError("The sub-agent returned an empty answer.")
        return result.text

    def tool(self) -> Tool:
        """The `delegate` tool, to add to the main agent's tool list."""
        return Tool(
            name="delegate",
            description=DELEGATE_DESCRIPTION,
            parameters={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The question for the sub-agent, complete in itself.",
                    }
                },
                "required": ["task"],
            },
            run=lambda args: self.run(require_string(args, "task")),
        )
