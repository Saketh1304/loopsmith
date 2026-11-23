import pytest

from loopsmith.agent import Agent
from loopsmith.llm import Message, ModelError, ModelReply, ToolCall, Usage
from loopsmith.llm.fake import FakeClient
from loopsmith.subagent import SUB_AGENT_SYSTEM_PROMPT, SubAgents
from loopsmith.tools import Tool, ToolError, Workspace, research_tools

FILE_TEXT = "x" * 5_000  # stands in for the contents of a file


def look_tool() -> Tool:
    """A read-only tool with a large result, like reading a file."""
    return Tool("look", "look tool", {"type": "object"}, run=lambda args: FILE_TEXT)


def changing_tool() -> Tool:
    return Tool(
        "write",
        "write tool",
        {"type": "object"},
        run=lambda args: "written",
        preview=lambda args: "Write a file",
    )


def delegate(task: str, call_id: str | None = None) -> ToolCall:
    return ToolCall("delegate", {"task": task}, id=call_id)


def run_main(client: FakeClient, subagents: SubAgents):
    return Agent(client, tools=[subagents.tool()], system="Main prompt.").run("Fix the bug")


def delegate_result(client: FakeClient):
    """The result of the delegate call, as the main agent's last call saw it."""
    history = client.calls[-1][0]
    return next(m for m in history if m.role == "tool").tool_results[0]


# One FakeClient serves both agents, as one real client does. Its replies
# are used in the order the calls happen: main, then the sub-agent's
# steps, then main again.


def test_the_sub_agent_starts_with_a_fresh_history_and_its_own_prompt():
    client = FakeClient(
        [delegate("Where is median defined?"), ToolCall("look", {}), "In stats.py.", "Done."]
    )

    run_main(client, SubAgents(client, [look_tool()]))

    history, system, tools = client.calls[1]  # the sub-agent's first call
    assert history == [Message("user", "Where is median defined?")]
    assert system == SUB_AGENT_SYSTEM_PROMPT
    # It has the research tools only. There is no `delegate` among them,
    # so a sub-agent cannot start another sub-agent.
    assert [tool.name for tool in tools] == ["look"]


def test_the_main_agent_gets_only_the_final_answer():
    client = FakeClient(
        [delegate("Where is median defined?", "d1"), ToolCall("look", {}), "In stats.py.", "Done."]
    )

    result = run_main(client, SubAgents(client, [look_tool()]))

    assert result.text == "Done."
    history, system, _ = client.calls[-1]
    assert system == "Main prompt."
    answer = delegate_result(client)
    assert (answer.name, answer.content, answer.is_error, answer.id) == (
        "delegate",
        "In stats.py.",
        False,
        "d1",
    )
    # The point of a sub-agent: what it read never enters the main history.
    assert len(history) == 3
    assert all(FILE_TEXT not in str(message) for message in history)


def test_each_sub_agent_run_starts_fresh():
    client = FakeClient(
        [delegate("First question"), "First answer.", delegate("Second question"), "B.", "Done."]
    )

    run_main(client, SubAgents(client, [look_tool()]))

    assert client.calls[3][0] == [Message("user", "Second question")]


def test_sub_agent_tokens_are_counted_separately_from_the_main_agents():
    client = FakeClient(
        [
            ModelReply("", (delegate("Q"),), usage=Usage(1_000, 10)),
            ModelReply("", (ToolCall("look", {}),), usage=Usage(400, 5)),
            ModelReply("Found it.", usage=Usage(2_100, 20, cached_tokens=300)),
            ModelReply("Done.", usage=Usage(1_100, 15)),
        ]
    )
    subagents = SubAgents(client, [look_tool()])

    result = run_main(client, subagents)

    assert subagents.runs == 1
    assert subagents.usage == Usage(input_tokens=2_500, output_tokens=25, cached_tokens=300)
    assert result.usage == Usage(input_tokens=2_100, output_tokens=25)
    # The main history stayed small even though the sub-agent's grew.
    assert result.context_tokens == 1_100


def test_a_sub_agent_that_runs_out_of_steps_is_an_error_result_not_a_crash():
    client = FakeClient(
        [delegate("Q"), ToolCall("look", {}), ToolCall("look", {}), "I will do it myself."]
    )

    result = run_main(client, SubAgents(client, [look_tool()], max_steps=2))

    answer = delegate_result(client)
    assert answer.is_error is True
    assert "used all 2 of its steps" in answer.content
    assert result.text == "I will do it myself."


class SubAgentCallsFail(FakeClient):
    """A FakeClient whose calls for a sub-agent fail."""

    def generate(self, messages, system=None, tools=None):
        if system == SUB_AGENT_SYSTEM_PROMPT:
            raise ModelError("server overloaded", status=503)
        return super().generate(messages, system=system, tools=tools)


def test_a_model_failure_inside_a_sub_agent_does_not_stop_the_main_run():
    client = SubAgentCallsFail([delegate("Q"), "I will read it myself."])

    result = run_main(client, SubAgents(client, [look_tool()]))

    answer = delegate_result(client)
    assert answer.is_error is True
    assert answer.content == "The sub-agent could not reach the model: server overloaded"
    assert result.finished is True


def test_an_empty_answer_from_a_sub_agent_is_an_error():
    client = FakeClient([delegate("Q"), "   ", "Done."])

    run_main(client, SubAgents(client, [look_tool()]))

    answer = delegate_result(client)
    assert answer.is_error is True
    assert "empty answer" in answer.content


def test_sub_agents_cannot_be_given_tools_that_change_things():
    with pytest.raises(ValueError, match="read-only and cannot be given: write"):
        SubAgents(FakeClient([]), [look_tool(), changing_tool()])


def test_delegate_needs_a_task():
    tool = SubAgents(FakeClient([]), [look_tool()]).tool()

    with pytest.raises(ToolError, match="task"):
        tool.run({})


def test_delegate_runs_without_asking_for_approval():
    tool = SubAgents(FakeClient([]), [look_tool()]).tool()

    assert tool.needs_approval is False
    assert tool.spec.parameters["required"] == ["task"]


def test_observers_see_what_the_sub_agent_does():
    tool_calls = []
    steps = []
    client = FakeClient(
        [
            delegate("Q"),
            ModelReply("", (ToolCall("look", {}),), usage=Usage(400, 5)),
            ModelReply("Found it.", usage=Usage(2_100, 20)),
            "Done.",
        ]
    )
    subagents = SubAgents(
        client,
        [look_tool()],
        on_tool=lambda call, result: tool_calls.append(call.name),
        on_usage=lambda step, usage: steps.append((step, usage.input_tokens)),
    )

    run_main(client, subagents)

    assert tool_calls == ["look"]
    assert steps == [(1, 400), (2, 2_100)]


def test_the_research_tools_only_look(tmp_path):
    tools = research_tools(Workspace(tmp_path))

    assert [tool.name for tool in tools] == ["list_files", "read_file"]
    assert not any(tool.needs_approval for tool in tools)
