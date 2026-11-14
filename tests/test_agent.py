from loopsmith.agent import Agent
from loopsmith.llm import Message, ToolCall, ToolResult
from loopsmith.llm.fake import FakeClient
from loopsmith.tools import Tool, ToolError


def make_tool(name: str, run) -> Tool:
    return Tool(name=name, description=f"{name} tool", parameters={"type": "object"}, run=run)


def echo_tool() -> Tool:
    return make_tool("echo", lambda args: f"echo: {args.get('text')}")


def test_plain_answer_finishes_in_one_step():
    client = FakeClient(["The answer is 4."])

    result = Agent(client, tools=[echo_tool()]).run("What is 2 + 2?")

    assert result.text == "The answer is 4."
    assert result.steps == 1
    assert result.finished is True


def test_tool_call_is_run_and_result_goes_back_to_the_model():
    client = FakeClient([ToolCall("echo", {"text": "hi"}, id="call-1"), "Done."])

    result = Agent(client, tools=[echo_tool()], system="Be brief.").run("Say hi")

    assert result.text == "Done."
    assert result.steps == 2

    # The second model call must see the whole history: the task, the
    # model's own tool call, then the tool's result.
    history, system, tools = client.calls[1]
    assert system == "Be brief."
    assert [tool.name for tool in tools] == ["echo"]
    assert history == [
        Message("user", "Say hi"),
        Message("assistant", "", (ToolCall("echo", {"text": "hi"}, id="call-1"),)),
        Message("tool", tool_results=(ToolResult("echo", "echo: hi", id="call-1"),)),
    ]


def test_several_tool_calls_in_one_turn_all_run_in_order():
    calls = [ToolCall("echo", {"text": "a"}), ToolCall("echo", {"text": "b"})]
    client = FakeClient([calls, "Done."])

    Agent(client, tools=[echo_tool()]).run("Say a then b")

    results = client.calls[1][0][-1].tool_results
    assert [result.content for result in results] == ["echo: a", "echo: b"]


def test_unknown_tool_becomes_an_error_result_not_a_crash():
    client = FakeClient([ToolCall("delete_everything", {}), "Sorry, I cannot do that."])

    result = Agent(client, tools=[echo_tool()]).run("Delete everything")

    error = client.calls[1][0][-1].tool_results[0]
    assert error.is_error is True
    assert "Unknown tool 'delete_everything'" in error.content
    assert "echo" in error.content
    assert result.finished is True


def test_tool_error_is_reported_to_the_model():
    def fail(args):
        raise ToolError("File 'x.py' does not exist.")

    client = FakeClient([ToolCall("read", {"path": "x.py"}), "It is missing."])

    Agent(client, tools=[make_tool("read", fail)]).run("Read x.py")

    error = client.calls[1][0][-1].tool_results[0]
    assert error.is_error is True
    assert error.content == "File 'x.py' does not exist."


def test_unexpected_tool_crash_does_not_kill_the_loop():
    def crash(args):
        raise ZeroDivisionError("division by zero")

    client = FakeClient([ToolCall("calc", {}), "The tool failed."])

    result = Agent(client, tools=[make_tool("calc", crash)]).run("Divide")

    error = client.calls[1][0][-1].tool_results[0]
    assert error.is_error is True
    assert "ZeroDivisionError" in error.content
    assert result.text == "The tool failed."


def test_step_limit_stops_a_model_that_never_finishes():
    endless = [ToolCall("echo", {"text": "again"}) for _ in range(10)]
    client = FakeClient(endless)

    result = Agent(client, tools=[echo_tool()], max_steps=3).run("Loop forever")

    assert result.finished is False
    assert result.steps == 3
    assert len(client.calls) == 3
    assert "3 steps" in result.text


def test_observer_sees_every_tool_call():
    seen = []
    client = FakeClient([ToolCall("echo", {"text": "hi"}), "Done."])

    Agent(
        client,
        tools=[echo_tool()],
        on_tool=lambda call, result: seen.append((call.name, result.content)),
    ).run("Say hi")

    assert seen == [("echo", "echo: hi")]


# --- Approval ---


def changing_tool(log: list) -> Tool:
    """A tool that needs approval and records when it really runs."""

    def run(args):
        log.append(args)
        return "changed"

    def preview(args):
        if "target" not in args:
            raise ToolError("Missing required argument 'target'.")
        return f"Change {args['target']}"

    return Tool("change", "changes things", {"type": "object"}, run=run, preview=preview)


def test_approved_action_runs():
    ran, asked = [], []
    client = FakeClient([ToolCall("change", {"target": "a.py"}), "Done."])

    def approve(call, preview):
        asked.append(preview)
        return True

    Agent(client, tools=[changing_tool(ran)], approve=approve).run("Change a.py")

    assert asked == ["Change a.py"]
    assert ran == [{"target": "a.py"}]
    assert client.calls[1][0][-1].tool_results[0].content == "changed"


def test_declined_action_does_not_run_and_the_model_is_told():
    ran = []
    client = FakeClient([ToolCall("change", {"target": "a.py"}), "Understood."])

    result = Agent(
        client, tools=[changing_tool(ran)], approve=lambda call, preview: False
    ).run("Change a.py")

    assert ran == []
    declined = client.calls[1][0][-1].tool_results[0]
    assert declined.is_error is True
    assert "declined" in declined.content
    assert result.text == "Understood."


def test_without_an_approver_changing_tools_are_refused():
    ran = []
    client = FakeClient([ToolCall("change", {"target": "a.py"}), "Understood."])

    Agent(client, tools=[changing_tool(ran)]).run("Change a.py")

    assert ran == []
    assert client.calls[1][0][-1].tool_results[0].is_error is True


def test_invalid_arguments_fail_before_the_user_is_asked():
    ran, asked = [], []
    client = FakeClient([ToolCall("change", {}), "Understood."])

    def approve(call, preview):
        asked.append(preview)
        return True

    Agent(client, tools=[changing_tool(ran)], approve=approve).run("Change something")

    assert asked == []
    assert ran == []
    assert "target" in client.calls[1][0][-1].tool_results[0].content


def test_read_only_tools_never_ask_for_approval():
    asked = []
    client = FakeClient([ToolCall("echo", {"text": "hi"}), "Done."])

    Agent(
        client, tools=[echo_tool()], approve=lambda call, preview: asked.append(preview)
    ).run("Say hi")

    assert asked == []
