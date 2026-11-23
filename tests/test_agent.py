from loopsmith.agent import Agent
from loopsmith.context import SUMMARY_SYSTEM_PROMPT, SummaryReport, TrimReport
from loopsmith.llm import Message, ModelError, ModelReply, ToolCall, ToolResult, Usage
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


def test_usages_add_field_by_field():
    assert Usage(10, 2, 4) + Usage(30, 5, 20) == Usage(40, 7, 24)


def test_token_usage_is_totalled_across_the_run():
    client = FakeClient(
        [
            ModelReply("", (ToolCall("echo", {"text": "hi"}),), usage=Usage(100, 10)),
            ModelReply("Done.", usage=Usage(150, 5, cached_tokens=90)),
        ]
    )

    result = Agent(client, tools=[echo_tool()]).run("Say hi")

    assert result.usage == Usage(input_tokens=250, output_tokens=15, cached_tokens=90)
    # The history size is what the last call sent, not the running total.
    assert result.context_tokens == 150


def test_usage_is_still_reported_when_the_step_limit_stops_the_run():
    replies = [
        ModelReply("", (ToolCall("echo", {"text": "x"}),), usage=Usage(100, 10))
        for _ in range(3)
    ]

    result = Agent(FakeClient(replies), tools=[echo_tool()], max_steps=2).run("Loop")

    assert result.finished is False
    assert result.usage == Usage(input_tokens=200, output_tokens=20)
    assert result.context_tokens == 100


def test_a_model_that_reports_no_usage_gives_zero_totals():
    result = Agent(FakeClient(["Hello."]), tools=[]).run("Say hello")

    assert result.usage == Usage()
    assert result.context_tokens == 0


def test_usage_observer_sees_each_step_that_reported_counts():
    seen = []
    client = FakeClient(
        [
            ModelReply("", (ToolCall("echo", {"text": "a"}),), usage=Usage(100, 10)),
            ToolCall("echo", {"text": "b"}),  # this reply reports no usage
            ModelReply("Done.", usage=Usage(300, 5)),
        ]
    )

    Agent(
        client, tools=[echo_tool()], on_usage=lambda step, usage: seen.append((step, usage))
    ).run("Say a then b")

    assert seen == [(1, Usage(100, 10)), (3, Usage(300, 5))]


def big_tool() -> Tool:
    return make_tool("big", lambda args: "x" * 6_000)


def big_call(input_tokens: int | None) -> ModelReply:
    """A reply asking for the big tool, reporting the given request size."""
    usage = None if input_tokens is None else Usage(input_tokens, 10)
    return ModelReply("", (ToolCall("big", {}),), usage=usage)


def first_result_seen_by_last_call(client: FakeClient) -> str:
    history = client.calls[-1][0]
    return next(m for m in history if m.role == "tool").tool_results[0].content


def test_old_tool_output_is_trimmed_once_the_budget_is_passed():
    # Four tool turns. The fourth request reports 50,000 tokens, over the
    # budget, so the oldest result (outside the three protected turns) is
    # replaced before the fifth request.
    client = FakeClient(
        [big_call(1_000), big_call(3_000), big_call(5_000), big_call(50_000), "Done."]
    )

    Agent(client, tools=[big_tool()], context_budget=30_000).run("Go")

    history = client.calls[-1][0]
    texts = [m.tool_results[0].content for m in history if m.role == "tool"]
    assert texts[0].startswith("[Old output removed to save space: 6,000 characters")
    assert texts[1:] == ["x" * 6_000] * 3


def test_history_under_the_budget_is_left_alone():
    client = FakeClient(
        [big_call(1_000), big_call(3_000), big_call(5_000), big_call(7_000), "Done."]
    )

    Agent(client, tools=[big_tool()], context_budget=30_000).run("Go")

    assert first_result_seen_by_last_call(client) == "x" * 6_000


def test_history_is_not_trimmed_when_the_model_reports_no_usage():
    client = FakeClient([big_call(None)] * 4 + ["Done."])

    Agent(client, tools=[big_tool()], context_budget=1).run("Go")

    assert first_result_seen_by_last_call(client) == "x" * 6_000


def test_a_budget_of_zero_turns_trimming_off():
    client = FakeClient(
        [big_call(1_000), big_call(3_000), big_call(5_000), big_call(50_000), "Done."]
    )

    Agent(client, tools=[big_tool()], context_budget=0).run("Go")

    assert first_result_seen_by_last_call(client) == "x" * 6_000


def test_a_big_new_result_is_counted_before_the_next_request_is_sent():
    # The fourth request was 25,000 tokens, under the 30,000 budget, but it
    # brought back about 20,000 tokens of output. Sending the fifth request
    # as it stood would cost about 45,000. The old check only looked at the
    # 25,000 and trimmed nothing; the estimate of the new output catches it.
    huge = make_tool("huge", lambda args: "x" * 60_000)
    huge_call = ModelReply("", (ToolCall("huge", {}),), usage=Usage(25_000, 10))
    client = FakeClient([big_call(1_000), big_call(3_000), big_call(5_000), huge_call, "Done."])

    Agent(client, tools=[big_tool(), huge], context_budget=30_000).run("Go")

    assert first_result_seen_by_last_call(client).startswith("[Old output removed")


def test_trim_observer_is_told_what_was_trimmed():
    seen = []
    client = FakeClient(
        [big_call(1_000), big_call(3_000), big_call(5_000), big_call(50_000), "Done."]
    )

    Agent(client, tools=[big_tool()], context_budget=30_000, on_trim=seen.append).run("Go")

    assert len(seen) == 1
    assert isinstance(seen[0], TrimReport)
    assert seen[0].results_trimmed == 1
    assert seen[0].tokens_freed > 1_900


# Summarising. These runs use a tool whose results are too small to trim
# (under 1,000 characters), so trimming frees nothing and the agent has to
# summarise. With a budget of 1,000 tokens, two old turns of 900
# characters are enough to be worth a summary.


def small_tool() -> Tool:
    return make_tool("small", lambda args: "y" * 900)


def small_call(input_tokens: int) -> ModelReply:
    return ModelReply("", (ToolCall("small", {}),), usage=Usage(input_tokens, 10))


def over_budget_script(summary_reply) -> list:
    """Five tool turns; the fifth request reports a size over the budget."""
    return [*[small_call(500)] * 4, small_call(5_000), summary_reply, "Done."]


def test_old_history_is_summarised_when_trimming_is_not_enough():
    client = FakeClient(over_budget_script(ModelReply("The bug is in median().")))

    result = Agent(client, tools=[small_tool()], system="Be brief.", context_budget=1_000).run(
        "Fix the bug"
    )

    # The sixth model call is the request for a summary: one plain message,
    # its own instructions, and no tools.
    request, system, tools = client.calls[5]
    assert len(request) == 1
    assert "Fix the bug" in request[0].text
    assert "small result: " + "y" * 900 in request[0].text
    assert system == SUMMARY_SYSTEM_PROMPT
    assert tools is None

    # The next real call sees the task with the summary, then only the
    # three most recent tool turns.
    history, system, _ = client.calls[6]
    assert system == "Be brief."
    assert history[0].text.startswith("Fix the bug\n\n[The earlier part")
    assert history[0].text.endswith("The bug is in median().")
    assert [message.role for message in history[1:]] == ["assistant", "tool"] * 3

    # The summary call is not a step of the task.
    assert result.text == "Done."
    assert result.steps == 6


def test_the_summary_is_written_from_what_trimming_removed():
    # One big result, then small ones. On the fifth step trimming replaces
    # the big result with a note and then a summary is still needed. The
    # summariser must see what the big result said, not the note.
    client = FakeClient(
        [
            big_call(500),
            *[small_call(600)] * 3,
            small_call(5_000),
            ModelReply("A summary."),
            "Done.",
        ]
    )

    Agent(client, tools=[big_tool(), small_tool()], context_budget=1_000).run("Fix the bug")

    request = next(msgs for msgs, system, _ in client.calls if system == SUMMARY_SYSTEM_PROMPT)
    text = request[0].text
    assert "big result: " + "x" * 2_000 + " [... 4,000 more characters]" in text
    assert "[Old output removed" not in text


def test_the_summary_call_is_counted_in_the_token_total():
    summary = ModelReply("A summary.", usage=Usage(700, 40))
    client = FakeClient(over_budget_script(summary))

    result = Agent(client, tools=[small_tool()], context_budget=1_000).run("Fix the bug")

    # Five tool steps (4 x 500 + 5,000 in, 10 out each) plus the summary.
    assert result.usage == Usage(input_tokens=7_700, output_tokens=90)


def test_summary_observer_is_told_how_much_was_replaced():
    seen = []
    summary = ModelReply("A summary.", usage=Usage(700, 40))
    client = FakeClient(over_budget_script(summary))

    Agent(client, tools=[small_tool()], context_budget=1_000, on_summary=seen.append).run("Go")

    # Two old turns, each an assistant message and a tool message.
    assert seen == [SummaryReport(messages_replaced=4, usage=Usage(700, 40))]


class SummaryFailsClient(FakeClient):
    """A FakeClient whose summary calls fail, as with a used-up quota."""

    def generate(self, messages, system=None, tools=None):
        if system == SUMMARY_SYSTEM_PROMPT:
            self.calls.append((list(messages), system, tools))
            raise ModelError("quota used up", status=429)
        return super().generate(messages, system=system, tools=tools)


def test_a_failed_summary_does_not_stop_the_run():
    seen = []
    client = SummaryFailsClient([*[small_call(500)] * 4, small_call(5_000), "Done."])

    result = Agent(
        client, tools=[small_tool()], context_budget=1_000, on_summary=seen.append
    ).run("Fix the bug")

    assert result.text == "Done."
    assert result.finished is True
    assert seen == [SummaryReport(error="quota used up")]
    # The run went on with the whole history, nothing lost.
    history = client.calls[-1][0]
    assert history[0] == Message("user", "Fix the bug")
    assert len([m for m in history if m.role == "tool"]) == 5


def test_an_empty_summary_leaves_the_history_alone():
    seen = []
    client = FakeClient(over_budget_script(ModelReply("   ")))

    Agent(client, tools=[small_tool()], context_budget=1_000, on_summary=seen.append).run("Go")

    assert seen[0].error == "the model returned an empty summary"
    assert len([m for m in client.calls[-1][0] if m.role == "tool"]) == 5


def test_no_summary_when_trimming_alone_gets_under_the_budget():
    # Results of 6,000 characters can be trimmed. Trimming the oldest one
    # frees about 2,000 tokens, which brings 31,000 back under 30,000.
    client = FakeClient(
        [big_call(1_000), big_call(3_000), big_call(5_000), big_call(31_000), "Done."]
    )
    seen = []

    Agent(client, tools=[big_tool()], context_budget=30_000, on_summary=seen.append).run("Go")

    assert seen == []
    assert len(client.calls) == 5  # no extra model call was made


def test_no_summary_when_the_old_part_is_too_small_to_be_worth_a_call():
    # Still far over the budget after trimming, but the old part is one
    # trimmed note. A summary of it would save nothing.
    client = FakeClient(
        [big_call(1_000), big_call(3_000), big_call(5_000), big_call(50_000), "Done."]
    )
    seen = []

    Agent(client, tools=[big_tool()], context_budget=30_000, on_summary=seen.append).run("Go")

    assert seen == []
    assert len(client.calls) == 5


def test_a_second_summary_builds_on_the_first_and_keeps_the_task_once():
    client = FakeClient(
        [
            *[small_call(500)] * 4,
            small_call(5_000),
            ModelReply("First summary."),
            small_call(500),
            small_call(5_000),
            ModelReply("Second summary."),
            "Done.",
        ]
    )

    Agent(client, tools=[small_tool()], context_budget=1_000).run("Fix the bug")

    second_request = client.calls[8][0][0].text
    assert "Summary of the work before this record:\nFirst summary." in second_request
    first_message = client.calls[-1][0][0].text
    assert first_message.count("Fix the bug") == 1
    assert "Second summary." in first_message
    assert "First summary." not in first_message
