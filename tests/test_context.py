import json

from loopsmith.context import (
    KEEP_RECENT_TOOL_TURNS,
    MIN_TRIM_CHARS,
    SUMMARY_RESULT_CHARS,
    TrimmedResult,
    TrimReport,
    estimate_history_tokens,
    estimate_tokens,
    render_transcript,
    split_history,
    summary_request,
    task_message,
    trim_history,
    trim_target,
    worth_summarising,
)
from loopsmith.llm import Message, ToolCall, ToolResult
from loopsmith.llm.gemini import build_request_body

BIG = "x" * 6_000  # about 2,000 tokens


def tool_turn(content: str, call_id: str, name: str = "read_file") -> list[Message]:
    """One assistant turn asking for a tool, and the turn with its result."""
    return [
        Message("assistant", "", (ToolCall(name, {}, id=call_id),), raw=[{"id": call_id}]),
        Message("tool", tool_results=(ToolResult(name, content, id=call_id),)),
    ]


def history(*contents: str) -> list[Message]:
    messages = [Message("user", "do the task")]
    for number, content in enumerate(contents, start=1):
        messages += tool_turn(content, f"call-{number}")
    return messages


def result_texts(messages: list[Message]) -> list[str]:
    return [m.tool_results[0].content for m in messages if m.role == "tool"]


def test_the_oldest_large_result_is_replaced_with_a_note():
    messages = history(BIG, "b", "c", "d")

    trimmed, report = trim_history(messages, tokens_to_free=100)

    texts = result_texts(trimmed)
    assert texts[0] == (
        "[Old output removed to save space: 6,000 characters from read_file. "
        "Run the tool again if you still need it.]"
    )
    assert texts[1:] == ["b", "c", "d"]
    assert report.results_trimmed == 1
    assert report.tokens_freed == estimate_tokens(BIG) - estimate_tokens(texts[0])


def test_a_trimmed_result_keeps_its_name_id_and_error_flag():
    messages = [
        Message("user", "go"),
        Message("tool", tool_results=(ToolResult("run_command", BIG, is_error=True, id="c1"),)),
        *history("b", "c", "d")[1:],
    ]

    trimmed, _ = trim_history(messages, tokens_to_free=100)

    result = trimmed[1].tool_results[0]
    assert (result.name, result.id, result.is_error) == ("run_command", "c1", True)
    assert result.content.startswith("[Old output removed")


def test_the_most_recent_tool_turns_are_never_trimmed():
    messages = history(*[BIG] * KEEP_RECENT_TOOL_TURNS)

    trimmed, report = trim_history(messages, tokens_to_free=1_000_000)

    assert trimmed == messages
    assert report == TrimReport()


def test_small_results_are_left_alone():
    small = "y" * (MIN_TRIM_CHARS - 1)
    messages = history(small, BIG, "c", "d", "e")

    trimmed, report = trim_history(messages, tokens_to_free=1_000_000)

    texts = result_texts(trimmed)
    assert texts[0] == small
    assert texts[1].startswith("[Old output removed")
    assert report.results_trimmed == 1


def test_trimming_stops_once_enough_has_been_freed():
    messages = history(BIG, BIG, BIG, "d", "e", "f")

    trimmed, report = trim_history(messages, tokens_to_free=2_500)

    texts = result_texts(trimmed)
    assert texts[0].startswith("[Old output removed")
    assert texts[1].startswith("[Old output removed")
    assert texts[2] == BIG  # two were enough
    assert report.results_trimmed == 2


def test_when_too_little_can_be_freed_it_trims_what_it_can():
    messages = history(BIG, "b", "c", "d")

    _, report = trim_history(messages, tokens_to_free=1_000_000)

    assert report.results_trimmed == 1
    assert report.tokens_freed < 1_000_000


def test_several_results_in_one_turn_are_trimmed_separately():
    turn = Message(
        "tool",
        tool_results=(
            ToolResult("read_file", BIG, id="a"),
            ToolResult("list_files", "small", id="b"),
            ToolResult("run_command", BIG, id="c"),
        ),
    )
    messages = [Message("user", "go"), turn, *history("x", "y", "z")[1:]]

    trimmed, report = trim_history(messages, tokens_to_free=1_000_000)

    contents = [result.content for result in trimmed[1].tool_results]
    assert contents[0].startswith("[Old output removed")
    assert contents[1] == "small"
    assert contents[2].startswith("[Old output removed")
    assert [result.id for result in trimmed[1].tool_results] == ["a", "b", "c"]
    assert report.results_trimmed == 2


def test_the_original_list_and_the_models_own_turns_are_untouched():
    messages = history(BIG, "b", "c", "d")
    before = list(messages)

    trimmed, _ = trim_history(messages, tokens_to_free=100)

    assert messages == before
    assert trimmed is not messages
    # Assistant turns carry the provider's raw parts and must be replayed
    # exactly, so they are the very same objects.
    for old, new in zip(messages, trimmed):
        if old.role != "tool":
            assert new is old


def test_an_already_trimmed_result_is_not_trimmed_again():
    messages = history(BIG, "b", "c", "d")
    once, _ = trim_history(messages, tokens_to_free=100)

    twice, report = trim_history(once, tokens_to_free=100)

    assert twice == once
    assert report == TrimReport()


def test_a_trim_reports_the_history_after_it_as_no_longer_cached():
    # Messages: 0 task, 1-2 first turn, 3-4 second turn (the big result), ...
    messages = history("a", BIG, "c", "d", "e")

    trimmed, report = trim_history(messages, tokens_to_free=100)

    # The cache still covers everything before the trimmed result. From
    # that result to the end, the next request has changed.
    assert result_texts(trimmed)[1].startswith("[Old output removed")
    assert report.uncached_tokens == estimate_history_tokens(trimmed[4:])
    assert 0 < report.uncached_tokens < estimate_history_tokens(trimmed)


def test_the_cache_is_lost_from_the_first_trimmed_result_not_the_last():
    messages = history(BIG, BIG, "c", "d", "e")

    trimmed, report = trim_history(messages, tokens_to_free=1_000_000)

    assert report.results_trimmed == 2
    assert report.uncached_tokens == estimate_history_tokens(trimmed[2:])


def test_a_trimmed_result_keeps_the_removed_text_for_the_summariser_only():
    messages = history(BIG, "b", "c", "d")

    trimmed, _ = trim_history(messages, tokens_to_free=100)

    result = trimmed[2].tool_results[0]
    assert isinstance(result, TrimmedResult)
    assert result.content.startswith("[Old output removed")
    assert result.original == BIG
    # The model is sent only the note...
    assert BIG not in json.dumps(build_request_body(trimmed))
    # ...and the summariser sees what the output said (cut to 2,000).
    transcript = render_transcript(trimmed)
    assert "read_file result: " + "x" * SUMMARY_RESULT_CHARS in transcript
    assert "[Old output removed" not in transcript


def test_the_trim_target_is_half_the_budget():
    assert trim_target(30_000) == 15_000


def test_split_keeps_the_recent_tool_turns_with_their_assistant_turns():
    messages = history("a", "b", "c", "d", "e")

    old, recent = split_history(messages)

    # Five tool turns: the first two are old, the last three are kept.
    assert result_texts(old) == ["a", "b"]
    assert result_texts(recent) == ["c", "d", "e"]
    # The kept part starts with the assistant turn that made the call, so
    # no tool result is left without its call.
    assert recent[0].role == "assistant"
    assert recent[0].tool_calls[0].id == "call-3"
    # Nothing is lost or repeated, and the task is in neither part.
    assert [messages[0], *old, *recent] == messages


def test_split_has_nothing_old_when_the_history_is_short():
    messages = history(*["a"] * KEEP_RECENT_TOOL_TURNS)

    old, recent = split_history(messages)

    assert old == []
    assert recent == messages[1:]


def test_history_size_is_estimated_from_its_characters():
    messages = [
        Message("assistant", "x" * 30, (ToolCall("echo", {"text": "hi"}),)),
        Message("tool", tool_results=(ToolResult("echo", "y" * 60),)),
    ]
    args_chars = len(str({"text": "hi"}))

    assert estimate_history_tokens(messages) == (30 + args_chars + 60) // 3


def test_an_assistant_turn_is_sized_by_its_raw_parts_when_it_has_them():
    raw = [{"functionCall": {"name": "echo"}, "thoughtSignature": "s" * 300}]
    message = Message("assistant", "", (ToolCall("echo", {}),), raw=raw)

    assert estimate_history_tokens([message]) == len(str(raw)) // 3


def test_summarising_needs_an_old_part_of_at_least_a_quarter_of_the_budget():
    old = [Message("tool", tool_results=(ToolResult("echo", "y" * 3_000),))]  # 1,000 tokens

    assert worth_summarising(old, budget=4_000) is True
    assert worth_summarising(old, budget=4_004) is False
    assert worth_summarising([], budget=4_000) is False


def test_the_transcript_shows_text_tool_calls_results_and_errors():
    messages = [
        Message("assistant", "Let me look.", (ToolCall("read_file", {"path": "a.py"}),)),
        Message(
            "tool",
            tool_results=(
                ToolResult("read_file", "print(1)"),
                ToolResult("run_command", "No such file", is_error=True),
            ),
        ),
    ]

    assert render_transcript(messages) == (
        "assistant: Let me look.\n"
        "assistant called read_file with {'path': 'a.py'}\n"
        "read_file result: print(1)\n"
        "run_command error: No such file"
    )


def test_long_results_are_cut_short_in_the_transcript():
    long_text = "z" * (SUMMARY_RESULT_CHARS + 500)
    messages = [Message("tool", tool_results=(ToolResult("read_file", long_text),))]

    line = render_transcript(messages)

    assert line.endswith("[... 500 more characters]")
    assert line.count("z") == SUMMARY_RESULT_CHARS


def test_the_summary_request_is_one_user_message_with_the_task_and_the_record():
    old = history("found the bug")[1:]

    request = summary_request("Fix the bug", "", old)

    assert len(request) == 1
    assert request[0].role == "user"
    assert "The task the agent was given:\nFix the bug" in request[0].text
    assert "read_file result: found the bug" in request[0].text
    assert "Summary of the work before" not in request[0].text


def test_the_summary_request_carries_an_earlier_summary_forward():
    request = summary_request("Fix the bug", "Tests were failing.", history("x")[1:])

    assert "Summary of the work before this record:\nTests were failing." in request[0].text


def test_the_task_message_is_just_the_task_until_there_is_a_summary():
    assert task_message("Fix the bug") == Message("user", "Fix the bug")


def test_the_task_message_keeps_the_task_word_for_word_above_the_summary():
    message = task_message("Fix the bug", "The bug is in median().")

    assert message.role == "user"
    assert message.text.startswith("Fix the bug\n\n[The earlier part of this work")
    assert message.text.endswith("The bug is in median().")
