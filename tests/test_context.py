from loopsmith.context import (
    KEEP_RECENT_TOOL_TURNS,
    MIN_TRIM_CHARS,
    TrimReport,
    estimate_tokens,
    trim_history,
    trim_target,
)
from loopsmith.llm import Message, ToolCall, ToolResult

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


def test_the_trim_target_is_half_the_budget():
    assert trim_target(30_000) == 15_000
