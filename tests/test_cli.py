import io

import pytest

from loopsmith.cli import ask_on_terminal, cached_note, main, print_retry, system_prompt
from loopsmith.config import API_KEY_VAR
from loopsmith.llm import Message, ModelError, ModelReply, ToolCall, Usage
from loopsmith.llm.fake import FakeClient
from loopsmith.subagent import SUB_AGENT_SYSTEM_PROMPT


class FailingClient:
    def generate(self, messages, system=None, tools=None):
        raise ModelError("Rate limit reached.", status=429, retryable=True)


class ListingClient(FakeClient):
    def list_models(self):
        return ["model-a", "model-b"]


def test_sends_task_to_model_and_prints_reply(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    client = FakeClient(["Hello from the model"])

    exit_code = main(["say hello"], client=client)

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "Hello from the model"
    history, system, tools = client.calls[0]
    assert history == [Message("user", "say hello")]
    assert system == system_prompt()
    assert {tool.name for tool in tools} == {
        "list_files",
        "read_file",
        "edit_file",
        "write_file",
        "run_command",
        "delegate",
    }


def test_agent_reads_a_real_file_from_the_current_folder(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("the secret word is mango", encoding="utf-8")
    client = FakeClient([ToolCall("read_file", {"path": "notes.txt"}), "The word is mango."])

    exit_code = main(["what is the secret word?"], client=client)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "The word is mango."
    # The tool trace goes to stderr, keeping stdout clean.
    assert "read_file(path='notes.txt')" in captured.err
    # The model received the file's contents as the tool result.
    tool_message = client.calls[1][0][-1]
    assert tool_message.tool_results[0].content == "the secret word is mango"


def test_hitting_the_step_limit_exits_with_failure(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([ToolCall("list_files", {}) for _ in range(5)])

    exit_code = main(["loop", "--max-steps", "2"], client=client)

    assert exit_code == 1
    assert "Stopped after 2 steps" in capsys.readouterr().out


def test_model_error_is_reported_cleanly(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["say hello"], client=FailingClient())

    assert exit_code == 1
    assert "Rate limit reached" in capsys.readouterr().err


def test_fails_clearly_without_key(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(API_KEY_VAR, raising=False)
    # Pretend no settings file exists anywhere on this machine.
    monkeypatch.setattr("loopsmith.config.find_env_file", lambda start=None: None)

    exit_code = main(["say hello"])

    assert exit_code == 1
    assert API_KEY_VAR in capsys.readouterr().err


def test_list_models_prints_names(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)

    exit_code = main(["--list-models"], client=ListingClient([]))

    assert exit_code == 0
    assert capsys.readouterr().out.split() == ["model-a", "model-b"]


def test_no_arguments_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main([])

    assert exit_info.value.code == 2


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])

    assert exit_info.value.code == 0
    assert "loopsmith" in capsys.readouterr().out


# --- Approval ---


class FakeTerminal(io.StringIO):
    """Stands in for the keyboard: a stdin that reports being a terminal."""

    def isatty(self):
        return True


WRITE_CALL = ToolCall("write_file", {"path": "new.txt", "content": "hello\n"})


def test_yes_flag_lets_changes_through(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    client = FakeClient([WRITE_CALL, "Created the file."])

    exit_code = main(["create new.txt", "--yes"], client=client)

    assert exit_code == 0
    assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "hello\n"


def test_user_can_approve_a_change_on_the_terminal(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", FakeTerminal("y\n"))
    client = FakeClient([WRITE_CALL, "Created the file."])

    main(["create new.txt"], client=client)

    assert (tmp_path / "new.txt").exists()
    shown = capsys.readouterr().err
    assert "Create file new.txt" in shown
    assert "+hello" in shown
    assert "Allow? [y/N]" in shown


@pytest.mark.parametrize("answer", ["n\n", "\n", "maybe\n", ""])
def test_anything_but_yes_declines_the_change(monkeypatch, tmp_path, capsys, answer):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", FakeTerminal(answer))
    client = FakeClient([WRITE_CALL, "I did not create it."])

    main(["create new.txt"], client=client)

    assert not (tmp_path / "new.txt").exists()
    assert "declined" in client.calls[1][0][-1].tool_results[0].content


def test_without_a_terminal_changes_are_refused(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO("y\n"))  # a pipe, not a terminal
    client = FakeClient([WRITE_CALL, "I could not create it."])

    main(["create new.txt"], client=client)

    assert not (tmp_path / "new.txt").exists()
    assert "--yes" in capsys.readouterr().err


def test_ask_on_terminal_accepts_yes_in_any_case(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", FakeTerminal("YES\n"))

    assert ask_on_terminal(WRITE_CALL, "Create file new.txt") is True


def test_token_usage_goes_to_stderr_not_stdout(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    client = FakeClient(
        [
            ModelReply("", (ToolCall("list_files", {}),), usage=Usage(1500, 20)),
            ModelReply("All done.", usage=Usage(1800, 40, cached_tokens=1200)),
        ]
    )

    exit_code = main(["look around"], client=client)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "All done."
    # A step with nothing cached says nothing about the cache.
    assert "step 1: 1,500 tokens in, 20 out" in captured.err
    assert "step 2: 1,800 tokens in (1,200 cached), 40 out" in captured.err
    # 1,200 of 3,300 is 36%.
    assert "tokens: 3,300 in (1,200 cached, 36%), 60 out; last request 1,800" in captured.err


def test_a_retry_after_an_http_error_shows_the_status(capsys):
    print_retry(ModelError("busy", status=503, retryable=True), delay=2.4, attempt=1)

    assert "model call failed (HTTP 503); retrying in 2s (attempt 1)" in capsys.readouterr().err


def test_a_retry_without_an_http_status_shows_what_happened(capsys):
    # A timeout, a lost connection or a malformed tool call has no HTTP
    # status at all; the error's own message says what went wrong.
    error = ModelError("Could not reach the model server: timed out", retryable=True)

    print_retry(error, delay=4.0, attempt=2)

    err = capsys.readouterr().err
    assert (
        "model call failed (Could not reach the model server: timed out); "
        "retrying in 4s (attempt 2)"
    ) in err
    assert "None" not in err


def test_a_long_retry_reason_is_shortened(capsys):
    print_retry(ModelError("x" * 200, retryable=True), delay=2.0, attempt=1)

    err = capsys.readouterr().err
    assert "x" * 77 + "...)" in err
    assert "x" * 78 not in err


def test_the_prompt_says_file_text_is_not_instructions():
    assert "not instructions from the user" in system_prompt()


def test_the_prompts_ask_for_independent_tool_calls_together():
    # Each reply is one request, and requests are the scarce resource.
    assert "several tool calls in one reply" in system_prompt()
    assert "several tool calls in one reply" in SUB_AGENT_SYSTEM_PROMPT
    # Both also point to reading part of a long file.
    assert "start_line and end_line" in system_prompt()
    assert "start_line and end_line" in SUB_AGENT_SYSTEM_PROMPT


def test_the_cached_count_is_shown_only_when_something_was_cached():
    assert cached_note(Usage(1_000, 10)) == ""
    assert cached_note(Usage(9_000, 10, cached_tokens=5_120)) == " (5,120 cached)"


def test_a_sub_agent_step_shows_its_cached_tokens(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    ask = ToolCall("delegate", {"task": "Anything here?"})
    client = FakeClient(
        [
            ModelReply("", (ask,), usage=Usage(1_000, 10)),
            ModelReply("Nothing.", usage=Usage(900, 20, cached_tokens=600)),
            ModelReply("Done.", usage=Usage(1_100, 15)),
        ]
    )

    main(["look"], client=client)

    assert "      .. sub-agent step 1: 900 tokens in (600 cached), 20 out" in (
        capsys.readouterr().err
    )


def test_a_sub_agent_trim_is_shown_indented(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "big.txt").write_text("x" * 6_000, encoding="utf-8")
    ask = ToolCall("delegate", {"task": "What is in big.txt?"})
    client = FakeClient(
        [
            ModelReply("", (ask,), usage=Usage(1_000, 10)),
            *[read_big_file(n) for n in (1_000, 3_000, 5_000, 50_000)],
            "Only x.",
            "It is all x.",
        ]
    )

    main(["what is in big.txt"], client=client)

    err = capsys.readouterr().err
    # Indented like the sub-agent's other lines, not like the main agent's.
    assert "      .. trimmed 1 old tool result, about" in err
    assert "\n  .. trimmed" not in err


def test_no_usage_line_when_the_model_reports_no_counts(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)

    main(["say hello"], client=FakeClient(["Hello."]))

    assert "tokens" not in capsys.readouterr().err


def read_big_file(input_tokens: int) -> ModelReply:
    call = ToolCall("read_file", {"path": "big.txt"})
    return ModelReply("", (call,), usage=Usage(input_tokens, 10))


def test_trimming_is_reported_on_stderr(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "big.txt").write_text("x" * 6_000, encoding="utf-8")
    replies = [read_big_file(n) for n in (1_000, 3_000, 5_000, 50_000)]
    client = FakeClient([*replies, "Done."])

    exit_code = main(["read it four times"], client=client)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "Done."
    assert "trimmed 1 old tool result, about" in captured.err
    assert "the next request re-reads about" in captured.err
    assert "tokens without the cache" in captured.err
    first_result = next(m for m in client.calls[-1][0] if m.role == "tool")
    assert first_result.tool_results[0].content.startswith("[Old output removed")


def test_context_budget_flag_can_turn_trimming_off(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "big.txt").write_text("x" * 6_000, encoding="utf-8")
    replies = [read_big_file(n) for n in (1_000, 3_000, 5_000, 50_000)]
    client = FakeClient([*replies, "Done."])

    main(["read it four times", "--context-budget", "0"], client=client)

    assert "trimmed" not in capsys.readouterr().err
    first_result = next(m for m in client.calls[-1][0] if m.role == "tool")
    assert first_result.tool_results[0].content == "x" * 6_000


def read_small_file(input_tokens: int) -> ModelReply:
    call = ToolCall("read_file", {"path": "small.txt"})
    return ModelReply("", (call,), usage=Usage(input_tokens, 10))


def test_summarising_is_reported_on_stderr(monkeypatch, tmp_path, capsys):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "small.txt").write_text("y" * 900, encoding="utf-8")
    client = FakeClient(
        [
            *[read_small_file(500)] * 4,
            read_small_file(5_000),
            ModelReply("The file is all y.", usage=Usage(700, 40)),
            "Done.",
        ]
    )

    exit_code = main(["read it five times", "--context-budget", "1000"], client=client)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "Done."
    assert (
        "summarised 4 old messages (one extra model call: 700 tokens in, 40 out); "
        "the next request re-reads the whole history without the cache"
    ) in captured.err
    assert "The file is all y." in client.calls[-1][0][0].text


def test_a_sub_agent_run_is_shown_indented_and_counted_in_the_total(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("the secret word is mango", encoding="utf-8")
    ask = ToolCall("delegate", {"task": "What is the secret word in notes.txt?"})
    client = FakeClient(
        [
            ModelReply("", (ask,), usage=Usage(1_000, 10)),
            ModelReply("", (ToolCall("read_file", {"path": "notes.txt"}),), usage=Usage(400, 5)),
            ModelReply("The word is mango.", usage=Usage(900, 20)),
            ModelReply("It is mango.", usage=Usage(1_100, 15)),
        ]
    )

    exit_code = main(["find the secret word"], client=client)

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.strip() == "It is mango."
    # The sub-agent's work is indented further than the main agent's.
    assert "      .. sub-agent step 1: 400 tokens in, 5 out" in captured.err
    assert "      -> read_file(path='notes.txt')" in captured.err
    assert "  -> delegate(task='What is the secret word in notes.txt?')" in captured.err
    assert (
        "tokens: 3,400 in (0 cached), 50 out; last request 1,100; "
        "1 sub-agent run used 1,300 of the tokens in"
    ) in captured.err
    # The main agent saw the answer, not the file.
    result = next(m for m in client.calls[-1][0] if m.role == "tool").tool_results[0]
    assert result.content == "The word is mango."


def test_the_system_prompt_tells_the_model_about_delegate():
    assert "delegate" in system_prompt()
