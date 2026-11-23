"""The provider's cache only helps when a request starts exactly like the
one before it. These tests check, through the real Gemini translation,
that the agent keeps it that way: each request only adds to the end of
the last one, and a trim changes nothing before the first trimmed result.
"""

import json

from loopsmith.agent import Agent
from loopsmith.cli import main
from loopsmith.llm import ToolCall
from loopsmith.llm.fake import FakeClient
from loopsmith.llm.gemini import GeminiClient
from loopsmith.tools import Tool


class ScriptedTransport:
    """Plays back Gemini replies in order and records every request body."""

    def __init__(self, replies: list[dict]):
        self._replies = list(replies)
        self.bodies: list[dict] = []

    def __call__(self, method, url, headers, body):
        # A copy through JSON: what would really be sent, and safe from any
        # later change to the objects inside the body.
        self.bodies.append(json.loads(json.dumps(body)))
        return 200, self._replies.pop(0)


def call_reply(number: int, input_tokens: int) -> dict:
    """A Gemini reply asking for the look tool, with a thought signature."""
    part = {
        "functionCall": {"name": "look", "args": {"n": number}},
        "thoughtSignature": f"signature-{number}",
    }
    return {
        "candidates": [{"content": {"role": "model", "parts": [part]}}],
        "usageMetadata": {"promptTokenCount": input_tokens, "candidatesTokenCount": 5},
    }


def text_reply(text: str) -> dict:
    return {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}


def look_tool(size: int) -> Tool:
    return Tool("look", "look tool", {"type": "object"}, run=lambda args: "z" * size)


def run(replies: list[dict], result_size: int, budget: int = 30_000) -> list[dict]:
    transport = ScriptedTransport(replies)
    client = GeminiClient("key", "some-model", transport)
    agent = Agent(client, [look_tool(result_size)], system="Be brief.", context_budget=budget)
    agent.run("Look four times")
    return transport.bodies


def everything_but_the_history(body: dict) -> dict:
    return {key: value for key, value in body.items() if key != "contents"}


def test_each_request_starts_exactly_like_the_one_before():
    replies = [call_reply(n, 1_000 * n) for n in (1, 2, 3, 4)] + [text_reply("Done.")]

    bodies = run(replies, result_size=500)

    assert len(bodies) == 5
    for before, after in zip(bodies, bodies[1:]):
        # The system prompt and the tool list never change during a run...
        assert everything_but_the_history(after) == everything_but_the_history(before)
        # ...and the history only grows at the end.
        assert after["contents"][: len(before["contents"])] == before["contents"]


def test_the_models_own_turns_go_back_exactly_as_they_came():
    replies = [call_reply(1, 1_000), text_reply("Done.")]

    bodies = run(replies, result_size=500)

    sent_back = bodies[1]["contents"][1]
    assert sent_back == {"role": "model", "parts": replies[0]["candidates"][0]["content"]["parts"]}


def test_a_trim_changes_the_request_only_from_the_trimmed_result_on():
    # The fourth request reports 50,000 tokens, over the budget, so the
    # first result (contents[2]) is trimmed before the fifth request.
    sizes = (1_000, 3_000, 5_000, 50_000)
    replies = [call_reply(n, size) for n, size in enumerate(sizes, start=1)]

    bodies = run([*replies, text_reply("Done.")], result_size=6_000)

    before, after = bodies[3], bodies[4]
    assert everything_but_the_history(after) == everything_but_the_history(before)
    # Still cached: the task and the model's first turn.
    assert after["contents"][:2] == before["contents"][:2]
    # The first change, where the cache stops.
    assert after["contents"][2] != before["contents"][2]
    assert "Old output removed" in json.dumps(after["contents"][2])
    # Everything after it is the same text as before, but it comes after a
    # change, so it is no longer at the cached start of the request.
    assert after["contents"][3 : len(before["contents"])] == before["contents"][3:]


def test_the_cli_sends_the_same_system_prompt_and_tools_on_every_step(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    read = ToolCall("read_file", {"path": "a.txt"})
    client = FakeClient([read, read, "Done."])

    main(["read a.txt twice"], client=client)

    systems = {system for _, system, _ in client.calls}
    tool_lists = [tools for _, _, tools in client.calls]
    assert len(systems) == 1
    assert tool_lists[0] == tool_lists[1] == tool_lists[2]
