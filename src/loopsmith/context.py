"""Context management: keeping the conversation history within a budget.

Every model call sends the whole history again, so a large tool result
is paid for on every later step. When the history passes a token budget,
the oldest large tool results are replaced with a short note.

Nothing is trimmed until the budget is passed, and then enough is
trimmed in one go to get well below it. Between trims the start of each
request is identical to the last one, which lets the provider's prompt
cache keep working.
"""

from dataclasses import dataclass, replace

from loopsmith.llm.base import Message, ToolResult

# Trim once a request has used more input tokens than this.
DEFAULT_CONTEXT_BUDGET = 30_000

# The tool results of the most recent turns are never trimmed: the model
# is most likely still working with them.
KEEP_RECENT_TOOL_TURNS = 3

# Results shorter than this are left alone. Replacing them saves little.
MIN_TRIM_CHARS = 1_000

# A rough figure for sizing one tool result. The exact count for a whole
# request comes from the provider; this is only used to decide how many
# results to trim. Code and command output measured about 3 on Gemini.
CHARS_PER_TOKEN = 3


@dataclass(frozen=True)
class TrimReport:
    results_trimmed: int = 0
    tokens_freed: int = 0  # an estimate, from CHARS_PER_TOKEN


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def trim_target(budget: int) -> int:
    """How far down to trim once the budget is passed: half of it.

    Trimming to just under the budget would mean trimming again on the
    next step. Going well below it makes trims rare. A thermostat works
    the same way: it switches on at one temperature and off at another.
    """
    return budget // 2


def trim_history(
    messages: list[Message],
    tokens_to_free: int,
    keep_recent: int = KEEP_RECENT_TOOL_TURNS,
) -> tuple[list[Message], TrimReport]:
    """Replace the oldest large tool results with a short note.

    Works from the oldest result forward and stops once about
    `tokens_to_free` tokens have been freed, or when nothing eligible is
    left. Returns a new list; the one passed in is not changed.

    Only the text of a result is replaced. The result itself stays, with
    its name and id, because the provider rejects a history in which a
    tool call has no matching result. The model's own turns are never
    touched.
    """
    tool_positions = [i for i, message in enumerate(messages) if message.role == "tool"]
    eligible = tool_positions[: max(len(tool_positions) - keep_recent, 0)]

    trimmed = list(messages)
    count = 0
    freed = 0
    for position in eligible:
        if freed >= tokens_to_free:
            break
        results = []
        for result in messages[position].tool_results:
            if freed < tokens_to_free and len(result.content) >= MIN_TRIM_CHARS:
                note = _note_for(result)
                freed += estimate_tokens(result.content) - estimate_tokens(note)
                count += 1
                # `replace` copies a frozen dataclass with one field
                # changed, like a "wither" method on a Java record.
                result = replace(result, content=note)
            results.append(result)
        trimmed[position] = replace(messages[position], tool_results=tuple(results))

    return trimmed, TrimReport(results_trimmed=count, tokens_freed=freed)


def _note_for(result: ToolResult) -> str:
    return (
        f"[Old output removed to save space: {len(result.content):,} characters "
        f"from {result.name}. Run the tool again if you still need it.]"
    )
