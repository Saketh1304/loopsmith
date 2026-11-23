"""Context management: keeping the conversation history within a budget.

Every model call sends the whole history again, so a large tool result
is paid for on every later step. When the history passes a token budget,
the oldest large tool results are replaced with a short note.

Nothing is trimmed until the budget is passed, and then enough is
trimmed in one go to get well below it. Between trims the start of each
request is identical to the last one, which lets the provider's prompt
cache keep working.

Trimming cannot help when the history is long but made of many small
pieces. For that case the older part of the history is replaced with a
summary written by the model. The functions here only build and apply
that summary; the agent makes the model call.
"""

from dataclasses import dataclass, replace

from loopsmith.llm.base import Message, ToolResult, Usage

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


# Summarising costs a model call, so it is only done when the older part
# of the history is at least this share of the budget.
SUMMARY_MIN_SHARE = 4  # one quarter

# In the text sent to the summariser, each tool result is cut to this.
SUMMARY_RESULT_CHARS = 2_000

SUMMARY_SYSTEM_PROMPT = (
    "You are writing a summary of a coding agent's work so far, so that the "
    "agent can continue with a shorter history. Cover, in this order:\n"
    "1. The goal.\n"
    "2. What has been done, including every file created or changed.\n"
    "3. What was learned: facts about the code, test results, errors.\n"
    "4. What is left to do.\n"
    "Be specific. Keep file names, function names, commands and exact error "
    "messages. Do not invent anything that is not in the record. Stay under "
    "400 words."
)


@dataclass(frozen=True)
class SummaryReport:
    """What happened when the agent tried to summarise old history."""

    messages_replaced: int = 0
    usage: Usage | None = None  # the cost of the extra model call
    error: str | None = None  # set when no summary could be made


@dataclass(frozen=True)
class TrimReport:
    results_trimmed: int = 0
    tokens_freed: int = 0  # an estimate, from CHARS_PER_TOKEN
    # The provider's cache only covers the start of a request, up to the
    # first change. This estimates the history from the first trimmed
    # result to the end: the part the next request sends without the
    # cache's discount.
    uncached_tokens: int = 0


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
    first_changed = None  # where the cached start of the request ends
    for position in eligible:
        if freed >= tokens_to_free:
            break
        results = []
        for result in messages[position].tool_results:
            if freed < tokens_to_free and len(result.content) >= MIN_TRIM_CHARS:
                note = _note_for(result)
                freed += estimate_tokens(result.content) - estimate_tokens(note)
                count += 1
                if first_changed is None:
                    first_changed = position
                # `replace` copies a frozen dataclass with one field
                # changed, like a "wither" method on a Java record.
                result = replace(result, content=note)
            results.append(result)
        trimmed[position] = replace(messages[position], tool_results=tuple(results))

    uncached = 0 if first_changed is None else estimate_history_tokens(trimmed[first_changed:])
    report = TrimReport(results_trimmed=count, tokens_freed=freed, uncached_tokens=uncached)
    return trimmed, report


def _note_for(result: ToolResult) -> str:
    return (
        f"[Old output removed to save space: {len(result.content):,} characters "
        f"from {result.name}. Run the tool again if you still need it.]"
    )


def split_history(
    messages: list[Message], keep_recent: int = KEEP_RECENT_TOOL_TURNS
) -> tuple[list[Message], list[Message]]:
    """Split the history into the old part to summarise and the part to keep.

    The first message (the task) belongs to neither: it is always kept.
    The recent part holds the last `keep_recent` tool turns, each together
    with the assistant turn that asked for it. A tool call and its result
    are never separated, because the provider rejects one without the
    other.
    """
    tool_positions = [i for i, message in enumerate(messages) if message.role == "tool"]
    if len(tool_positions) <= keep_recent:
        return [], messages[1:]
    cut = tool_positions[-keep_recent] if keep_recent > 0 else len(messages)
    if cut > 1 and messages[cut - 1].role == "assistant":
        cut -= 1  # take the assistant turn that made these tool calls too
    return messages[1:cut], messages[cut:]


def estimate_history_tokens(messages: list[Message]) -> int:
    """A rough size for part of the history, from its character count."""
    chars = 0
    for message in messages:
        if message.raw is not None:
            # What is really sent for this turn, thought signatures included.
            chars += len(str(message.raw))
        else:
            chars += len(message.text)
            chars += sum(len(str(call.args)) for call in message.tool_calls)
        chars += sum(len(result.content) for result in message.tool_results)
    return chars // CHARS_PER_TOKEN


def worth_summarising(old: list[Message], budget: int) -> bool:
    return estimate_history_tokens(old) >= budget // SUMMARY_MIN_SHARE


def summary_request(task: str, earlier_summary: str, old: list[Message]) -> list[Message]:
    """Build the one message that asks the model for a summary.

    The old turns are written out as plain text, not replayed as real
    turns. Plain text works the same for every provider and has no
    ordering rules to break.
    """
    sections = [f"The task the agent was given:\n{task}"]
    if earlier_summary:
        sections.append(f"Summary of the work before this record:\n{earlier_summary}")
    sections.append(f"Record of the work, oldest first:\n{render_transcript(old)}")
    sections.append("Write the summary now.")
    return [Message("user", "\n\n".join(sections))]


def render_transcript(messages: list[Message]) -> str:
    lines = []
    for message in messages:
        if message.text:
            lines.append(f"{message.role}: {message.text}")
        for call in message.tool_calls:
            lines.append(f"assistant called {call.name} with {call.args}")
        for result in message.tool_results:
            kind = "error" if result.is_error else "result"
            lines.append(f"{result.name} {kind}: {_shorten(result.content)}")
    return "\n".join(lines)


def _shorten(text: str) -> str:
    if len(text) <= SUMMARY_RESULT_CHARS:
        return text
    omitted = len(text) - SUMMARY_RESULT_CHARS
    return f"{text[:SUMMARY_RESULT_CHARS]} [... {omitted:,} more characters]"


def task_message(task: str, summary: str = "") -> Message:
    """The first message of the history: the task, plus the summary if any.

    The summary is put inside the first user message, not added as a
    message of its own, so the turns still alternate between the user
    and the model. The task itself is always kept word for word.
    """
    if not summary:
        return Message("user", task)
    return Message(
        "user",
        f"{task}\n\n"
        "[The earlier part of this work was replaced by the summary below to "
        "save space. The most recent steps follow it in full.]\n"
        f"{summary}",
    )
