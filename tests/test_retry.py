import pytest

from loopsmith.llm import Message, ModelError, ModelReply
from loopsmith.llm.retry import MAX_DELAY_SECONDS, RetryingClient, backoff_delay


class FlakyClient:
    """Raises the given errors in order, then answers."""

    def __init__(self, errors: list[ModelError]):
        self._errors = list(errors)
        self.attempts = 0

    def generate(self, messages, system=None, tools=None):
        self.attempts += 1
        if self._errors:
            raise self._errors.pop(0)
        return ModelReply(text="ok")


def rate_limited(retry_after=None) -> ModelError:
    return ModelError("Rate limit", status=429, retryable=True, retry_after=retry_after)


def test_retries_until_the_call_succeeds():
    inner = FlakyClient([rate_limited(), rate_limited()])
    waits = []

    reply = RetryingClient(inner, sleep=waits.append).generate([Message("user", "hi")])

    assert reply.text == "ok"
    assert inner.attempts == 3
    assert len(waits) == 2
    assert waits[1] > waits[0]  # each wait is longer than the last


def test_non_retryable_error_is_raised_at_once():
    inner = FlakyClient([ModelError("Bad key", status=403, retryable=False)])
    waits = []

    with pytest.raises(ModelError, match="Bad key"):
        RetryingClient(inner, sleep=waits.append).generate([Message("user", "hi")])

    assert inner.attempts == 1
    assert waits == []


def test_gives_up_after_max_attempts():
    inner = FlakyClient([rate_limited() for _ in range(10)])
    waits = []

    with pytest.raises(ModelError, match="Rate limit"):
        RetryingClient(inner, max_attempts=3, sleep=waits.append).generate(
            [Message("user", "hi")]
        )

    assert inner.attempts == 3
    assert len(waits) == 2


def test_uses_the_wait_the_server_asked_for():
    inner = FlakyClient([rate_limited(retry_after=7.0)])
    waits = []

    RetryingClient(inner, sleep=waits.append).generate([Message("user", "hi")])

    assert waits == [7.0]


def test_observer_is_told_about_each_retry():
    inner = FlakyClient([rate_limited(retry_after=30.0)])
    seen = []

    RetryingClient(
        inner,
        sleep=lambda seconds: None,
        on_retry=lambda error, delay, attempt: seen.append((error.status, delay, attempt)),
    ).generate([Message("user", "hi")])

    assert seen == [(429, 30.0, 1)]


def test_backoff_doubles_and_is_capped():
    assert 2.0 <= backoff_delay(1) <= 3.0
    assert 4.0 <= backoff_delay(2) <= 6.0
    assert 8.0 <= backoff_delay(3) <= 12.0
    assert backoff_delay(20) == MAX_DELAY_SECONDS


def test_the_servers_wait_wins_even_past_our_own_cap():
    # Retrying after 60 seconds when the server said 120 can only fail
    # again and waste an attempt.
    assert backoff_delay(1, retry_after=120.0) == 120.0
    assert backoff_delay(20, retry_after=200.0) == 200.0


def test_a_tiny_server_delay_does_not_shorten_the_backoff():
    # Seen in a real run: Gemini answered 429 with a suggested wait of
    # under a second, the retry went out at once and failed again.
    assert 2.0 <= backoff_delay(1, retry_after=0.0) <= 3.0
    assert 4.0 <= backoff_delay(2, retry_after=0.4) <= 6.0


def test_a_longer_server_delay_wins_over_the_backoff():
    assert backoff_delay(1, retry_after=34.0) == 34.0


def test_a_very_long_server_wait_is_not_retried():
    # "Retry in 18 hours" is a used-up daily quota. Waiting a minute and
    # trying again cannot work, so the error is raised at once.
    inner = FlakyClient([rate_limited(retry_after=64_903.0)])
    waits = []

    with pytest.raises(ModelError):
        RetryingClient(inner, sleep=waits.append).generate([Message("user", "hi")])

    assert inner.attempts == 1
    assert waits == []


def test_a_wait_of_a_few_minutes_is_retried_after_the_full_wait():
    inner = FlakyClient([rate_limited(retry_after=120.0)])
    waits = []

    RetryingClient(inner, sleep=waits.append).generate([Message("user", "hi")])

    assert waits == [120.0]
    assert inner.attempts == 2
