"""Retries for temporary model failures.

`RetryingClient` wraps any ModelClient and is itself a ModelClient, so
the agent cannot tell the difference. This is the decorator pattern.
"""

import random
import time
from typing import Callable

from loopsmith.llm.base import Message, ModelClient, ModelError, ModelReply, ToolSpec

MAX_ATTEMPTS = 4
BASE_DELAY_SECONDS = 2.0
MAX_DELAY_SECONDS = 60.0

# Called before each wait with (error, seconds to wait, attempt number).
RetryObserver = Callable[[ModelError, float, int], None]


class RetryingClient:
    def __init__(
        self,
        inner: ModelClient,
        max_attempts: int = MAX_ATTEMPTS,
        sleep: Callable[[float], None] = time.sleep,
        on_retry: RetryObserver | None = None,
    ):
        self._inner = inner
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._on_retry = on_retry

    def generate(
        self,
        messages: list[Message],
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> ModelReply:
        attempt = 1
        while True:
            try:
                return self._inner.generate(messages, system=system, tools=tools)
            except ModelError as error:
                # A bad key or an unknown model fails the same way every
                # time, so only errors marked retryable are tried again.
                if not error.retryable or attempt >= self._max_attempts:
                    raise
                delay = backoff_delay(attempt, error.retry_after)
                if self._on_retry is not None:
                    self._on_retry(error, delay, attempt)
                self._sleep(delay)
                attempt += 1


def backoff_delay(attempt: int, retry_after: float | None = None) -> float:
    """How long to wait before the next attempt.

    If the server said how long to wait, use that. Otherwise double the
    wait each time (2s, 4s, 8s...) and add a random part, so that many
    clients that failed together do not all retry at the same instant.
    """
    if retry_after is not None:
        return min(retry_after, MAX_DELAY_SECONDS)
    delay = BASE_DELAY_SECONDS * (2 ** (attempt - 1))
    return min(delay + random.uniform(0, delay / 2), MAX_DELAY_SECONDS)
