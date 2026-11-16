"""Retries for temporary model failures.

`RetryingClient` wraps any ModelClient and is itself a ModelClient, so
the agent cannot tell the difference. This is the decorator pattern.
"""

import random
import time
from typing import Callable

from loopsmith.llm.base import (
    LONG_WAIT_SECONDS,
    Message,
    ModelClient,
    ModelError,
    ModelReply,
    ToolSpec,
)

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
                # "Retry in 18 hours" means a daily quota is used up. The
                # waits here are capped at a minute, so every retry would
                # fail the same way. Stop now and report it.
                if error.retry_after is not None and error.retry_after > LONG_WAIT_SECONDS:
                    raise
                delay = backoff_delay(attempt, error.retry_after)
                if self._on_retry is not None:
                    self._on_retry(error, delay, attempt)
                self._sleep(delay)
                attempt += 1


def backoff_delay(attempt: int, retry_after: float | None = None) -> float:
    """How long to wait before the next attempt.

    Double the wait each time (2s, 4s, 8s...) and add a random part, so
    that many clients that failed together do not all retry at the same
    instant. If the server said how long to wait, wait at least that long.

    The server's figure is a minimum, never a shortcut. Gemini sometimes
    suggests a wait of under a second, and retrying that quickly just
    fails again and uses up one of the few attempts.
    """
    delay = BASE_DELAY_SECONDS * (2 ** (attempt - 1))
    delay += random.uniform(0, delay / 2)
    if retry_after is not None:
        delay = max(delay, retry_after)
    return min(delay, MAX_DELAY_SECONDS)
