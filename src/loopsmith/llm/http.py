"""A minimal JSON-over-HTTP helper built on the standard library.

Kept separate from the provider code so tests can replace it with a fake
and never touch the network.
"""

import json
import urllib.error
import urllib.request
from typing import Callable

from loopsmith.llm.base import ModelError

# (method, url, headers, body) -> (status code, parsed JSON body)
Transport = Callable[[str, str, dict[str, str], dict | None], tuple[int, dict]]

DEFAULT_TIMEOUT_SECONDS = 60


def _parse_json(raw: bytes) -> dict:
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def send_json(
    method: str, url: str, headers: dict[str, str], body: dict | None
) -> tuple[int, dict]:
    """Send one request and return (status, JSON body).

    HTTP error statuses (4xx, 5xx) are returned, not raised, so the caller
    can decide what each one means. Only a failure to reach the server
    raises ModelError.
    """
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    for name, value in headers.items():
        request.add_header(name, value)
    if data is not None:
        request.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT_SECONDS) as response:
            return response.status, _parse_json(response.read())
    except urllib.error.HTTPError as error:
        return error.code, _parse_json(error.read())
    except (urllib.error.URLError, TimeoutError) as error:
        reason = getattr(error, "reason", error)
        raise ModelError(
            f"Could not reach the model server: {reason}", retryable=True
        ) from error
