"""Tests for the HTTP helper, run against a real server on localhost."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from loopsmith.llm import ModelError
from loopsmith.llm.http import send_json


class EchoHandler(BaseHTTPRequestHandler):
    """Replies with what it received. The path picks the status code."""

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        received = json.loads(self.rfile.read(length))
        status = 429 if self.path == "/limited" else 200
        payload = {
            "received": received,
            "api_key_header": self.headers.get("x-goog-api-key"),
            "content_type": self.headers.get("Content-Type"),
        }
        self._reply(status, json.dumps(payload).encode())

    def do_GET(self):
        self._reply(200, b"this is not json")

    def _reply(self, status: int, data: bytes):
        self.send_response(status)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass  # keep test output quiet


@pytest.fixture
def server_url(monkeypatch):
    # Make sure requests to localhost never go through a proxy.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    server = HTTPServer(("127.0.0.1", 0), EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_post_sends_json_body_and_headers(server_url):
    status, body = send_json(
        "POST", f"{server_url}/ok", {"x-goog-api-key": "k"}, {"hello": "world"}
    )

    assert status == 200
    assert body["received"] == {"hello": "world"}
    assert body["api_key_header"] == "k"
    assert body["content_type"] == "application/json"


def test_error_status_is_returned_not_raised(server_url):
    status, body = send_json("POST", f"{server_url}/limited", {}, {"a": 1})

    assert status == 429
    assert body["received"] == {"a": 1}


def test_non_json_response_becomes_empty_dict(server_url):
    status, body = send_json("GET", f"{server_url}/anything", {}, None)

    assert status == 200
    assert body == {}


def test_unreachable_server_raises_retryable_model_error(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")

    # Port 1 on localhost has nothing listening.
    with pytest.raises(ModelError) as error_info:
        send_json("GET", "http://127.0.0.1:1/", {}, None)

    assert error_info.value.retryable is True
