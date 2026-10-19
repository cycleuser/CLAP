"""P2/P3 provider layer tests: SSE parsing against a real local HTTP server,
key masking, availability probing, fallback routing (all offline, loopback)."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from clap.core.providers import (
    FallbackRouter,
    OllamaProvider,
    OpenAICompatProvider,
    ProviderError,
    discover_providers,
)


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    """Minimal SSE server: /v1/models -> list; /v1/chat/completions -> 2 deltas + usage."""

    fail_with: int = 0  # set by tests to force an HTTP failure

    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        if self.path.endswith("/models"):
            body = json.dumps({"data": [{"id": "fake-model"}]}).encode()
            code = self.fail_with or 200
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.fail_with:
            self.send_response(self.fail_with)
            body = b'{"error": {"message": "Unauthorized"}}'
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for payload in (
            {"choices": [{"delta": {"content": "你好"}}]},
            {"choices": [{"delta": {"content": "，世界"}}]},
            {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 4}},
            "[DONE]",
        ):
            if payload == "[DONE]":
                self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())


@pytest.fixture()
def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("CLAP_ONLINE_BASE_URL", raising=False)
    monkeypatch.delenv("CLAP_ONLINE_API_KEY", raising=False)


def test_streaming_and_usage_capture(fake_server):
    p = OpenAICompatProvider("fake", fake_server, api_key="sk-test-123456789")
    out = list(p.chat([{"role": "user", "content": "hi"}], model="fake-model", stream=True))
    assert out == ["你好", "，世界"]
    assert p.last_meta["tokens"] == {"in": 11, "out": 4}
    assert p.last_meta["provider"] == "fake"
    assert p.last_meta["model"] == "fake-model"


def test_masked_key_never_leaks(fake_server):
    p = OpenAICompatProvider("fake", fake_server, api_key="sk-test-123456789")
    masked = p.masked_key()
    assert "sk-test" not in masked
    assert repr(p).find("sk-test") < 0


def test_http_error_masked_in_message(fake_server):
    FakeOpenAIHandler.fail_with = 401
    try:
        p = OpenAICompatProvider("fake", fake_server, api_key="sk-test-123456789")
        with pytest.raises(ProviderError) as ei:
            list(p.chat([{"role": "user", "content": "hi"}], model="fake-model"))
        msg = str(ei.value)
        assert "sk-test" not in msg  # key never appears in error text
    finally:
        FakeOpenAIHandler.fail_with = 0


def test_is_available(fake_server):
    p = OpenAICompatProvider("fake", fake_server)
    assert p.is_available() is True
    assert p.list_models() == ["fake-model"]
    FakeOpenAIHandler.fail_with = 500
    try:
        assert p.is_available() is False
    finally:
        FakeOpenAIHandler.fail_with = 0


def test_env_provider_discovered(monkeypatch, fake_server):
    monkeypatch.setenv("CLAP_ONLINE_BASE_URL", fake_server)
    monkeypatch.setenv("CLAP_ONLINE_API_KEY", "sk-env-key-987654321")
    monkeypatch.delenv("CLAP_OLLAMA_LOCAL", raising=False)
    reg = discover_providers()
    # W2/W3: ollama defaults to the OpenAI-compatible cloud endpoint; the
    # local daemon only appears with CLAP_OLLAMA_LOCAL=1.
    assert "ollama-local" not in reg
    assert "ollama-cloud" in reg
    assert "online" in reg
    assert reg["online"].base == fake_server


def test_privacy_mode_blocks_online(fake_server):
    p = OpenAICompatProvider("fake", fake_server)

    class NullLocal(OllamaProvider):
        id = "fake-local"

        def chat(self, messages, model, stream=True):
            yield "本地回答"

    local2 = NullLocal()
    router = FallbackRouter(p, local2, privacy_mode=True)
    out = list(router.chat([{"role": "user", "content": "q"}], model="fake-model"))
    assert out == ["本地回答"]
    assert router.events[0]["event"] == "privacy_block"


def test_fallback_on_primary_failure(fake_server):
    FakeOpenAIHandler.fail_with = 503
    try:
        p = OpenAICompatProvider("fake", fake_server)

        class NullLocal(OllamaProvider):
            id = "fake-local"

            def chat(self, messages, model, stream=True):
                def gen():
                    yield "本地接管"
                    self.last_meta = {"provider": self.id, "model": model, "latency_ms": 5, "tokens": {"in": 1, "out": 1}}

                return gen()

        router = FallbackRouter(p, NullLocal(), privacy_mode=False)
        out = list(router.chat([{"role": "user", "content": "q"}], model="fake-model"))
        assert out == ["本地接管"]
        assert any(e["event"] == "fallback" and e["detail"]["from"] == "fake" for e in router.events)
    finally:
        FakeOpenAIHandler.fail_with = 0


def test_fallback_raise_without_local(fake_server):
    FakeOpenAIHandler.fail_with = 503
    try:
        p = OpenAICompatProvider("fake", fake_server)
        router = FallbackRouter(p, None, privacy_mode=False)
        with pytest.raises(ProviderError):
            list(router.chat([{"role": "user", "content": "q"}], model="fake-model"))
    finally:
        FakeOpenAIHandler.fail_with = 0


def test_discovery_masks_never_error(monkeypatch):
    """Registry discovery works on this machine without leaking key material."""
    monkeypatch.delenv("CLAP_OLLAMA_LOCAL", raising=False)
    reg = discover_providers()
    assert "ollama-local" not in reg  # API-only by default (W3)
    assert reg, "expected at least one configured provider"
    for pid, p in reg.items():
        assert pid and p.base
        dumped = repr(p) + str(p)
        assert "sk-" not in dumped
