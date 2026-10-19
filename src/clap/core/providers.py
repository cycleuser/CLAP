"""Provider layer (CLAP-NEXT F6): local ollama + online OpenAI-compat endpoints.

Security invariants (handoff.md 硬约束):

- API keys are read ONLY from:
  1. env `CLAP_ONLINE_BASE_URL` + `CLAP_ONLINE_API_KEY` (generic "online"),
  2. `~/.config/opencode/opencode.json` (`providers.<id>.options`.
     `baseURL`/`apiKey`),
  3. `~/.pi/agent/models.json` (`providers.<id>.baseUrl`) with the key taken
     from pi's sibling auth store `~/.pi/agent/auth.json` (`<id>.key`) —
     documented extension: models.json carries the model registry but not
     keys.
- Key values are never written into sessions, logs, exports or screenshots;
  display is masked (`masked_key`); `repr`/`str` never expose the key.
- `privacy_mode` (default in session manifests) disables online providers.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

PROBE_TIMEOUT_S = 8.0
CHAT_TIMEOUT_S = 120.0


class ProviderError(Exception):
    """Raised when a provider call fails (network/auth/protocol)."""


class Provider:
    """Abstract chat provider: streaming chat, model list, availability."""

    id: str = "abstract"
    name: str = "abstract"
    base: str = ""

    def chat(self, messages, model: str, stream: bool = True):
        """Yield assistant text deltas; records usage into `last_meta`."""
        raise NotImplementedError

    def list_models(self) -> list[str]:
        raise NotImplementedError

    def is_available(self) -> bool:
        raise NotImplementedError

    def masked_key(self) -> str:
        return ""

    def __repr__(self):  # keys must never leak through any object dump
        return f"<Provider {self.id} base={self.base!r}>"


class OllamaProvider(Provider):
    """Ollama over the native protocol (local by default).

    `host` accepts any ollama-native endpoint (e.g. the ollama.com cloud);
    `headers` may carry auth (in-memory only, never logged). On this machine
    no local server runs, so scenario runs must use a reachable host —
    recorded honestly in results.
    """

    def __init__(self, host: str = "http://127.0.0.1:11434", headers: dict | None = None, pid: str | None = None):
        self.id = pid or "ollama-local"
        self.name = "Ollama (local)"
        self.base = host
        self._headers = dict(headers or {})
        self.last_meta: dict = {}

    def _client(self):
        import ollama

        return ollama.Client(host=self.base, headers=self._headers) if self._headers else ollama.Client(host=self.base)

    def chat(self, messages, model: str, stream: bool = True):

        self.last_meta = {"provider": self.id, "model": model, "tokens": {"in": None, "out": None}}
        t0 = time.time()
        try:
            res = self._client().chat(model=model, messages=messages, stream=stream)
            stream_iter = res if stream else [res]
            for chunk in stream_iter:
                try:
                    if chunk.get("done"):
                        self.last_meta = {
                            "provider": self.id,
                            "model": model,
                            "latency_ms": int((time.time() - t0) * 1000),
                            "tokens": {
                                "in": chunk.get("prompt_eval_count"),
                                "out": chunk.get("eval_count"),
                            },
                        }
                except (AttributeError, TypeError):
                    pass
                try:
                    text = chunk["message"]["content"]
                except (KeyError, TypeError, IndexError):
                    text = getattr(getattr(chunk, "message", None), "content", "")
                if text:
                    yield text
        except KeyboardInterrupt:
            raise
        except Exception as e:
            raise ProviderError(f"{self.id}: {e}") from e

    def list_models(self) -> list[str]:
        try:
            data = self._client().list()
            return [m["model"] for m in data.get("models", [])]
        except Exception:
            return []

    def is_available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base}/api/tags", timeout=PROBE_TIMEOUT_S) as r:
                return r.status == 200
        except Exception:
            return False


class OpenAICompatProvider(Provider):
    """OpenAI-compatible chat endpoint (streaming SSE).

    Used for every online endpoint (`deepseek`, `ollama-cloud`,
    `cycleuser-halogen`, `cycleuser-strata`, `moxing`, env-configured
    "online"). The api_key is held locally and NEVER logged or persisted.
    """

    def __init__(self, pid: str, base_url: str, api_key: str = "", name: str = ""):
        self.id = pid
        self.name = name or pid
        self.base = base_url.rstrip("/")
        self._api_key = api_key or ""
        self.last_meta: dict = {}

    def masked_key(self) -> str:
        if not self._api_key:
            return "<none>"
        tail = self._api_key[-4:] if len(self._api_key) >= 8 else "****"
        return f"***len{len(self._api_key)}…{tail}"

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    def chat(self, messages, model: str, stream: bool = True):
        self.last_meta = {"provider": self.id, "model": model, "tokens": {"in": None, "out": None}}
        t0 = time.time()
        url = f"{self.base}/chat/completions"
        body = {"model": model, "messages": messages, "stream": bool(stream)}
        if stream:
            # ask for usage in the final chunk where the server supports it
            body["stream_options"] = {"include_usage": True}
        req = urllib.request.Request(
            url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), headers=self._headers()
        )
        try:
            resp = urllib.request.urlopen(req, timeout=CHAT_TIMEOUT_S)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", "ignore")[:200]
                # strip any echo'd credentials defensively
                detail = detail.replace(self._api_key, "***") if self._api_key else detail
            except Exception:
                pass
            raise ProviderError(f"{self.id}: HTTP {e.code}{': ' + detail if detail else ''}") from e
        except Exception as e:
            raise ProviderError(f"{self.id}: {e}") from e

        out_tokens = None
        prompt_tokens = None

        def _gen():
            nonlocal out_tokens, prompt_tokens
            try:
                for raw in resp:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        obj = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    usage = obj.get("usage")
                    if isinstance(usage, dict):
                        prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                        out_tokens = usage.get("completion_tokens", out_tokens)
                    choices = obj.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content")
                    if delta:
                        yield delta
            finally:
                try:
                    resp.close()
                except Exception:
                    pass
                self.last_meta = {
                    "provider": self.id,
                    "model": model,
                    "latency_ms": int((time.time() - t0) * 1000),
                    "tokens": {"in": prompt_tokens, "out": out_tokens},
                }

        return _gen() if stream else list(_gen())

    def list_models(self) -> list[str]:
        try:
            req = urllib.request.Request(f"{self.base}/models", headers=self._headers())
            with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_S) as r:
                data = json.loads(r.read())
            return [m.get("id", "") for m in data.get("data", [])]
        except Exception:
            return []

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base}/models", headers=self._headers())
            with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_S) as r:
                return r.status == 200
        except Exception:
            return False


# --------------------------------------------------------------------------
# discovery (three permitted key/config sources; keys stay in memory only)
# --------------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def discover_providers() -> dict[str, Provider]:
    """Build the provider registry: local ollama + all configured online ends.

    Order: always `ollama-local`; then env CLAP_ONLINE_*; then opencode.json
    providers; then pi models.json (+ auth.json keys). Unreachable/dup ids
    are tolerated (later sources do not overwrite earlier ones).
    """
    providers: dict[str, Provider] = {"ollama-local": OllamaProvider()}

    # 1. env (generic)
    env_base = os.environ.get("CLAP_ONLINE_BASE_URL", "")
    env_key = os.environ.get("CLAP_ONLINE_API_KEY", "")
    if env_base:
        providers.setdefault("online", OpenAICompatProvider("online", env_base, env_key, "env-configured online"))

    home = Path.home()
    seen_bases: set[str] = {""}
    seen_bases.add("http://127.0.0.1:11434")
    seen_bases.add("http://127.0.0.1:11434/v1")  # ollama-local speaks the native api

    # 2. opencode.json
    oc = _load_json(home / ".config" / "opencode" / "opencode.json")
    for pid, cfg in (oc.get("provider") or {}).items():
        opts = (cfg or {}).get("options") or {}
        base = opts.get("baseURL", "")
        if not base or base.rstrip("/") in seen_bases:
            continue
        seen_bases.add(base.rstrip("/"))
        providers.setdefault(pid, OpenAICompatProvider(pid, base, opts.get("apiKey", ""), (cfg or {}).get("name", pid) or pid))

    # 3. pi models.json (+ auth.json for keys)
    models = _load_json(home / ".pi" / "agent" / "models.json")
    auth = _load_json(home / ".pi" / "agent" / "auth.json")
    for pid, cfg in (models.get("providers") or {}).items():
        base = (cfg or {}).get("baseUrl", "")
        if not base:
            continue
        if base.rstrip("/") in seen_bases:
            continue
        seen_bases.add(base.rstrip("/"))
        key = (auth.get(pid) or {}).get("key", "") or cfg.get("apiKey", "")
        providers.setdefault(pid, OpenAICompatProvider(pid, base, key, pid))

    # the ollama/ollama-local native tag namespace never means an online provider
    providers.pop("ollama", None)

    return providers


def provider_from_names(registry: dict, provider_id: str | None) -> Provider:
    if provider_id and provider_id in registry:
        return registry[provider_id]
    if provider_id:
        raise ProviderError(f"provider {provider_id!r} not configured (have: {sorted(registry)})")
    return registry.get("ollama-local") or next(iter(registry.values()))


def masked_endpoint_table(registry: dict) -> list[dict]:
    """Provider endpoint inventory (P3/S6/T4): config paths, never key values."""
    rows = []
    for pid, p in sorted(registry.items()):
        rows.append(
            {
                "id": pid,
                "base": p.base,
                "available": p.is_available(),
                "models": p.list_models(),
                "masked_key": p.masked_key() if hasattr(p, "masked_key") else "",
                "kind": "ollama-native" if isinstance(p, OllamaProvider) else "openai-compat",
            }
        )
    return rows


class FallbackRouter:
    """manual/fallback routing (F6): try primary first; on failure fall back
    to local; every switch is recorded as a manifest-history event."""

    def __init__(self, primary: Provider, local: OllamaProvider | None, privacy_mode: bool = True):
        self.primary = primary
        self.local = local
        self.privacy_mode = privacy_mode
        self.events: list[dict] = []

    def _online(self, p: Provider) -> bool:
        return not isinstance(p, OllamaProvider)

    def chat(self, messages, model: str, stream: bool = True):
        """Yield streamed chunks; switch to local on primary failure.

        Generator form: all provider probing happens during iteration, so
        callers consume this like any other chunk stream; routing events are
        collected in `self.events` for the session manifest.
        """
        if self.privacy_mode and self._online(self.primary):
            self.events.append(
                {"event": "privacy_block", "ts": int(time.time() * 1000), "detail": {"provider": self.primary.id}}
            )
            if self.local is None:
                raise ProviderError("online provider requested but privacy_mode=true and no local fallback")
            yield from self.local.chat(messages, model, stream=True)
            return

        started = False
        try:
            for chunk in self.primary.chat(messages, model, stream=stream):
                started = True
                yield chunk
            if not started:
                raise ProviderError(f"{self.primary.id}: empty response")
        except (ProviderError, urllib.error.URLError, urllib.error.HTTPError, OSError) as e:
            if started:
                raise ProviderError(f"{self.primary.id}: {e}") from e
            if self.local is not None and self.local is not self.primary:
                self.events.append(
                    {
                        "event": "fallback",
                        "ts": int(time.time() * 1000),
                        "detail": {"from": self.primary.id, "to": self.local.id, "reason": str(e)[:120]},
                    }
                )
                yield from self.local.chat(messages, model, stream=True)
            else:
                raise
