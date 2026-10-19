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

    def list_models(self, timeout: float = PROBE_TIMEOUT_S) -> list[str]:
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

    def __init__(self, pid: str, base_url: str, api_key: str = "", name: str = "",
                 models: list[str] | None = None):
        self.id = pid
        self.name = name or pid
        self.base = base_url.rstrip("/")
        self._api_key = api_key or ""
        self._fallback_models = list(models or [])
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

    def list_models(self, timeout: float = PROBE_TIMEOUT_S) -> list[str]:
        try:
            req = urllib.request.Request(f"{self.base}/models", headers=self._headers())
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = json.loads(r.read())
            ids = [m.get("id", "") for m in data.get("data", []) if m.get("id")]
            return ids or list(self._fallback_models)
        except Exception:
            return list(self._fallback_models)

    def is_available(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base}/models", headers=self._headers())
            with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_S) as r:
                return r.status == 200
        except Exception:
            return False


# --------------------------------------------------------------------------
# discovery (W2/W3: unified OpenAI-compatible HTTP path + env assembler)
# --------------------------------------------------------------------------

# W3: the masked inventory is vendored INTO the repo (contains no keys), so
# discovery never depends on a machine-external path.
PROBE_LIST_BUNDLED = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "resources", "opencode_models.json")
)

# env var stem → provider identity in the inventory. The generic rule also
# accepts any `<PROVIDER>_API_KEY` whose stem names a listed provider.
ENV_PROVIDER_MAP = {
    "DEEPSEEK": "deepseek",
    "OLLAMA_CLOUD": "ollama-cloud",
    "OLLAMA": "ollama-cloud",
    "NVIDIA": "nvidia",
    "NVIDIA_NIM": "nvidia",
    "HF": "huggingface",
    "HUGGINGFACE": "huggingface",
    "MINIMAX": "minimax-cn",
    "MINIMAX_CN": "minimax-cn",
    "TENCENT": "tencent",
    "TENCENT_CODING_PLAN": "tencent-coding-plan",
    "TENCENT_TOKENHUB": "tencent-tokenhub",
    "TENCENT_HY3": "hy3-preview",
    "HY3": "hy3-preview",
    "MOXING": "moxing",
    "OPENCODE": "opencode",
    "CYCLEUSER_HALOGEN": "cycleuser-halogen",
    "HALOGEN": "cycleuser-halogen",
    "CYCLEUSER_STRATA": "cycleuser-strata",
    "STRATA": "cycleuser-strata",
}

_KEY_SUFFIXES = ("_API_KEY", "_TOKEN", "_KEY")

# inventory entries that speak the Anthropic protocol — an OpenAI-compatible
# client cannot talk to them, so they are surfaced as unsupported, not called.
ANTHROPIC_STYLE = {"minimax-cn"}


def _load_json(path: Path) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def mask_secret(value: str) -> str:
    """Display form of a credential: length + last 4 chars, never the value."""
    if not value:
        return "<none>"
    tail = value[-4:] if len(value) >= 8 else "****"
    return f"***len{len(value)}…{tail}"


def load_probe_list(path: str | None = None) -> dict:
    """Load the masked inventory {providers:[{id,baseURL}], models:[{providerID,modelID}]}.

    Lookup order (W3): explicit path → env `CLAP_PROBE_LIST` → bundled copy.
    Returns {"providers": {id: baseURL}, "models": {id: [modelID]}, "path": str|None}.
    """
    explicit = os.environ.get("CLAP_PROBE_LIST", "")
    candidates = [path] if path else [explicit, PROBE_LIST_BUNDLED]
    for cand in candidates:
        if not cand or not os.path.exists(cand):
            continue
        try:
            with open(cand, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        provs = {
            p["id"]: p.get("baseURL", "")
            for p in data.get("providers", [])
            if isinstance(p, dict) and p.get("id")
        }
        models: dict[str, list[str]] = {}
        for m in data.get("models", []):
            pid = (m or {}).get("providerID", "")
            mid = (m or {}).get("modelID", "")
            if pid and mid:
                models.setdefault(pid, []).append(mid)
        return {"providers": provs, "models": models, "path": cand}
    return {"providers": {}, "models": {}, "path": None}


def env_key_candidates(environ: dict | None = None) -> list[tuple[str, str]]:
    """Env vars shaped like credentials — NAMES returned with values held in
    memory only (`*_API_KEY` / `*_TOKEN` / `*_KEY`, non-empty, len >= 8)."""
    env = os.environ if environ is None else environ
    out = []
    for name, value in env.items():
        upper = name.upper()
        if upper == "CLAP_PROBE_LIST":
            continue
        if any(upper.endswith(suf) for suf in _KEY_SUFFIXES) and value and len(value) >= 8:
            out.append((name, value))
    return out


def env_stem_provider(env_name: str, known_ids) -> str | None:
    """Map an env var name to a provider identity present in the inventory."""
    stem = env_name
    for suf in _KEY_SUFFIXES:
        if stem.upper().endswith(suf):
            stem = stem[: -len(suf)]
            break
    upper = stem.upper()
    if upper in ENV_PROVIDER_MAP and ENV_PROVIDER_MAP[upper] in known_ids:
        return ENV_PROVIDER_MAP[upper]
    for cand in (upper.lower().replace("_", "-"), upper.lower()):
        if cand in known_ids:
            return cand
    return None


class EnvAssembler:
    """W2: register providers from env credentials via the vendored inventory."""

    def __init__(self, probe: dict | None = None, environ: dict | None = None):
        self.probe = probe if probe is not None else load_probe_list()
        self.environ = os.environ if environ is None else environ
        self.sources: dict[str, dict] = {}  # provider id -> masked display row
        self.registered: dict[str, Provider] = {}

    def _row(self, pid: str, base: str, env_name: str, masked: str, configured: bool, unsupported: str = "") -> dict:
        return {
            "provider": pid,
            "base": base,
            "env": env_name,  # NAME only
            "masked": masked,  # never the value
            "models": list(self.probe.get("models", {}).get(pid, [])),
            "configured": configured,
            "unsupported": unsupported,
        }

    def assemble(self) -> dict[str, Provider]:
        """Returns provider id -> provider, plus masked rows in `self.sources`."""
        registered: dict[str, Provider] = {}
        base_by_id = self.probe.get("providers", {})
        known = set(base_by_id)

        for pid, base in base_by_id.items():
            self.sources[pid] = self._row(pid, base, "", "<none>", False)

        for env_name, value in env_key_candidates(self.environ):
            pid = env_stem_provider(env_name, known)
            if pid is None:
                continue  # not an LLM identity in the inventory → ignored
            if pid in self.sources and self.sources[pid]["configured"]:
                self.sources[pid]["also_env"] = self.sources[pid].get("also_env", []) + [env_name]
                continue  # first credential wins
            if pid in ANTHROPIC_STYLE:
                self.sources[pid] = self._row(
                    pid, base_by_id[pid], env_name, mask_secret(value), True,
                    unsupported="anthropic-protocol endpoint",
                )
                continue
            prov = OpenAICompatProvider(pid, base_by_id[pid], value, f"{pid} (env)")
            registered[pid] = prov
            self.sources[pid] = self._row(pid, base_by_id[pid], env_name, prov.masked_key(), True)

        self.registered = registered
        return registered


def discover_providers(probe_path: str | None = None) -> dict[str, Provider]:
    """Build the unified provider registry (W2).

    Every provider is an OpenAI-compatible HTTP endpoint. Sources (later ones
    do not overwrite earlier ids):
      1. env `CLAP_ONLINE_BASE_URL` (+`CLAP_ONLINE_API_KEY`) as "online";
      2. env-assembled providers (W2 assembler via the vendored inventory);
      3. `~/.config/opencode/opencode.json` providers;
      4. `~/.pi/agent/models.json` (+ keys from `auth.json`);
      5. ollama = `https://ollama.com/v1` (registered as `ollama-cloud`);
      6. local ollama daemon, ONLY when `CLAP_OLLAMA_LOCAL=1`.
    """
    providers: dict[str, Provider] = {}
    seen: set[str] = {""}

    # 0) providers saved through the web settings panel (highest priority:
    #    an explicit choice in the UI should beat env/config discovery).
    from clap.core import settings_store

    saved = settings_store.load()
    for pid, row in saved.get("providers", {}).items():
        base = (row.get("base") or "").strip()
        if not base:
            continue
        providers[pid] = OpenAICompatProvider(
            pid, base, row.get("apiKey", ""), row.get("name") or pid, models=row.get("models")
        )
        seen.add(base.rstrip("/"))

    env_base = os.environ.get("CLAP_ONLINE_BASE_URL", "")
    if env_base:
        providers["online"] = OpenAICompatProvider(
            "online", env_base, os.environ.get("CLAP_ONLINE_API_KEY", ""), "env-configured online"
        )
        seen.add(env_base.rstrip("/"))

    for pid, prov in EnvAssembler(load_probe_list(probe_path)).assemble().items():
        providers.setdefault(pid, prov)
        seen.add(prov.base.rstrip("/"))

    home = Path.home()
    oc = _load_json(home / ".config" / "opencode" / "opencode.json")
    for pid, cfg in (oc.get("provider") or {}).items():
        opts = (cfg or {}).get("options") or {}
        base = opts.get("baseURL", "")
        if not base or base.rstrip("/") in seen:
            continue
        seen.add(base.rstrip("/"))
        providers.setdefault(
            pid, OpenAICompatProvider(pid, base, opts.get("apiKey", ""), (cfg or {}).get("name", pid) or pid)
        )

    pi_models = _load_json(home / ".pi" / "agent" / "models.json")
    pi_auth = _load_json(home / ".pi" / "agent" / "auth.json")
    local_enabled = os.environ.get("CLAP_OLLAMA_LOCAL", "") == "1"
    for pid, cfg in (pi_models.get("providers") or {}).items():
        base = (cfg or {}).get("baseUrl", "")
        if not base or base.rstrip("/") in seen:
            continue
        if pid == "ollama-local" and not local_enabled:
            continue  # W2: the local daemon needs an explicit opt-in
        seen.add(base.rstrip("/"))
        key = (pi_auth.get(pid) or {}).get("key", "") or (cfg or {}).get("apiKey", "")
        providers.setdefault(pid, OpenAICompatProvider(pid, base, key, pid))

    if "ollama-cloud" not in providers:
        providers["ollama-cloud"] = OpenAICompatProvider(
            "ollama-cloud", "https://ollama.com/v1", (pi_auth.get("ollama-cloud") or {}).get("key", ""),
            "ollama cloud",
        )

    if local_enabled:
        providers.setdefault(
            "ollama-local",
            OpenAICompatProvider("ollama-local", "http://127.0.0.1:11434/v1", "", "ollama local daemon"),
        )

    providers.pop("ollama", None)
    return providers


def provider_from_names(registry: dict, provider_id: str | None) -> Provider:
    if provider_id and provider_id in registry:
        return registry[provider_id]
    if provider_id:
        raise ProviderError(f"provider {provider_id!r} not configured (have: {sorted(registry)})")
    return registry.get("ollama-local") or next(iter(registry.values()))


def known_models(provider_id: str) -> list[str]:
    """Model ids known without touching the network (settings + inventory)."""
    from clap.core import settings_store

    saved = settings_store.load().get("providers", {}).get(provider_id, {}).get("models", [])
    inventory = load_probe_list().get("models", {}).get(provider_id, [])
    return list(dict.fromkeys(list(saved) + list(inventory)))


def default_model_for(registry: dict, provider_id: str) -> str:
    """First known model, without a blocking network probe.

    Live `/models` is only consulted when neither the saved settings nor the
    vendored inventory knows the provider — page loads must stay instant.
    """
    models = known_models(provider_id)
    if models:
        return models[0]
    prov = registry.get(provider_id)
    if prov is not None:
        fallback = getattr(prov, "_fallback_models", []) or []
        if fallback:
            return fallback[0]
        live = prov.list_models(timeout=2.0)
        if live:
            return live[0]
    return ""


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
