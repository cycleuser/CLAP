"""Local settings store (providers + default model).

Location: `~/.config/clap/settings.json`, created with mode 0600. It holds
provider endpoints and API keys entered through the web settings panel.

Security rules
--------------
* The file is written atomically and chmod 0600.
* Keys are never returned by the API: callers get `mask_secret()` output.
* Nothing here is logged. Errors are scrubbed before reaching HTTP responses.
* Environment variables still work; a provider configured here takes
  precedence over the same provider discovered from env/config files.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

_LOCK = threading.RLock()

DEFAULT_PROVIDER = "deepseek"
DEFAULT_MODEL = "deepseek-flash"


def settings_path() -> Path:
    override = os.environ.get("CLAP_SETTINGS_PATH", "")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".config" / "clap" / "settings.json"


def _default_payload() -> dict:
    return {"providers": {}, "default": {"provider": DEFAULT_PROVIDER, "model": DEFAULT_MODEL}}


def load() -> dict:
    """Read settings; always returns a well-formed dict."""
    path = settings_path()
    data: dict = {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            data = raw
    except Exception:
        data = {}
    out = _default_payload()
    providers = data.get("providers")
    if isinstance(providers, dict):
        out["providers"] = {
            str(k): {
                "base": str(v.get("base") or "").strip(),
                "apiKey": str(v.get("apiKey") or "").strip(),
                "name": str(v.get("name") or "").strip(),
                "models": _normalise_models(v.get("models")),
            }
            for k, v in providers.items()
            if isinstance(v, dict)
        }
    default = data.get("default")
    if isinstance(default, dict) and default.get("provider"):
        out["default"] = {
            "provider": str(default.get("provider") or ""),
            "model": str(default.get("model") or ""),
        }
    return out


def _normalise_models(value) -> list[str]:
    """Accept ['a'], [{'id': 'a'}], or {'a': {...}} and return model ids."""
    out: list[str] = []
    if isinstance(value, dict):
        candidates = list(value.keys())
    elif isinstance(value, list):
        candidates = []
        for item in value:
            if isinstance(item, str):
                candidates.append(item)
            elif isinstance(item, dict):
                mid = item.get("id") or item.get("modelID") or item.get("name")
                if mid:
                    candidates.append(str(mid))
    else:
        candidates = []
    for c in candidates:
        c = str(c).strip()
        if c and c not in out:
            out.append(c)
    return out


def save(data: dict) -> dict:
    path = settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _default_payload()
    if isinstance(data.get("providers"), dict):
        payload["providers"] = data["providers"]
    if isinstance(data.get("default"), dict) and data["default"].get("provider"):
        payload["default"] = {
            "provider": str(data["default"].get("provider")),
            "model": str(data["default"].get("model") or ""),
        }
    tmp = path.with_suffix(".tmp")
    with _LOCK:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    return payload


def set_provider(provider_id: str, base: str = "", api_key: str | None = None,
                 name: str = "", keep_key: bool = True, models: list[str] | None = None) -> dict:
    """Create/update a provider. `api_key=None` keeps the stored key."""
    data = load()
    pid = str(provider_id or "").strip()
    if not pid:
        raise ValueError("provider id is required")
    row = dict(data["providers"].get(pid) or {})
    if base:
        row["base"] = base.strip()
    if name:
        row["name"] = name.strip()
    if api_key is not None:
        row["apiKey"] = api_key.strip()
    if models is not None:
        row["models"] = _normalise_models(models)
    if not row.get("base"):
        raise ValueError("base URL is required")
    if not keep_key and api_key is None:
        row.pop("apiKey", None)
    data["providers"][pid] = row
    return save(data)


def remove_provider(provider_id: str) -> dict:
    data = load()
    data["providers"].pop(str(provider_id or ""), None)
    return save(data)


def set_default(provider: str, model: str) -> dict:
    data = load()
    if not provider:
        raise ValueError("provider is required")
    data["default"] = {"provider": provider, "model": model or ""}
    return save(data)


def import_json(text: str) -> dict:
    """Import provider configuration from a JSON document.

    Accepted shapes (detected automatically):
      * opencode: {"provider": {"id": {"options": {"baseURL","apiKey"},
                    "models": {...}}}}
      * pi:       {"providers": {"id": {"baseUrl": "...", "models": [...]}},
                   "auth": {"id": {"key": "..."}}}
      * CLAP:     {"providers": {"id": {"base","apiKey","models"}}}
      * list:     [{"id","base","apiKey","models"}]

    Keys are written to the local settings file only; the return value never
    contains a key value.
    """
    try:
        doc = json.loads(text)
    except Exception as exc:
        raise ValueError(f"JSON 解析失败：{exc}") from exc
    if not isinstance(doc, (dict, list)):
        raise ValueError("JSON 顶层必须是对象或数组")

    data = load()
    imported: list[str] = []
    skipped: list[str] = []
    auth = doc.get("auth") if isinstance(doc, dict) else None

    def merge(pid: str, base: str, key: str, models) -> None:
        pid = str(pid or "").strip()
        base = str(base or "").strip()
        if not pid or not base:
            skipped.append(pid or "?")
            return
        row = dict(data["providers"].get(pid) or {})
        row["base"] = base
        if key:
            row["apiKey"] = str(key)
        model_ids = _normalise_models(models)
        if model_ids:
            row["models"] = model_ids
        data["providers"][pid] = row
        imported.append(pid)

    if isinstance(doc, list):
        for item in doc:
            if not isinstance(item, dict):
                continue
            merge(
                item.get("id") or item.get("providerID") or item.get("name"),
                item.get("base") or item.get("baseURL") or item.get("baseUrl"),
                item.get("apiKey") or item.get("key"),
                item.get("models"),
            )
    else:
        providers = doc.get("provider") if isinstance(doc.get("provider"), dict) else doc.get("providers")
        if isinstance(providers, dict):
            for pid, cfg in providers.items():
                if not isinstance(cfg, dict):
                    continue
                opts = cfg.get("options") if isinstance(cfg.get("options"), dict) else {}
                key = (
                    opts.get("apiKey")
                    or cfg.get("apiKey")
                    or cfg.get("key")
                    or ((auth or {}).get(pid) or {}).get("key", "")
                )
                merge(
                    pid,
                    opts.get("baseURL") or cfg.get("baseURL") or cfg.get("baseUrl") or cfg.get("base"),
                    key,
                    cfg.get("models"),
                )
        elif isinstance(doc.get("providers"), list):
            for item in doc["providers"]:
                if isinstance(item, dict):
                    merge(
                        item.get("id") or item.get("name"),
                        item.get("base") or item.get("baseURL") or item.get("baseUrl"),
                        item.get("apiKey") or item.get("key"),
                        item.get("models"),
                    )
    if imported:
        save(data)
    return {"imported": imported, "skipped": skipped, "providers": load()["providers"]}


def masked(value: str) -> str:
    if not value:
        return ""
    tail = value[-4:] if len(value) >= 8 else "****"
    return f"***len{len(value)}…{tail}"
