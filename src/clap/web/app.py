"""CLAP web workspace: chat, branches, annotations, ZIM knowledge, grounded mode.

Pages
  /            chat workspace (default landing)
  /kb          ZIM external knowledge base (list / search / read)
  /timeline    timeline view rendered by clap.core.timeline
  /api-docs    endpoint reference

Chat endpoints
  GET  /api/models          provider-grouped models (env assembly, masked keys)
  POST /api/chat            SSE chat through the provider layer; supports
                            grounded mode (`grounded`, `zims`), branch
                            selection, renewal summary and cancellation
  POST /api/chat/stop       cancel the running stream

Session endpoints
  GET  /api/session/get     current session (branch via ?branch=)
  POST /api/session/new | save | load | fork
  POST /api/session/export  md / html / json (inline content, optional path)
  POST /api/session/share   build a .clapshare package
  POST /api/session/adopt   import a .clapshare package
  POST /api/session/renew   deterministic renewal digest

ZIM endpoints
  GET  /api/zim/list
  POST /api/zim/search
  GET  /api/zim/article?zim=&path=
  POST /api/zim/suggest

Security: credentials are read in memory only (env / opencode.json /
pi auth.json); they are never logged, persisted or echoed. Every response
carries masked forms (`***lenNN…tail`) at most; error text is scrubbed.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from clap.core import export as EXP
from clap.core import grounded as GR
from clap.core import markdown_render as MD
from clap.core import providers as PR
from clap.core import renew as RN
from clap.core import session as SM
from clap.core import sharing as SH
from clap.core import timeline as TL
from clap.core import zim_kb
from clap.core.knowledge_base import KnowledgeBase

app = Flask(__name__)
# one design system for every page: the stylesheet is inlined into <style>
# so a page never issues a second request (and never reaches a CDN).
app.jinja_loader.searchpath.append(str(Path(__file__).parent / "static"))
kb = KnowledgeBase()

DEFAULT_PROVIDER = "deepseek"
GROUNDED_TOP_K = 6

# Preferred default model per provider (used when the provider is configured
# but no explicit choice was saved in settings).
PREFERRED_MODELS = {"deepseek": "deepseek-flash"}


def _text(lang: str, zh: str, en: str) -> str:
    """Pick a message language (kept for simple two-way messages)."""
    return en if str(lang or "").lower().startswith("en") else zh


# Server-side user-facing messages, all ten interface languages. Missing
# entries fall back to English, then Chinese.
UI_TEXT = {
    "empty_message": {
        "zh": "消息为空", "en": "Message is empty", "ja": "メッセージが空です",
        "fr": "Le message est vide", "ru": "Сообщение пустое", "de": "Die Nachricht ist leer",
        "es": "El mensaje está vacío", "pt": "A mensagem está vazia", "it": "Il messaggio è vuoto",
        "ko": "메시지가 비어 있습니다",
    },
    "unknown_mode": {
        "zh": "未知的对话模式", "en": "Unknown chat mode", "ja": "不明なチャットモード",
        "fr": "Mode de discussion inconnu", "ru": "Неизвестный режим чата",
        "de": "Unbekannter Chat-Modus", "es": "Modo de chat desconocido",
        "pt": "Modo de conversa desconhecido", "it": "Modalità di chat sconosciuta",
        "ko": "알 수 없는 채팅 모드",
    },
    "bad_message_id": {
        "zh": "message_id 必须指向这条分支上的提问消息",
        "en": "message_id must point to a user message on this branch",
        "ja": "message_id はこのブランチ上のユーザーメッセージを指す必要があります",
        "fr": "message_id doit désigner un message utilisateur de cette branche",
        "ru": "message_id должен указывать на сообщение пользователя в этой ветке",
        "de": "message_id muss auf eine Benutzernachricht in diesem Zweig zeigen",
        "es": "message_id debe apuntar a un mensaje del usuario en esta rama",
        "pt": "message_id deve apontar para uma mensagem do utilizador neste ramo",
        "it": "message_id deve indicare un messaggio utente in questo ramo",
        "ko": "message_id는 이 분기의 사용자 메시지를 가리켜야 합니다",
    },
    "provider_missing": {
        "zh": "服务商「{pid}」还没有配置（环境变量或配置文件里找不到它的凭证）",
        "en": 'Provider "{pid}" is not configured (no credentials in the environment or config files)',
        "ja": "プロバイダー「{pid}」が未設定です（環境変数や設定ファイルに認証情報がありません）",
        "fr": "Le fournisseur « {pid} » n'est pas configuré (aucun identifiant dans l'environnement ou les fichiers de configuration)",
        "ru": "Провайдер «{pid}» не настроен (нет учётных данных в окружении или файлах конфигурации)",
        "de": "Anbieter „{pid}“ ist nicht konfiguriert (keine Zugangsdaten in Umgebung oder Konfigurationsdateien)",
        "es": "El proveedor «{pid}» no está configurado (no hay credenciales en el entorno ni en los archivos de configuración)",
        "pt": "O fornecedor «{pid}» não está configurado (sem credenciais no ambiente ou nos ficheiros de configuração)",
        "it": "Il fornitore «{pid}» non è configurato (nessuna credenziale nell'ambiente o nei file di configurazione)",
        "ko": '공급자 "{pid}"이(가) 구성되지 않았습니다 (환경 변수나 설정 파일에 자격 증명 없음)',
    },
    "no_model": {
        "zh": "服务商「{pid}」没有可用的模型",
        "en": 'Provider "{pid}" has no available model',
        "ja": "プロバイダー「{pid}」に利用可能なモデルがありません",
        "fr": "Le fournisseur « {pid} » n'a aucun modèle disponible",
        "ru": "У провайдера «{pid}» нет доступных моделей",
        "de": "Anbieter „{pid}“ hat kein verfügbares Modell",
        "es": "El proveedor «{pid}» no tiene modelos disponibles",
        "pt": "O fornecedor «{pid}» não tem modelos disponíveis",
        "it": "Il fornitore «{pid}» non ha modelli disponibili",
        "ko": '공급자 "{pid}"에 사용 가능한 모델이 없습니다',
    },
    "abstain": {
        "zh": "来源未提及（检索没有找到可用的材料，换个关键词或指定资料库）。",
        "en": "The material does not mention this (the search found nothing; try other keywords or choose a specific library).",
        "ja": "資料には言及されていません（検索で材料が見つかりませんでした。別のキーワードかライブラリを指定してください）。",
        "fr": "Le document n'en parle pas (la recherche n'a rien trouvé ; essayez d'autres mots-clés ou une autre bibliothèque).",
        "ru": "В материале это не упоминается (поиск ничего не нашёл; попробуйте другие ключевые слова или библиотеку).",
        "de": "Das Material erwähnt das nicht (die Suche fand nichts; andere Suchbegriffe oder eine bestimmte Bibliothek wählen).",
        "es": "El material no lo menciona (la búsqueda no encontró nada; prueba otras palabras o una biblioteca concreta).",
        "pt": "O material não menciona isto (a pesquisa não encontrou nada; tente outras palavras ou uma biblioteca específica).",
        "it": "Il materiale non lo menziona (la ricerca non ha trovato nulla; prova altre parole o una libreria specifica).",
        "ko": "자료에 언급되지 않았습니다 (검색에서 자료를 찾지 못했습니다. 다른 키워드나 특정 라이브러리를 지정하세요).",
    },
}


def _msg(lang: str, key: str, **vars) -> str:
    """Localized server message (10 languages, English/Chinese fallback)."""
    tag = (str(lang or "").split("-")[0].lower()) or "zh"
    table = UI_TEXT.get(key, {})
    text = table.get(tag) or table.get("en") or table.get("zh") or key
    return text.format(**vars) if vars else text

# --------------------------------------------------------------------------
# provider registry (cached: availability probes are network calls)
# --------------------------------------------------------------------------

_registry_lock = threading.Lock()
_registry_cache: dict = {"value": None, "ts": 0.0}
_CACHE_TTL_S = 60.0


def get_registry() -> dict[str, PR.Provider]:
    with _registry_lock:
        now = time.time()
        if _registry_cache["value"] is None or now - _registry_cache["ts"] > _CACHE_TTL_S:
            _registry_cache["value"] = PR.discover_providers()
            _registry_cache["ts"] = now
        return _registry_cache["value"]


def reset_registry_cache() -> None:
    with _registry_lock:
        _registry_cache["value"] = None
        _registry_cache["ts"] = 0.0


def default_selection(registry: dict) -> dict:
    """Default provider/model: saved settings → deepseek-flash → live first."""
    from clap.core import settings_store

    pref = settings_store.load().get("default") or {}
    pid = pref.get("provider") if pref.get("provider") in registry else ""
    if not pid:
        pid = DEFAULT_PROVIDER if DEFAULT_PROVIDER in registry else ""
    if not pid:
        preferred = [p for p in sorted(registry) if p in ("deepseek", "ollama-cloud", "huggingface")]
        pid = preferred[0] if preferred else (sorted(registry)[0] if registry else "")
    if not pid:
        return {"provider": "", "model": ""}

    prov = registry.get(pid)
    live = prov.list_models() if prov is not None else []
    wanted = pref.get("model") if pref.get("provider") == pid else ""
    model = ""
    if wanted and (not live or wanted in live):
        model = wanted
    elif pid in PREFERRED_MODELS and (not live or PREFERRED_MODELS[pid] in live):
        model = PREFERRED_MODELS[pid]
    else:
        model = PR.default_model_for(registry, pid)
    return {"provider": pid, "model": model}


# --------------------------------------------------------------------------
# session state (one shared in-memory session drives the web workspace)
# --------------------------------------------------------------------------

_session_lock = threading.RLock()
_current: dict | None = None


def current_session() -> dict:
    global _current
    with _session_lock:
        if _current is None:
            from clap.utils.config import load_config

            cfg = load_config()
            sel = default_selection(get_registry())
            _current = SM.new_session(
                chat_model=sel["model"] or cfg.chat_model,
                provider=sel["provider"] or "",
                purpose="答疑",
            )
            SM.ensure_main_branch(_current)
        return _current


def set_session(s: dict | None) -> None:
    global _current
    with _session_lock:
        _current = s


def _main_branch_id(s: dict) -> str:
    return (SM.main_branch(s) or {}).get("id") or "b1"


def _branch_id(s: dict, requested: str | None) -> str:
    if requested and SM.get_branch(s, requested):
        return requested
    return _main_branch_id(s)


def _tools_branch_id(s: dict) -> str:
    """Where machine-issued turns (tests/probes) go: a side branch, never the
    conversation's main line, so live checks don't pollute the dialog."""
    for b in s.get("branches") or []:
        if b.get("name") == "tools":
            return b["id"]
    parent_path = SM.branch_path(s, _main_branch_id(s))
    parent = parent_path[-1]["id"] if parent_path else None
    b = SM.fork_branch(s, parent, name="tools", note="web API checks", author="system")
    return b["id"]


# --------------------------------------------------------------------------
# chat streaming with cancellation
# --------------------------------------------------------------------------

_active_lock = threading.Lock()
_active_flags: dict[str, dict] = {}


def _new_stream_id() -> str:
    with _active_lock:
        return f"s{int(time.time() * 1000)}-{len(_active_flags) + 1}"


def _scrub(text: str, registry: dict) -> str:
    """Never echo key material into any response body."""
    out = str(text)
    for p in registry.values():
        key = getattr(p, "_api_key", "") or ""
        if key and key in out:
            out = out.replace(key, "***")
    return out


def _convo(s: dict, branch_id: str, upto_user_turns: int | None = None) -> list[dict]:
    """API messages = full branch context (spine + own lineage)."""
    msgs: list[dict] = []
    users = 0
    for m in SM.branch_full_path(s, branch_id):
        if m.get("role") not in ("user", "assistant", "system"):
            continue
        if m.get("role") != "system" and not (m.get("content") or "").strip():
            continue
        msgs.append({"role": m["role"], "content": m["content"]})
        if m["role"] == "user":
            users += 1
            if upto_user_turns is not None and users >= upto_user_turns:
                break
    return msgs


@app.route("/api/models")
def api_models():
    """Provider-grouped models + env-assembly result (masked, no key values)."""
    registry = get_registry()
    probe = PR.load_probe_list()
    assembler = PR.EnvAssembler(probe)
    env_registry = assembler.assemble()
    env_rows = assembler.sources

    groups = []
    for pid in sorted(probe["providers"]):
        prov = registry.get(pid)
        row = env_rows.get(pid) or {
            "provider": pid,
            "base": probe["providers"].get(pid, ""),
            "env": "",
            "masked": "<none>",
            "models": list(probe["models"].get(pid, [])),
            "configured": False,
            "unsupported": "",
        }
        if prov is None:
            groups.append(
                {
                    "provider": pid,
                    "base": row["base"],
                    "configured": row["configured"],
                    "env": row["env"],
                    "masked": row["masked"],
                    "available": False,
                    "unsupported": row["unsupported"],
                    "models": row["models"][:80],
                }
            )
            continue
        live = prov.list_models()
        from_env = pid in env_registry
        groups.append(
            {
                "provider": pid,
                "base": prov.base,
                "configured": True,
                "env": row["env"] if from_env else "",
                "also_env": row.get("also_env", []),
                "source": "env" if from_env else "config",
                "masked": prov.masked_key(),
                "available": prov.is_available(),
                "unsupported": "",
                "models": (live or row["models"])[:80],
            }
        )
    for pid in sorted(set(registry) - set(probe["providers"])):
        prov = registry[pid]
        groups.append(
            {
                "provider": pid,
                "base": prov.base,
                "configured": True,
                "env": "",
                "masked": prov.masked_key(),
                "available": prov.is_available(),
                "unsupported": "",
                "models": prov.list_models()[:80],
            }
        )
    return jsonify(
        {
            "success": True,
            "groups": groups,
            "default": default_selection(registry),
            "env_assembly": [
                {"provider": r["provider"], "env": r["env"], "masked": r["masked"], "configured": r["configured"]}
                for r in env_rows.values()
                if r["configured"]
            ],
        }
    )


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Streaming chat through the provider layer (SSE).

    Body: {message, provider?, model?, branch?, lane?, grounded?, zims?,
           top_k?, renew_summary?, mode?, message_id?, lang?}

    mode:
      "send"        normal new question (default)
      "regenerate"  re-answer the user message `message_id`: its replies are
                    dropped and generated again on the same branch
      "edit"        replace the user message `message_id` with `message`,
                    drop its replies, then answer
    """
    data = request.get_json(silent=True) or {}
    lang = (data.get("lang") or "").strip()
    mode = (data.get("mode") or "send").strip()
    msg = (data.get("message") or "").strip()
    if mode not in ("send", "regenerate", "edit"):
        return jsonify({"success": False, "error": _msg(lang, "unknown_mode")}), 400
    if mode != "regenerate" and not msg:
        return jsonify({"success": False, "error": _msg(lang, "empty_message")}), 400

    registry = get_registry()
    pid = data.get("provider") or default_selection(registry)["provider"]
    if pid not in registry:
        return (
            jsonify(
                {
                    "success": False,
                    "error": _msg(lang, "provider_missing", pid=pid),
                }
            ),
            400,
        )

    model = (data.get("model") or "").strip() or PR.default_model_for(registry, pid)
    if not model:
        return (
            jsonify(
                {
                    "success": False,
                    "error": _msg(lang, "no_model", pid=pid),
                }
            ),
            400,
        )

    s = current_session()
    branch = _tools_branch_id(s) if data.get("lane") == "tools" else _branch_id(s, data.get("branch"))

    # ---- regenerate / edit: redo the last answer in place ------------------
    if mode in ("regenerate", "edit"):
        target_id = (data.get("message_id") or "").strip()
        target = SM.message(s, target_id) if target_id else None
        if target is None or target.get("role") != "user":
            return (
                jsonify(
                    {
                        "success": False,
                        "error": _msg(lang, "bad_message_id"),
                    }
                ),
                400,
            )
        with _session_lock:
            try:
                removed = SM.delete_replies_after(s, branch, target_id)
            except SM.SessionError as e:
                return jsonify({"success": False, "error": _scrub(str(e), registry)}), 400
            if mode == "edit":
                old_text = target.get("content") or ""
                target["content"] = msg
                target["edited_from"] = old_text[:500]
                target["ts"] = SM.now_ms()
            else:
                msg = target.get("content") or ""
            SM._add_history(
                s,
                "edit" if mode == "edit" else "regenerate",
                {
                    "branch": branch,
                    "message": target_id,
                    "removed": removed,
                    "text": msg[:200],
                },
            )

    # ---- grounded mode: retrieve BEFORE answering -------------------------
    grounded = bool(data.get("grounded"))
    sources: list[GR.Source] = []
    system_prompt = ""
    if grounded:
        query = msg
        try:
            hits = zim_kb.library().search(query, data.get("zims") or None, top_k=int(data.get("top_k") or GROUNDED_TOP_K))
            sources, _ = GR.build_context(hits)
        except Exception:
            hits, sources = [], []
        if sources:
            system_prompt = GR.render_system_prompt(sources)

    renew_summary = (data.get("renew_summary") or "").strip()

    with _session_lock:
        if mode == "send":
            path = SM.branch_path(s, branch)
            parent = path[-1]["id"] if path else None
            SM.append_message(
                s, "user", msg, parent=parent, branch_id=branch, model=model, provider=pid,
                grounded=grounded, zims=list(data.get("zims") or []) if grounded else None,
            )
        convo = _convo(s, branch)

    if renew_summary:
        convo = [RN.system_message_for(renew_summary)] + convo
    if system_prompt:
        convo = [{"role": "system", "content": system_prompt}] + convo

    stream_id = _new_stream_id()
    with _active_lock:
        _active_flags[stream_id] = {"cancelled": False}

    def sse():
        parts: list[str] = []
        meta = {
            "provider": pid,
            "model": model,
            "branch": branch,
            "stream_id": stream_id,
            "latency_ms": None,
            "tokens": {"in": None, "out": None},
            "error": None,
            "grounded": grounded,
            "sources": [src.as_dict() for src in sources],
            "audit": None,
            "renewed": bool(renew_summary),
        }
        t0 = time.time()
        yield f"data: {json.dumps({'start': True, 'stream_id': stream_id, 'meta': meta}, ensure_ascii=False)}\n\n"
        if grounded and sources:
            yield f"data: {json.dumps({'sources': meta['sources']}, ensure_ascii=False)}\n\n"

        abstained = grounded and not sources and not data.get("allow_ungrounded")
        if abstained:
            parts.append(
                _msg(lang, "abstain")
            )
            yield f"data: {json.dumps({'delta': parts[-1]}, ensure_ascii=False)}\n\n"
        else:
            try:
                for delta in registry[pid].chat(messages=convo, model=model, stream=True):
                    if _active_flags.get(stream_id, {}).get("cancelled"):
                        meta["error"] = "cancelled"
                        break
                    parts.append(delta)
                    yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
                last = getattr(registry[pid], "last_meta", {}) or {}
                meta["tokens"] = last.get("tokens", meta["tokens"])
            except Exception as e:  # never leak key material
                meta["error"] = _scrub(f"{type(e).__name__}: {e}"[:300], registry)

        text = "".join(parts)
        if meta["error"] is None and not text:
            meta["error"] = f"{pid}: 空响应"

        if grounded:
            meta["audit"] = GR.audit(text, sources) if text else None

        meta["latency_ms"] = int((time.time() - t0) * 1000)

        with _session_lock:
            path = SM.branch_path(s, branch)
            parent = path[-1]["id"] if path else None
            SM.append_message(
                s,
                "assistant",
                text,
                parent=parent,
                branch_id=branch,
                model=meta["model"],
                provider=meta["provider"],
                latency_ms=meta["latency_ms"],
                tokens=meta["tokens"],
                error=meta["error"] or "",
                sources=meta["sources"] or None,
                audit=meta["audit"],
                grounded=grounded or None,
            )
        with _active_lock:
            _active_flags.pop(stream_id, None)
        yield "data: " + json.dumps({"done": True, "meta": meta}, ensure_ascii=False) + "\n\n"

    return Response(
        sse(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/api/chat/stop", methods=["POST"])
def api_chat_stop():
    """Cancel the running stream (the UI's 暂停 button)."""
    body = request.get_json(silent=True) or {}
    sid = body.get("stream_id", "")
    stopped: list[str] = []
    with _active_lock:
        if sid and sid in _active_flags:
            _active_flags[sid]["cancelled"] = True
            stopped = [sid]
        elif not sid:
            stopped = [k for k, v in _active_flags.items() if not v.get("cancelled")]
            for k in stopped:
                _active_flags[k]["cancelled"] = True
    return jsonify({"success": bool(stopped), "stopped": stopped})


# --------------------------------------------------------------------------
# settings: provider credentials and the default model
# --------------------------------------------------------------------------


@app.route("/api/settings", methods=["GET"])
def api_settings_get():
    """Provider configuration + default model (keys masked, never returned)."""
    from clap.core import settings_store as SS

    registry = get_registry()
    saved = SS.load()
    probe = PR.load_probe_list()
    rows = []
    for pid in sorted(set(saved["providers"]) | set(registry) | set(probe["providers"])):
        prov = registry.get(pid)
        saved_row = saved["providers"].get(pid) or {}
        models = (prov.list_models() if prov else []) or saved_row.get("models") or probe["models"].get(pid, [])
        rows.append(
            {
                "id": pid,
                "base": prov.base if prov else (saved_row.get("base") or probe["providers"].get(pid, "")),
                "masked": prov.masked_key() if prov else SS.masked(saved_row.get("apiKey", "")),
                "configured": prov is not None,
                "source": "settings" if saved_row else ("discovered" if prov else "inventory"),
                "available": prov.is_available() if prov else False,
                "models": models,
                "models_manual": list(saved_row.get("models") or []),
            }
        )
    return jsonify({"success": True, "default": default_selection(registry), "providers": rows})


@app.route("/api/settings/provider", methods=["POST"])
def api_settings_provider():
    """Save a provider endpoint/key/models. Missing key → keep the stored one."""
    from clap.core import settings_store as SS

    body = request.get_json(silent=True) or {}
    pid = (body.get("id") or "").strip()
    base = (body.get("base") or "").strip()
    key = body.get("key")
    if key is not None and not str(key).strip():
        key = None  # empty field means "do not change"
    models = body.get("models")
    if isinstance(models, str):
        models = [m.strip() for m in models.replace("，", ",").split(",") if m.strip()]
    elif not isinstance(models, list):
        models = None
    try:
        SS.set_provider(pid, base=base, api_key=key, name=(body.get("name") or "").strip(), models=models)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    reset_registry_cache()
    row = SS.load()["providers"].get(pid, {})
    return jsonify(
        {
            "success": True,
            "provider": {
                "id": pid,
                "base": row.get("base", ""),
                "masked": SS.masked(row.get("apiKey", "")),
                "models": row.get("models", []),
            },
        }
    )


@app.route("/api/settings/import", methods=["POST"])
def api_settings_import():
    """Import providers from a pasted/uploaded JSON document (keys stay local)."""
    from clap.core import settings_store as SS

    body = request.get_json(silent=True) or {}
    text = body.get("json") or ""
    if not text.strip():
        return jsonify({"success": False, "error": "JSON 为空"}), 400
    try:
        result = SS.import_json(text)
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    reset_registry_cache()
    return jsonify(
        {
            "success": True,
            "imported": result["imported"],
            "skipped": result["skipped"],
            "count": len(result["imported"]),
        }
    )


@app.route("/api/settings/provider/<provider_id>", methods=["DELETE"])
def api_settings_provider_delete(provider_id):
    from clap.core import settings_store as SS

    SS.remove_provider(provider_id)
    reset_registry_cache()
    return jsonify({"success": True, "id": provider_id})


@app.route("/api/settings/default", methods=["POST"])
def api_settings_default():
    from clap.core import settings_store as SS

    body = request.get_json(silent=True) or {}
    try:
        SS.set_default((body.get("provider") or "").strip(), (body.get("model") or "").strip())
    except ValueError as e:
        return jsonify({"success": False, "error": str(e)}), 400
    reset_registry_cache()
    return jsonify({"success": True, "default": SS.load().get("default")})


@app.route("/api/settings/test", methods=["POST"])
def api_settings_test():
    body = request.get_json(silent=True) or {}
    pid = (body.get("id") or "").strip()
    prov = get_registry().get(pid)
    if prov is None:
        return jsonify({"success": False, "error": _msg(body.get("lang", ""), "provider_missing", pid=pid)}), 400
    return jsonify(
        {
            "success": True,
            "id": pid,
            "available": prov.is_available(),
            "models": prov.list_models()[:30],
            "masked": prov.masked_key(),
            "base": prov.base,
        }
    )


# --------------------------------------------------------------------------
# session endpoints
# --------------------------------------------------------------------------


def _message_view(m: dict) -> dict:
    """Session message + rendered HTML (math/code/citations) for the UI."""
    view = dict(m)
    content = m.get("content") or ""
    try:
        view["html"] = (
            MD.render_markdown(content, citations=(m.get("role") == "assistant"))
            if m.get("role") == "assistant"
            else MD.render_plain(content)
        )
    except Exception:
        view["html"] = MD.render_plain(content)
    return view


@app.route("/api/session/get")
def api_session_get():
    s = current_session()
    b = _branch_id(s, request.args.get("branch"))
    return jsonify(
        {
            "success": True,
            "session_id": s["manifest"]["session_id"],
            "purpose": s["manifest"].get("purpose", ""),
            "branch_id": b,
            "messages": [_message_view(m) for m in SM.branch_full_path(s, b)],
            "branches": [
                {
                    "id": x.get("id"),
                    "name": x.get("name"),
                    "model": x.get("model"),
                    "fork_at": x.get("fork_at"),
                    "turns": len([m for m in SM.branch_path(s, x.get("id")) if m.get("role") == "user"]),
                }
                for x in s.get("branches", [])
            ],
            "annotations": s.get("annotations", []),
            "history": [
                h
                for h in s["manifest"].get("history", [])
                if h.get("event") in ("privacy_note", "renewal", "edit", "regenerate")
            ][-10:],
        }
    )


@app.route("/api/session/new", methods=["POST"])
def api_session_new():
    set_session(None)
    s = current_session()
    return jsonify({"success": True, "session_id": s["manifest"]["session_id"]})


@app.route("/api/session/save", methods=["POST"])
def api_session_save():
    path = (request.get_json(silent=True) or {}).get("path", "")
    if not path:
        return jsonify({"success": False, "error": "path required"}), 400
    s = current_session()
    try:
        SM.save_session(s, path)
        return jsonify({"success": True, "path": path, "session_id": s["manifest"]["session_id"]})
    except Exception as e:
        return jsonify({"success": False, "error": _scrub(str(e), get_registry())}), 500


@app.route("/api/session/load", methods=["POST"])
def api_session_load():
    path = (request.get_json(silent=True) or {}).get("path", "")
    if not path:
        return jsonify({"success": False, "error": "path required"}), 400
    try:
        s = SM.load_session(path)
    except Exception as e:
        return jsonify({"success": False, "error": _scrub(f"无法加载：{e}", get_registry())}), 400
    set_session(s)
    return jsonify({"success": True, "path": path, "session_id": s["manifest"]["session_id"]})


@app.route("/api/session/fork", methods=["POST"])
def api_session_fork():
    body = request.get_json(silent=True) or {}
    at = body.get("at", "")
    if not at:
        return jsonify({"success": False, "error": "at（消息 id）必填"}), 400
    s = current_session()
    try:
        b = SM.fork_branch(
            s,
            at,
            name=body.get("name", ""),
            model=body.get("model") or None,
            note=body.get("note", ""),
            author=body.get("author", "teacher") or "teacher",
        )
    except Exception as e:
        return jsonify({"success": False, "error": _scrub(str(e), get_registry())}), 400
    return jsonify({"success": True, "branch": b})


@app.route("/api/session/export", methods=["POST"])
def api_session_export():
    body = request.get_json(silent=True) or {}
    fmt = (body.get("format") or "md").lower()
    branch = body.get("branch") or None
    s = current_session()
    try:
        content, written = EXP.export(s, fmt=fmt, branch_id=branch, path=body.get("path") or None)
    except Exception as e:
        return jsonify({"success": False, "error": _scrub(str(e), get_registry())}), 400
    return jsonify(
        {
            "success": True,
            "format": fmt,
            "path": written,
            "chars": len(content),
            "content": content if body.get("inline", True) else "",
        }
    )


@app.route("/api/session/share", methods=["POST"])
def api_session_share():
    """Pack the current session into a `.clapshare` (no keys, no local paths)."""
    body = request.get_json(silent=True) or {}
    s = current_session()
    tmp = Path(body.get("tmp") or "/tmp") / f"clap-share-{s['manifest']['session_id']}.clap"
    try:
        SM.save_session(s, tmp)
        pkg = SH.pack(tmp, out_dir=body.get("out_dir") or None)
    except Exception as e:
        return jsonify({"success": False, "error": _scrub(str(e), get_registry())}), 400
    finally:
        try:
            tmp.unlink()
        except Exception:
            pass
    return jsonify({"success": True, "package": str(pkg), "size": pkg.stat().st_size})


@app.route("/api/session/adopt", methods=["POST"])
def api_session_adopt():
    body = request.get_json(silent=True) or {}
    package = body.get("package", "")
    if not package:
        return jsonify({"success": False, "error": "package 路径必填"}), 400
    try:
        rec = SH.adopt(package, policy=body.get("policy") or "rename")
    except Exception as e:
        return jsonify({"success": False, "error": _scrub(str(e), get_registry())}), 400
    return jsonify({"success": True, "record": rec})


@app.route("/api/session/renew", methods=["POST"])
def api_session_renew():
    body = request.get_json(silent=True) or {}
    s = current_session()
    branch = _branch_id(s, body.get("branch"))
    try:
        digest = RN.renewal(s, branch)
    except Exception as e:
        return jsonify({"success": False, "error": _scrub(str(e), get_registry())}), 400
    return jsonify({"success": True, **digest})


@app.route("/api/annotate", methods=["POST"])
def api_annotate():
    """Structured annotation on a message of the current session (F5)."""
    body = request.get_json(silent=True) or {}
    kind = body.get("kind", "")
    if kind not in SM.ANNOTATION_KINDS:
        return jsonify({"success": False, "error": f"kind 必须是 {list(SM.ANNOTATION_KINDS)} 之一"}), 400
    target = body.get("target", "")
    if not target:
        return jsonify({"success": False, "error": "target（消息 id）必填"}), 400
    s = current_session()
    try:
        ann = SM.add_annotation(
            s,
            target,
            kind,
            tags=[t for t in (body.get("tags") or []) if t],
            note=body.get("note", ""),
            author=body.get("author", "teacher") or "teacher",
        )
    except Exception as e:
        return jsonify({"success": False, "error": _scrub(str(e), get_registry())}), 400
    return jsonify({"success": True, "annotation": ann, "count": len(s.get("annotations", []))})


@app.route("/api/annotate/export", methods=["POST"])
def api_annotate_export():
    body = request.get_json(silent=True) or {}
    path = body.get("path", "")
    if not path:
        return jsonify({"success": False, "error": "path required"}), 400
    try:
        out = SM.export_annotations(current_session(), path, fmt=body.get("format", "csv"))
        return jsonify({"success": True, "path": out, "rows": len(SM.annotation_rows(current_session()))})
    except Exception as e:
        return jsonify({"success": False, "error": _scrub(str(e), get_registry())}), 500


# --------------------------------------------------------------------------
# ZIM external knowledge base
# --------------------------------------------------------------------------


@app.route("/api/zim/list")
def api_zim_list():
    try:
        lib = zim_kb.library()
        infos = lib.list_zims()
    except Exception as e:
        return jsonify({"success": False, "error": _scrub(f"{type(e).__name__}: {e}", get_registry()), "zims": []}), 500
    return jsonify(
        {
            "success": True,
            "roots": list(getattr(lib, "roots", None) or zim_kb.roots()),
            "zims": [z.as_dict() for z in infos],
        }
    )


@app.route("/api/zim/search", methods=["POST"])
def api_zim_search():
    body = request.get_json(silent=True) or {}
    query = (body.get("query") or "").strip()
    if not query:
        return jsonify({"success": False, "error": "query 不能为空"}), 400
    try:
        hits = zim_kb.library().search(
            query, body.get("zims") or None, top_k=int(body.get("top_k") or 8)
        )
    except Exception as e:
        return jsonify({"success": False, "error": _scrub(f"{type(e).__name__}: {e}", get_registry())}), 500
    return jsonify({"success": True, "query": query, "results": [h.as_dict() for h in hits]})


@app.route("/api/zim/article")
def api_zim_article():
    zim = request.args.get("zim", "")
    path = request.args.get("path", "")
    q = request.args.get("q", "")
    if not zim or not path:
        return jsonify({"success": False, "error": "zim 与 path 必填"}), 400
    try:
        art = zim_kb.library().article(zim, path)
    except KeyError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception as e:
        return jsonify({"success": False, "error": _scrub(f"{type(e).__name__}: {e}", get_registry())}), 500
    match = zim_kb.locate_passage(art["text"], q) if q else None
    return jsonify({"success": True, "match": match, **art})


@app.route("/api/render", methods=["POST"])
def api_render():
    """Render Markdown + math + code to self-contained HTML."""
    body = request.get_json(silent=True) or {}
    text = body.get("text") or ""
    try:
        rendered = MD.render_markdown(text, citations=bool(body.get("citations", True)))
    except Exception as e:
        return jsonify({"success": False, "error": _scrub(f"{type(e).__name__}: {e}", get_registry())}), 500
    return jsonify({"success": True, "html": rendered})


@app.route("/api/zim/suggest", methods=["POST"])
def api_zim_suggest():
    body = request.get_json(silent=True) or {}
    try:
        items = zim_kb.library().suggestions(body.get("prefix", ""), body.get("zim", ""), int(body.get("limit") or 8))
    except Exception as e:
        return jsonify({"success": False, "error": _scrub(f"{type(e).__name__}: {e}", get_registry())}), 500
    return jsonify({"success": True, "results": items})


# --------------------------------------------------------------------------
# pages
# --------------------------------------------------------------------------


@app.route("/timeline")
def timeline_page():
    """Timeline view for a `.clap` file, rendered by clap.core.timeline."""
    file_path = request.args.get("file", "")
    if not file_path:
        return (
            render_template(
                "timeline.html",
                error_key="timeline.paramMissing",
                error=None,
                html_body="",
                file_param="",
                title="Timeline",
            ),
            400,
        )
    try:
        s = SM.load_session(file_path)
        body = TL.render_timeline_html(s, title="CLAP Timeline")
        return render_template(
            "timeline.html",
            error_key=None,
            error=None,
            html_body=body,
            file_param=file_path,
            title=file_path.rsplit("/", 1)[-1],
        )
    except Exception as e:
        return (
            render_template(
                "timeline.html",
                error_key=None,
                error=_scrub(f"{type(e).__name__}: {e}", get_registry()),
                html_body="",
                file_param=file_path,
                title="Timeline",
            ),
            404,
        )


@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/kb")
def kb_page():
    return render_template("kb.html")


@app.route("/api-docs")
def api_docs():
    return render_template("api_docs.html")


# --------------------------------------------------------------------------
# kb endpoints (UNCHANGED contract)
# --------------------------------------------------------------------------


@app.route("/api/kb/index", methods=["POST"])
def kb_index():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "error": "No file"})
        f = request.files["file"]
        import os as _os

        path = _os.path.join("/tmp", f.filename)
        f.save(path)
        result = kb.index_document(path)
        _os.remove(path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": _scrub(str(e), get_registry())})


@app.route("/api/kb/search", methods=["POST"])
def kb_search():
    try:
        data = request.get_json() or {}
        results = kb.search(data.get("query", ""), data.get("k", 5))
        return jsonify({"success": True, "results": results})
    except Exception as e:
        return jsonify({"success": False, "error": _scrub(str(e), get_registry())})


@app.route("/api/kb/stats")
def kb_stats():
    return jsonify(kb.get_stats())


@app.route("/api/kb/clear", methods=["POST"])
def kb_clear():
    return jsonify(kb.clear())


def main():
    import argparse

    parser = argparse.ArgumentParser(description="CLAP Web Server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5057)
    args = parser.parse_args()

    print(f"CLAP web server: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
