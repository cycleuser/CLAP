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

DEFAULT_PROVIDER = "ollama-cloud"
GROUNDED_TOP_K = 6

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
    """Default provider/model, resolved from live availability (no guessing)."""
    pid = DEFAULT_PROVIDER if DEFAULT_PROVIDER in registry else ""
    if not pid:
        preferred = [p for p in sorted(registry) if p in ("deepseek", "ollama-cloud", "huggingface")]
        pid = preferred[0] if preferred else (sorted(registry)[0] if registry else "")
    model = PR.default_model_for(registry, pid) if pid else ""
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
           top_k?, renew_summary?}
    """
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"success": False, "error": "消息为空"}), 400

    registry = get_registry()
    pid = data.get("provider") or default_selection(registry)["provider"]
    if pid not in registry:
        return (
            jsonify({"success": False, "error": f"provider「{pid}」未配置（环境变量或配置文件里没有它的凭证）"}),
            400,
        )

    model = (data.get("model") or "").strip() or PR.default_model_for(registry, pid)
    if not model:
        return jsonify({"success": False, "error": f"provider「{pid}」没有可用模型"}), 400

    s = current_session()
    branch = _tools_branch_id(s) if data.get("lane") == "tools" else _branch_id(s, data.get("branch"))

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
            parts.append("来源未提及（检索未命中任何材料，请换关键词或指定 ZIM 库）。")
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
# session endpoints
# --------------------------------------------------------------------------


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
            "messages": SM.branch_full_path(s, b),
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
            "history": [h for h in s["manifest"].get("history", []) if h.get("event") in ("privacy_note", "renewal")][-8:],
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
    if not zim or not path:
        return jsonify({"success": False, "error": "zim 与 path 必填"}), 400
    try:
        art = zim_kb.library().article(zim, path)
    except KeyError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception as e:
        return jsonify({"success": False, "error": _scrub(f"{type(e).__name__}: {e}", get_registry())}), 500
    return jsonify({"success": True, **art})


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
                error="缺少 file 参数。示例：/timeline?file=/path/to/session.clap",
                html_body="",
                file_param="",
                title="时间轴视图",
            ),
            400,
        )
    try:
        s = SM.load_session(file_path)
        body = TL.render_timeline_html(s, title="CLAP 时间轴")
        return render_template(
            "timeline.html",
            error=None,
            html_body=body,
            file_param=file_path,
            title=file_path.rsplit("/", 1)[-1],
        )
    except Exception as e:
        return (
            render_template(
                "timeline.html",
                error=_scrub(f"无法渲染：{type(e).__name__}: {e}", get_registry()),
                html_body="",
                file_param=file_path,
                title="时间轴视图",
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
