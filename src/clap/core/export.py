"""Session export (Markdown / HTML / JSON) with provenance and citations.

Every exporter takes an optional branch id; when omitted the main branch is
used. Exports are self-contained: the HTML carries its own inline CSS (the
same blue-grey design tokens as the web UI) and references no CDN, font or
script, so a saved file opens correctly offline.
"""

from __future__ import annotations

import json
from datetime import datetime

from clap.core import session as SM

ROLE_ZH = {"user": "教师", "assistant": "助手", "system": "系统"}


def _ts(ms) -> str:
    if not ms:
        return ""
    try:
        return datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


def _branch_messages(session: dict, branch_id: str | None) -> tuple[str, list[dict]]:
    bid = branch_id or (SM.main_branch(session) or {}).get("id") or ""
    if not bid:
        return "", list(session.get("messages") or [])
    try:
        msgs = SM.branch_full_path(session, bid)
    except Exception:
        msgs = SM.branch_path(session, bid)
    return bid, [m for m in msgs if m.get("role") in ("user", "assistant", "system")]


def _ann_map(session: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for a in session.get("annotations") or []:
        out.setdefault(a.get("target", ""), []).append(a)
    return out


def _meta_line(m: dict) -> str:
    bits = []
    if m.get("provider") or m.get("model"):
        bits.append(f"{m.get('provider') or '?'} · {m.get('model') or '?'}")
    if m.get("latency_ms") is not None:
        bits.append(f"{m['latency_ms']} ms")
    tok = m.get("tokens") or {}
    if tok.get("out") is not None:
        bits.append(f"输出 {tok['out']} tok")
    if m.get("error"):
        bits.append(f"错误：{m['error']}")
    return " · ".join(bits)


def _sources_of(m: dict) -> list[dict]:
    return list(m.get("sources") or [])


def _audit_of(m: dict) -> dict:
    return dict(m.get("audit") or {})


def export_markdown(session: dict, branch_id: str | None = None) -> str:
    bid, msgs = _branch_messages(session, branch_id)
    anns = _ann_map(session)
    man = session.get("manifest") or {}
    branch = SM.get_branch(session, bid) or {}
    lines: list[str] = []
    title = branch.get("name") or "主对话"
    lines.append(f"# CLAP 会话 · {title}")
    lines.append("")
    lines.append(f"- 会话 ID：`{man.get('session_id', '')}`")
    lines.append(f"- 分支：`{bid or '—'}`（{len(msgs)} 条消息）")
    lines.append(f"- 用途：{man.get('purpose', '')}")
    lines.append(f"- 导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    for i, m in enumerate(msgs, start=1):
        role = ROLE_ZH.get(m.get("role"), m.get("role", ""))
        meta = _meta_line(m)
        lines.append(f"## {i}. {role}" + (f" · {meta}" if meta else ""))
        lines.append("")
        lines.append(m.get("content") or "")
        lines.append("")
        for a in anns.get(m.get("id", ""), []):
            tags = "、".join(a.get("tags") or [])
            note = f" — {a.get('note')}" if a.get("note") else ""
            lines.append(f"> 标注［{a.get('kind', '')}］{tags}{note}")
            lines.append("")
        srcs = _sources_of(m)
        if srcs:
            lines.append("**引用来源**")
            lines.append("")
            for s in srcs:
                lines.append(f"- [{s.get('n', '')}] {s.get('title', '')}（{s.get('zim', '')} · {s.get('path', '')}）")
            lines.append("")
        aud = _audit_of(m)
        if aud:
            lines.append(
                f"**依据度**：{int(round(float(aud.get('score', 0)) * 100))}%"
                f" · {aud.get('sentences_grounded', 0)}/{aud.get('sentences_total', 0)} 句有据"
            )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


_HTML_HEAD = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{title}</title>
<style>
  :root {{ --blue-700:#1d4ed8; --blue-600:#2563eb; --blue-100:#dbeafe; --blue-50:#eff6ff;
    --grey-900:#111827; --grey-700:#374151; --grey-500:#6b7280; --grey-300:#d1d5db;
    --grey-100:#f3f4f6; --grey-50:#f9fafb; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --blue-700:#93b4ff; --blue-600:#7aa2ff; --blue-100:#1e3a8a; --blue-50:#172033;
      --grey-900:#e5e7eb; --grey-700:#cbd5e1; --grey-500:#94a3b8; --grey-300:#334155;
      --grey-100:#1e293b; --grey-50:#0f172a; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--grey-50); color:var(--grey-900);
    font:16px/1.7 "PingFang SC","Hiragino Sans GB","Microsoft YaHei",system-ui,sans-serif; }}
  .page {{ max-width:920px; margin:0 auto; padding:32px 20px 64px; }}
  h1 {{ font-size:26px; margin:0 0 6px; }}
  .sub {{ color:var(--grey-500); font-size:14px; margin-bottom:28px; }}
  .turn {{ background:#fff; border:1px solid var(--grey-300); border-radius:14px;
    padding:16px 20px; margin:0 0 14px; }}
  @media (prefers-color-scheme: dark) {{ .turn {{ background:var(--grey-100); }} }}
  .head {{ display:flex; flex-wrap:wrap; gap:8px 14px; align-items:baseline;
    font-size:13px; color:var(--grey-500); margin-bottom:8px; }}
  .head b {{ color:var(--blue-600); font-size:15px; }}
  .content {{ white-space:pre-wrap; word-break:break-word; }}
  .cite {{ display:inline-block; background:var(--blue-50); color:var(--blue-700);
    border:1px solid var(--blue-100); border-radius:6px; padding:0 6px; margin:0 2px;
    font-size:12px; cursor:default; }}
  .ann {{ margin-top:10px; font-size:13px; color:var(--grey-700);
    background:var(--blue-50); border-left:3px solid var(--blue-600); padding:6px 10px; border-radius:6px; }}
  .src {{ margin-top:10px; font-size:13px; color:var(--grey-700); }}
  .src li {{ margin:2px 0; }}
  .audit {{ display:inline-block; margin-top:10px; font-size:12px; color:var(--blue-700);
    background:var(--blue-50); border:1px solid var(--blue-100); border-radius:999px; padding:2px 10px; }}
  .ungrounded {{ border-color:#f59e0b !important; }}
  @media (max-width:480px) {{ .page {{ padding:18px 12px 40px; }} h1 {{ font-size:21px; }} .turn {{ padding:12px 14px; }} }}
</style></head><body><div class="page">
<h1>{title}</h1>
<div class="sub">{sub}</div>
"""

_HTML_TAIL = "</div></body></html>\n"


def _esc(text: str) -> str:
    return (
        (text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _cite_html(content: str) -> str:
    """Wrap [n] markers so they read as citation chips (escape everything else)."""
    import re

    safe = _esc(content)
    return re.sub(r"\[(\d+(?:\s*[,，]\s*\d+)*)\]", r'<span class="cite">[\1]</span>', safe)


def export_html(session: dict, branch_id: str | None = None) -> str:
    bid, msgs = _branch_messages(session, branch_id)
    anns = _ann_map(session)
    man = session.get("manifest") or {}
    branch = SM.get_branch(session, bid) or {}
    title = f"CLAP 会话 · {branch.get('name') or '主对话'}"
    sub = (
        f"会话 {man.get('session_id', '')} · 分支 {bid or '—'} · {len(msgs)} 条消息 · "
        f"导出于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    parts = [_HTML_HEAD.format(title=_esc(title), sub=_esc(sub))]
    for i, m in enumerate(msgs, start=1):
        role = ROLE_ZH.get(m.get("role"), m.get("role", ""))
        meta = _meta_line(m)
        head = f"<b>{i}. {_esc(role)}</b>" + (f"<span>{_esc(meta)}</span>" if meta else "")
        cls = "turn ungrounded" if (m.get("audit") or {}).get("score", 1) < 0.6 and m.get("audit") else "turn"
        parts.append(f'<article class="{cls}"><div class="head">{head}</div>')
        parts.append(f'<div class="content">{_cite_html(m.get("content") or "")}</div>')
        for a in anns.get(m.get("id", ""), []):
            tags = "、".join(a.get("tags") or [])
            note = f" — {_esc(a.get('note'))}" if a.get("note") else ""
            parts.append(f'<div class="ann">标注［{_esc(a.get("kind", ""))}］{_esc(tags)}{note}</div>')
        srcs = _sources_of(m)
        if srcs:
            items = "".join(
                f"<li>[{s.get('n', '')}] {_esc(s.get('title', ''))}"
                f"（{_esc(s.get('zim', ''))} · {_esc(s.get('path', ''))}）</li>"
                for s in srcs
            )
            parts.append(f'<div class="src"><ul>{items}</ul></div>')
        aud = _audit_of(m)
        if aud:
            pct = int(round(float(aud.get("score", 0)) * 100))
            parts.append(
                f'<div class="audit">依据度 {pct}% · '
                f'{aud.get("sentences_grounded", 0)}/{aud.get("sentences_total", 0)} 句有据</div>'
            )
        parts.append("</article>")
    parts.append(_HTML_TAIL)
    return "".join(parts)


def export_json(session: dict, branch_id: str | None = None) -> str:
    bid, msgs = _branch_messages(session, branch_id)
    payload = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "branch_id": bid,
        "messages": msgs,
        "annotations": [
            a for a in (session.get("annotations") or []) if a.get("target") in {m.get("id") for m in msgs}
        ],
        "manifest": {
            k: (session.get("manifest") or {}).get(k)
            for k in ("session_id", "purpose", "language", "privacy_mode", "created")
        },
        "session": session,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


FORMATS = {"md": export_markdown, "markdown": export_markdown, "html": export_html, "json": export_json}


def export(session: dict, fmt: str = "md", branch_id: str | None = None, path: str | None = None) -> tuple[str, str]:
    """Return (content, written_path or '')."""
    fn = FORMATS.get((fmt or "md").lower())
    if fn is None:
        raise ValueError(f"未知导出格式 {fmt!r}（可选：md / html / json）")
    content = fn(session, branch_id)
    if path:
        from pathlib import Path

        p = Path(path)
        if p.parent and str(p.parent) not in ("", "."):
            p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return content, str(p)
    return content, ""
