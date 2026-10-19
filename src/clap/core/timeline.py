"""CLAP-NEXT F4: timeline / branch-tree visualization.

`render_timeline_html` produces a fully self-contained HTML document (inline
CSS/JS, zero external links) showing per-turn cards with real provenance
(model/provider/latency/tokens — rendered as "—" when unknown, never
fabricated), branch sections with fork markers, and annotation badges.

Fidelity rule: only data present in the v2 session appears in the figure.
"""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

from clap.core import session as session_mod
from clap.core.session import KIND_LABELS

# footer version string; single source in session.py
CAP_VERSION = getattr(session_mod, "CLAP_VERSION", "2.0.0")

NULL_MARK = "—"

# labels used when the timeline is rendered for a non-Chinese interface
KIND_LABELS_EN = {
    "keypoint": "Key point",
    "question": "Question",
    "generated": "Generated",
    "reviewed": "Review",
    "error_fix": "Correction",
    "knowledge_point": "Knowledge point",
}
MAIN_LINE = {
    "zh": "主线", "en": "Main line", "ja": "メイン", "fr": "Ligne principale",
    "ru": "Основная линия", "de": "Hauptlinie", "es": "Línea principal",
    "pt": "Linha principal", "it": "Linea principale", "ko": "메인 라인",
}
PURPOSE_LABELS = {"zh": {"答疑": "答疑", "备课": "备课"}, "en": {"答疑": "Q&A", "备课": "Lesson prep"}}


def _kind_label(kind: str, lang: str) -> str:
    if lang == "zh":
        return KIND_LABELS.get(kind, kind)
    return KIND_LABELS_EN.get(kind, kind)


def _branch_name(branch: dict, lang: str) -> str:
    name = branch.get("name") or branch.get("id")
    if name == "主线":
        return MAIN_LINE.get(lang, "Main line")
    return name


def _purpose_label(purpose: str, lang: str) -> str:
    return PURPOSE_LABELS.get("en" if lang != "zh" else "zh", {}).get(purpose, purpose)


def _esc(text) -> str:
    return html.escape(str(text if text is not None else ""))


def _maybe(value) -> str:
    return _esc(value) if value not in (None, "") else NULL_MARK


def build_template_html() -> str:
    """The HTML/CSS shell with a `{{CONTENT}}` marker (assets template source)."""
    try:
        res = Path(__file__).resolve().parent.parent / "gui" / "assets" / "timeline.html"
        return res.read_text(encoding="utf-8")
    except Exception:  # package data missing (rare): minimal inline shell
        return (
            "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>CLAP Timeline</title></head>"
            "<body>{{CONTENT}}</body></html>"
        )


def _annotation_details(session: dict, msg_id: str, lang: str = "zh") -> tuple[str, str]:
    badges = ""
    notes = []
    for a in session.get("annotations", []):
        if a.get("target") != msg_id:
            continue
        label = _kind_label(a.get("kind", ""), lang)
        badges += f'<span class="badge ann-{_esc(a.get("kind", ""))}">{_esc(label)}</span>'
        parts = []
        if a.get("tags"):
            parts.append(_esc(",".join(a["tags"])))
        if a.get("note"):
            parts.append(_esc(a["note"]))
        if a.get("author"):
            parts.append(f"@{_esc(a['author'])}")
        notes.append(f'<div class="annnote">{_esc(label)}: {" · ".join(parts)}</div>')
    return badges, "".join(notes)


def _turn_card(session: dict, m: dict, index: int, branch_root_id: str | None,
               lang: str = "zh") -> str:
    role = m.get("role", "")
    tok = m.get("tokens") or {}
    badges, ann_notes = _annotation_details(session, m.get("id", ""), lang)
    classes = "card"
    if role == "user":
        classes += " user"
    elif role == "system":
        classes += " system"
    if m.get("id") == branch_root_id:
        classes += " branch-root"
    if m.get("error"):
        classes += " errored"

    latency = NULL_MARK if m.get("latency_ms") is None else f'{_esc(m["latency_ms"])} ms'
    tokens = f'{NULL_MARK if tok.get("in") is None else _esc(tok["in"])} / {NULL_MARK if tok.get("out") is None else _esc(tok["out"])}'
    err_line = f'<div class="errline">error: {_esc(m.get("error"))}</div>' if m.get("error") else ""
    return (
        f'<div class="{classes}">'
        f'<div class="turnno">{index}</div>'
        f'<div class="body">'
        f'<div class="meta">'
        f'<span class="role role-{_esc(role)}">{_esc(role)}</span>'
        f'<span class="model">{_maybe(m.get("model"))}</span>'
        f'<span class="provider">{_maybe(m.get("provider"))}</span>'
        f'<span class="lat">{latency}</span>'
        f'<span class="tok">{tokens} tok</span>'
        f'<span class="mid">id:{_esc(m.get("id"))}</span>'
        f"{badges}"
        f"</div>"
        f"{err_line}"
        f'<div class="content">{_esc(m.get("content", ""))}</div>'
        f"{ann_notes}"
        f"</div></div>"
    )


def render_timeline_html(session: dict, title: str = "CLAP Session Timeline",
                         lang: str = "zh") -> str:
    """Render the full self-contained timeline HTML for a v2 session."""
    man = session.get("manifest", {})
    doc = man.get("document") or {}

    sections = []
    for b in session.get("branches", []):
        bmodel = b.get("model") or man.get("chat_model") or NULL_MARK
        own = session_mod.branch_path(session, b["id"])
        full = session_mod.branch_full_path(session, b["id"])
        fork_note = ""
        if b.get("fork_at"):
            fork_at_msg = session_mod.message(session, b["fork_at"]) or {}
            fork_note = (
                f'<span class="forkmeta">fork ← {b["fork_at"]}'
                f'{": " + _esc((fork_at_msg.get("content") or "")[:24]) if fork_at_msg else ""}</span>'
            )
        # index = position over the full context path (1-based)
        cards = [
            _turn_card(session, m, i + 1, b.get("root"), lang)
            for i, m in enumerate(full)
        ]
        sections.append(
            f'<section class="branch" id="{_esc(b["id"])}">'
            f'<h2><span class="bname">{_esc(_branch_name(b, lang))}</span>'
            f'<span class="bmodel">{_esc(bmodel)}</span>'
            f'<span class="bturns">own {len(own)} turns / context {len(full)}</span>'
            f"{fork_note}</h2>"
            f'<div class="turns">{"".join(cards)}</div>'
            f"</section>"
        )

    gen_time = datetime.now().isoformat(timespec="seconds")
    footer_note = "Self-contained offline HTML, no external links" if lang != "zh" else "自包含离线 HTML，无外链"
    payload = f"""
<div class="wrap">
  <header>
    <h1>{_esc(title)}</h1>
    <div class="facts">
      <span>session <b>{_esc(man.get("session_id", NULL_MARK))}</b></span>
      <span>purpose <b>{_esc(_purpose_label(man.get("purpose", NULL_MARK), lang))}</b></span>
      <span>chat model <b>{_esc(man.get("chat_model") or NULL_MARK)}</b></span>
      <span>provider <b>{_esc(man.get("provider") or NULL_MARK)}</b></span>
      <span>embed model <b>{_esc(man.get("embed_model") or NULL_MARK)}</b></span>
      <span>document <b>{_esc(doc.get("name") or NULL_MARK)}</b></span>
      <span>privacy_mode <b>{_esc("on" if man.get("privacy_mode") else "off")}</b></span>
      <span>routing <b>{_esc(man.get("routing") or NULL_MARK)}</b></span>
      <span>messages <b>{len(session.get("messages", []))}</b></span>
      <span>branches <b>{len(session.get("branches", []))}</b></span>
      <span>annotations <b>{len(session.get("annotations", []))}</b></span>
      <span>created <b>{_esc(session.get("created") or NULL_MARK)}</b></span>
    </div>
  </header>
  {"".join(sections)}
  <footer>Generated by CLAP {getattr(session_mod, 'CLAP_VERSION', '2.0.0')} · {gen_time} · {footer_note}</footer>
</div>"""
    return build_template_html().replace("{{CONTENT}}", payload)


def branch_tree_spec(session: dict) -> dict:
    """Machine-readable branch-tree for the SVG exporter (F4)."""
    nodes = []
    for b in session.get("branches", []):
        own = session_mod.branch_path(session, b["id"])
        full = session_mod.branch_full_path(session, b["id"])
        nodes.append(
            {
                "id": b["id"],
                "name": b["name"] or b["id"],
                "fork_at": b.get("fork_at"),
                "model": b.get("model") or session.get("manifest", {}).get("chat_model"),
                "own_len": len(own),
                "context_len": len(full),
                "root": b.get("root"),
                "path_ids": [m["id"] for m in own],
            }
        )
    return {
        "branches": nodes,
        "annotations": len(session.get("annotations", [])),
        "session_id": session.get("manifest", {}).get("session_id", ""),
    }
