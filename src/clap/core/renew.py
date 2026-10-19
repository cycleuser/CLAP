"""Session renewal («续期»): bridge a long conversation into fresh context.

A renewal is a deterministic digest of the active branch (no model call, so
it works with every provider and cannot hallucinate). The digest is injected
as a `system` message ahead of the next question, keeping the thread of the
conversation while dropping raw context weight.
"""

from __future__ import annotations

from datetime import datetime

from clap.core import session as SM

DIGEST_PER_MESSAGE = 160


def branch_tokens_estimate(session: dict, branch_id: str | None = None) -> int:
    """Rough context footprint: CJK≈1 token/char, Latin≈1 token/4 chars."""
    bid = branch_id or (SM.main_branch(session) or {}).get("id") or ""
    msgs = SM.branch_full_path(session, bid) if bid else []
    chars = 0
    for m in msgs:
        text = m.get("content") or ""
        chars += len(text)
    return int(chars / 2.2)


def branch_digest(session: dict, branch_id: str | None = None) -> dict:
    bid = branch_id or (SM.main_branch(session) or {}).get("id") or ""
    msgs = [m for m in (SM.branch_full_path(session, bid) if bid else []) if m.get("role") in ("user", "assistant")]
    anns = [a for a in (session.get("annotations") or []) if a.get("target") in {m.get("id") for m in msgs}]
    branch = SM.get_branch(session, bid) or {}
    man = session.get("manifest") or {}

    users = [m for m in msgs if m.get("role") == "user"]
    first_ts = msgs[0].get("ts") if msgs else None
    last_ts = msgs[-1].get("ts") if msgs else None

    lines = [
        "# 会话续期摘要",
        f"- 会话：{man.get('session_id', '')}（{branch.get('name') or '主对话'}）",
        f"- 轮次：{len(users)} 问 / {len(msgs) - len(users)} 答",
        f"- 时间：{_fmt(first_ts)} → {_fmt(last_ts)}",
    ]
    if anns:
        kinds: dict[str, int] = {}
        for a in anns:
            kinds[a.get("kind", "")] = kinds.get(a.get("kind", ""), 0) + 1
        lines.append("- 标注：" + "、".join(f"{k}×{v}" for k, v in kinds.items()))
    lines.append("")
    lines.append("## 关键内容（按轮次）")
    for i, m in enumerate(msgs, start=1):
        role = "教师" if m.get("role") == "user" else "助手"
        text = (m.get("content") or "").strip().replace("\n", " ")
        if len(text) > DIGEST_PER_MESSAGE:
            text = text[:DIGEST_PER_MESSAGE] + "…"
        lines.append(f"{i}. {role}：{text}")
    if anns:
        lines.append("")
        lines.append("## 标注要点")
        for a in anns:
            tags = "、".join(a.get("tags") or [])
            note = a.get("note") or ""
            lines.append(f"- ［{a.get('kind', '')}］{tags}{('：' + note) if note else ''}")
    summary = "\n".join(lines).strip()
    return {
        "branch_id": bid,
        "summary": summary,
        "chars": len(summary),
        "messages": len(msgs),
        "turns": len(users),
        "est_tokens_before": branch_tokens_estimate(session, bid),
        "est_tokens_after": int(len(summary) / 2.2),
    }


def renewal(session: dict, branch_id: str | None = None) -> dict:
    """Build a renewal digest and record it in the manifest history."""
    digest = branch_digest(session, branch_id)
    SM._add_history(
        session,
        "renewal",
        {
            "branch": digest["branch_id"],
            "messages": digest["messages"],
            "est_tokens_before": digest["est_tokens_before"],
            "est_tokens_after": digest["est_tokens_after"],
        },
    )
    return digest


def system_message_for(summary: str) -> dict:
    """The bridge message injected ahead of the next question."""
    return {
        "role": "system",
        "content": "以下是本会话此前的续期摘要，请在回答时保持上下文一致：\n\n" + (summary or ""),
    }


def _fmt(ms) -> str:
    if not ms:
        return "—"
    try:
        return datetime.fromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "—"
