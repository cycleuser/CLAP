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

LABELS = {
    "zh": {"title": "# 会话续期摘要", "session": "会话", "turns": "轮次", "period": "时间",
           "annotations": "标注", "content": "## 关键内容（按轮次）", "teacher": "教师",
           "assistant": "助手", "notes": "## 标注要点"},
    "en": {"title": "# Session renewal summary", "session": "Session", "turns": "Turns",
           "period": "Period", "annotations": "Annotations", "content": "## Key content (by turn)",
           "teacher": "Teacher", "assistant": "Assistant", "notes": "## Annotation notes"},
    "ja": {"title": "# セッション更新の要約", "session": "セッション", "turns": "ターン",
           "period": "期間", "annotations": "注釈", "content": "## 主な内容（ターン別）",
           "teacher": "教師", "assistant": "アシスタント", "notes": "## 注釈の要点"},
    "fr": {"title": "# Résumé de renouvellement", "session": "Session", "turns": "Tours",
           "period": "Période", "annotations": "Annotations", "content": "## Contenu clé (par tour)",
           "teacher": "Enseignant", "assistant": "Assistant", "notes": "## Points d'annotation"},
    "ru": {"title": "# Сводка обновления", "session": "Сессия", "turns": "Ходы",
           "period": "Период", "annotations": "Пометки", "content": "## Ключевое (по ходам)",
           "teacher": "Преподаватель", "assistant": "Ассистент", "notes": "## Основные пометки"},
    "de": {"title": "# Zusammenfassung der Erneuerung", "session": "Sitzung", "turns": "Runden",
           "period": "Zeitraum", "annotations": "Anmerkungen", "content": "## Kerninhalt (je Runde)",
           "teacher": "Lehrkraft", "assistant": "Assistent", "notes": "## Anmerkungen"},
    "es": {"title": "# Resumen de renovación", "session": "Sesión", "turns": "Turnos",
           "period": "Periodo", "annotations": "Anotaciones", "content": "## Contenido clave (por turno)",
           "teacher": "Docente", "assistant": "Asistente", "notes": "## Notas de anotación"},
    "pt": {"title": "# Resumo de renovação", "session": "Sessão", "turns": "Turnos",
           "period": "Período", "annotations": "Anotações", "content": "## Conteúdo principal (por turno)",
           "teacher": "Docente", "assistant": "Assistente", "notes": "## Notas de anotação"},
    "it": {"title": "# Riepilogo del rinnovo", "session": "Sessione", "turns": "Turni",
           "period": "Periodo", "annotations": "Annotazioni", "content": "## Contenuto chiave (per turno)",
           "teacher": "Docente", "assistant": "Assistente", "notes": "## Note di annotazione"},
    "ko": {"title": "# 세션 갱신 요약", "session": "세션", "turns": "턴",
           "period": "기간", "annotations": "주석", "content": "## 핵심 내용(턴별)",
           "teacher": "교사", "assistant": "도우미", "notes": "## 주석 요점"},
}
KIND_EN = {"keypoint": "Key point", "question": "Question", "generated": "Generated",
           "reviewed": "Review", "error_fix": "Correction", "knowledge_point": "Knowledge point"}


def branch_tokens_estimate(session: dict, branch_id: str | None = None) -> int:
    """Rough context footprint: CJK≈1 token/char, Latin≈1 token/4 chars."""
    bid = branch_id or (SM.main_branch(session) or {}).get("id") or ""
    msgs = SM.branch_full_path(session, bid) if bid else []
    chars = 0
    for m in msgs:
        text = m.get("content") or ""
        chars += len(text)
    return int(chars / 2.2)


def branch_digest(session: dict, branch_id: str | None = None, lang: str = "zh") -> dict:
    L = LABELS.get(lang if lang in LABELS else "en", LABELS["en"])
    bid = branch_id or (SM.main_branch(session) or {}).get("id") or ""
    msgs = [m for m in (SM.branch_full_path(session, bid) if bid else []) if m.get("role") in ("user", "assistant")]
    anns = [a for a in (session.get("annotations") or []) if a.get("target") in {m.get("id") for m in msgs}]
    branch = SM.get_branch(session, bid) or {}
    man = session.get("manifest") or {}

    users = [m for m in msgs if m.get("role") == "user"]
    first_ts = msgs[0].get("ts") if msgs else None
    last_ts = msgs[-1].get("ts") if msgs else None

    branch_name = branch.get("name") or ("Main line" if lang != "zh" else "主对话")
    if lang != "zh" and branch_name == "主线":
        branch_name = "Main line"
    lines = [
        L["title"],
        f"- {L['session']}: {man.get('session_id', '')} ({branch_name})",
        f"- {L['turns']}: {len(users)} Q / {len(msgs) - len(users)} A",
        f"- {L['period']}: {_fmt(first_ts)} → {_fmt(last_ts)}",
    ]
    if anns:
        kinds: dict[str, int] = {}
        for a in anns:
            kinds[a.get("kind", "")] = kinds.get(a.get("kind", ""), 0) + 1
        lines.append(f"- {L['annotations']}: " + "、".join(f"{k}×{v}" for k, v in kinds.items()))
    lines.append("")
    lines.append(L["content"])
    for i, m in enumerate(msgs, start=1):
        role = L["teacher"] if m.get("role") == "user" else L["assistant"]
        text = (m.get("content") or "").strip().replace("\n", " ")
        if len(text) > DIGEST_PER_MESSAGE:
            text = text[:DIGEST_PER_MESSAGE] + "…"
        lines.append(f"{i}. {role}：{text}")
    if anns:
        lines.append("")
        lines.append(L["notes"])
        for a in anns:
            tags = "、".join(a.get("tags") or [])
            note = a.get("note") or ""
            kind = a.get("kind", "")
            if lang != "zh":
                kind = KIND_EN.get(kind, kind)
            sep = ": " if lang != "zh" else "："
            lines.append(f"- [{kind}] {tags}{(sep + note) if note else ''}")
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


def renewal(session: dict, branch_id: str | None = None, lang: str = "zh") -> dict:
    """Build a renewal digest and record it in the manifest history."""
    digest = branch_digest(session, branch_id, lang)
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
