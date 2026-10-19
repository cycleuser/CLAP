"""CLAP-NEXT F4/F5: SVG/HTML figure exporters (matplotlib, blue-grey palette).

Palette discipline (paper F4): only blues and greys; no third hue. Text
labels are rendered to paths by matplotlib's svg.fonttype="path" so CJK
glyphs survive without font dependencies. All numbers come from the session
itself; unknown provenance renders as "—".

Public API:
- write_timeline_svg(session, path)  -> turn/branch timeline SVG
- write_timeline_png(session, path)  -> same figure as PNG (screenshots)
- write_annotation_distribution(session, path) -> S5 kind-distribution PNG
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from clap.core import session as session_mod  # noqa: E402
from clap.core.session import ANNOTATION_KINDS, KIND_LABELS  # noqa: E402

NULL_MARK = "—"

# blue-grey only, per the paper figure style
BLUE_DARK = "#1d4ed8"
BLUE = "#2563eb"
BLUE_LIGHT = "#93c5fd"
BLUE_PALE = "#eff6ff"
GREY_900 = "#111827"
GREY_700 = "#374151"
GREY_500 = "#6b7280"
GREY_300 = "#d1d5db"
GREY_100 = "#f3f4f6"


class _CJKFonts:
    done = False


def configure_cjk() -> bool:
    """Configure CJK-capable default font + svg text-as-paths.

    MUST run before any figure creation (matplotlib's findfont caches
    resolutions — late rcParams changes are too late and CJK text breaks).
    Called at import of this module; idempotent.
    """
    matplotlib.rcParams["svg.fonttype"] = "path"
    matplotlib.rcParams["axes.unicode_minus"] = False
    try:
        from matplotlib import font_manager

        names = {f.name for f in font_manager.fontManager.ttflist}
        for cand in ("PingFang SC", "Hiragino Sans GB", "Heiti SC", "Songti SC", "Arial Unicode MS", "Microsoft YaHei"):
            if cand in names:
                matplotlib.rcParams["font.family"] = [cand]
                _CJKFonts.done = True
                return True
    except Exception:
        pass
    return False


# CJK font configuration at import time (before any figure can exist)
configure_cjk()


def _style_axes(fig, ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GREY_300)
    ax.tick_params(colors=GREY_700, labelsize=9)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    configure_cjk()  # idempotent safety net


def _layout(session: dict):
    """Rows = branches; per branch the full context path (turn cards)."""
    branches = []
    for b in session.get("branches", []):
        full = session_mod.branch_full_path(session, b["id"])
        branches.append((b, session_mod.branch_path(session, b["id"]), full))
    return branches


def write_timeline_svg(session: dict, path, dpi: float = 150.0) -> str:
    """Branch timeline figure as SVG (turns as colored cards, annotations
    as markers). Returns the written path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["svg.fonttype"] = "path"

    branches = _layout(session)
    n_rows = max(1, len(branches))
    max_len = max(1, max(len(full) for _, _, full in branches))
    fig_h = 1.1 + 0.85 * n_rows + 0.25 * max_len * 0.06
    fig, ax = plt.subplots(figsize=(min(14, 3 + max_len * 0.55), max(3.0, fig_h)))

    man = session.get("manifest", {})
    for row, (b, _own, full) in enumerate(reversed(branches)):  # main line last → top
        y = row
        bmodel = b.get("model") or man.get("chat_model") or "?"
        ax.text(
            -0.4, y + 0.18, f'{b["name"]} ({b["id"]})',
            ha="right", va="center", fontsize=10, color=GREY_900, fontweight="bold",
        )
        ax.text(-0.4, y - 0.24, bmodel, ha="right", va="center", fontsize=8, color=GREY_500)

        for idx, m in enumerate(full):
            role = m.get("role", "")
            if role == "user":
                face, edge, txt = BLUE_LIGHT, BLUE, GREY_900
            elif role == "assistant":
                face, edge, txt = "#ffffff", GREY_500, GREY_700
            else:
                face, edge, txt = GREY_100, GREY_300, GREY_500
            if m.get("error"):
                edge = "#b91c1c"
            box = dict(boxstyle="round,pad=0.32", facecolor=face, edgecolor=edge, linewidth=1.2)
            label = f"{idx + 1}. {m.get('id', '?')}"
            ax.text(idx + 0.5, y, label, ha="center", va="center", fontsize=8,
                    color=txt, bbox=box)

            # annotation markers under the card
            anns = [a for a in session.get("annotations", []) if a.get("target") == m.get("id")]
            for k, a in enumerate(anns):
                kind = a.get("kind", "")
                ax.text(
                    idx + 0.5, y - 0.34 - 0.16 * k,
                    KIND_LABELS.get(kind, kind),
                    ha="center", va="center", fontsize=7, color=BLUE_DARK,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor=BLUE_PALE, edgecolor=BLUE, linewidth=0.8),
                )

            # fork marker: branch root cards get a triangle above
            if m.get("id") == b.get("root") and b.get("fork_at"):
                ax.annotate(
                    "fork", xy=(idx + 0.5, y + 0.42), ha="center", fontsize=7,
                    color=BLUE_DARK,
                )

    ax.set_xlim(-len(str(len(branches))) * 0.6 - 2.2, max_len + 1)
    ax.set_ylim(-1.0, n_rows - 0.35)
    ax.set_xticks(range(1, max_len + 1))
    ax.set_xticklabels([str(i) for i in range(1, max_len + 1)], fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("turn index over branch context path", fontsize=9, color=GREY_700)
    _style_axes(fig, ax)
    title = (
        f'Session {man.get("session_id", "?")} · '
        f'purpose {man.get("purpose", "?")} · '
        f'{len(session.get("messages", []))} messages, '
        f'{len(branches)} branches, {len(session.get("annotations", []))} annotations'
    )
    ax.set_title(title, fontsize=11, color=GREY_900, pad=12, loc="left")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return str(path)


def write_timeline_png(session: dict, path, dpi: float = 130.0) -> str:
    """Same figure as PNG (for quick viewing); returns the written path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    branches = _layout(session)
    n_rows = max(1, len(branches))
    max_len = max(1, max(len(full) for _, _, full in branches))
    fig, ax = plt.subplots(figsize=(min(14, 3 + max_len * 0.55), max(3.0, 1.1 + 0.95 * n_rows)))
    man = session.get("manifest", {})
    for row, (b, _own, full) in enumerate(reversed(branches)):
        y = row
        bmodel = b.get("model") or man.get("chat_model") or "?"
        ax.text(-0.4, y + 0.18, f'{b["name"]} ({b["id"]})', ha="right", va="center",
                fontsize=10, color=GREY_900, fontweight="bold")
        ax.text(-0.4, y - 0.24, bmodel, ha="right", va="center", fontsize=8, color=GREY_500)
        for idx, m in enumerate(full):
            role = m.get("role", "")
            if role == "user":
                face, edge = BLUE_LIGHT, BLUE
            elif role == "assistant":
                face, edge = "#ffffff", GREY_500
            else:
                face, edge = GREY_100, GREY_300
            if m.get("error"):
                edge = "#b91c1c"
            ax.text(
                idx + 0.5, y, f"{idx + 1}. {m.get('id', '?')}",
                ha="center", va="center", fontsize=8, color=GREY_900,
                bbox=dict(boxstyle="round,pad=0.32", facecolor=face, edgecolor=edge, linewidth=1.2),
            )
            anns = [a for a in session.get("annotations", []) if a.get("target") == m.get("id")]
            for k, a in enumerate(anns):
                ax.text(idx + 0.5, y - 0.36 - 0.17 * k, KIND_LABELS.get(a.get("kind", ""), a.get("kind", "")),
                        ha="center", va="center", fontsize=7, color=BLUE_DARK,
                        bbox=dict(boxstyle="round,pad=0.18", facecolor=BLUE_PALE, edgecolor=BLUE, linewidth=0.8))
    ax.set_xlim(-2.2, max_len + 1)
    ax.set_ylim(-1.0, n_rows - 0.35)
    ax.set_yticks([])
    ax.set_xticks(range(1, max_len + 1))
    _style_axes(fig, ax)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    return str(path)


def write_annotation_distribution(session: dict, path) -> str:
    """S5: kind × count distribution figure (learning-analytics preview)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = session_mod.count_kinds(session)
    kinds = list(ANNOTATION_KINDS)
    values = [counts[k] for k in kinds]
    labels = [KIND_LABELS[k] for k in kinds]

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    y = list(range(len(kinds)))[::-1]
    ax.barh(y, values, color=BLUE, edgecolor=BLUE_DARK, height=0.62)
    for yi, v in zip(y, values, strict=True):
        if v == 0:
            ax.text(0.08, yi, "0", va="center", fontsize=9, color=GREY_500)
        else:
            ax.text(v + 0.06, yi, str(v), va="center", fontsize=10, color=GREY_900)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("annotation count", fontsize=10, color=GREY_700)
    total = sum(values)
    ax.set_title(
        f'Teacher annotations by kind (n={total}, session '
        f'{session.get("manifest", {}).get("session_id", "?")})',
        fontsize=11, loc="left", color=GREY_900,
    )
    ax.set_xlim(0, max(1, max(values) * 1.25))
    _style_axes(fig, ax)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="png", bbox_inches="tight", dpi=140)
    plt.close(fig)
    return str(path)
