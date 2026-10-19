"""Render assistant Markdown with math and code, server-side and offline.

Why server-side
---------------
The page must stay self-contained (no CDN, no external fonts or scripts).
Rendering Markdown here means the browser receives finished HTML:

* **Markdown** — mistune (already a CLAP dependency) with tables,
  strikethrough and autolinks; raw HTML in model output is escaped.
* **Math** — `$...$`, `$$...$$`, `\\(...\\)`, `\\[...\\]` are converted to
  MathML with latex2mathml, which browsers render natively (no KaTeX download).
* **Code** — fenced blocks are highlighted with Pygments when it is available.
* **Citations** — `[n]` is turned into a chip carrying the *sentence* it sits
  in, so clicking it can jump to the matching passage in the original ZIM
  article.

Everything is pure Python; failures degrade to escaped text instead of
throwing.
"""

from __future__ import annotations

import html
import re

import mistune

try:  # syntax highlighting is optional
    from pygments import highlight as _pyg_highlight
    from pygments.formatters import HtmlFormatter as _PygFormatter
    from pygments.lexers import get_lexer_by_name as _pyg_lexer
    from pygments.util import ClassNotFound as _PygNotFound

    HAS_PYGMENTS = True
except Exception:  # pragma: no cover - optional dependency
    HAS_PYGMENTS = False

try:  # LaTeX → MathML
    from latex2mathml.converter import convert as _latex_to_mathml

    HAS_MATHML = True
except Exception:  # pragma: no cover - optional dependency
    HAS_MATHML = False


_FENCE_RE = re.compile(r"^([ \t]*)```([A-Za-z0-9_+.#-]*)[ \t]*\n(.*?)^\1```[ \t]*$", re.S | re.M)
_DISPLAY_MATH_RE = re.compile(r"\$\$(.+?)\$\$", re.S)
_BRACKET_MATH_RE = re.compile(r"\\\[(.+?)\\\]", re.S)
_INLINE_DOLLAR_RE = re.compile(r"(?<![\\$])\$(?!\s)([^$\n]{1,400}?)(?<!\s)\$(?!\$)")
_INLINE_PAREN_RE = re.compile(r"\\\((.+?)\\\)")

_CITE_RE = re.compile(r"\[(\d+(?:\s*[,，]\s*\d+)*)\]")
_SENTENCE_BOUNDARIES = "。！？!?；;\n"

_DANGEROUS_HREF = re.compile(r'href="\s*(javascript|data|vbscript):', re.I)


def _esc(text: str) -> str:
    return html.escape(text or "", quote=True)


def _render_code(code: str, lang: str) -> str:
    code = code.rstrip("\n")
    label = _esc(lang or "text")
    body_html: str
    if HAS_PYGMENTS and lang:
        try:
            lexer = _pyg_lexer(lang)
            body_html = _pyg_highlight(code, lexer, _PygFormatter(nowrap=True))
        except _PygNotFound:
            body_html = _esc(code)
    elif HAS_PYGMENTS and not lang:
        body_html = _esc(code)
    else:
        body_html = _esc(code)
    return (
        f'<div class="code-block"><div class="code-head">{label}</div>'
        f'<pre><code class="hl">{body_html}</code></pre></div>'
    )


def _render_math(tex: str, display: bool) -> str:
    tex = tex.strip()
    if HAS_MATHML and tex:
        try:
            mathml = _latex_to_mathml(tex)
            tag = "div" if display else "span"
            cls = "math math-block" if display else "math math-inline"
            return f'<{tag} class="{cls}" data-tex="{_esc(tex)}">{mathml}</{tag}>'
        except Exception:
            pass
    tag = "div" if display else "span"
    cls = "math math-block raw" if display else "math math-inline raw"
    return f'<{tag} class="{cls}"><code>{_esc(tex)}</code></{tag}>'


def _sentence_at(text: str, start: int, end: int) -> str:
    """Sentence (or line) containing the citation at [start, end).

    When a citation sits on its own line after a display formula («…\n[1]。»)
    the raw sentence is nearly empty; in that case walk back up to two more
    sentence boundaries so the locating query still carries real prose.
    """
    def extract(left_start: int) -> tuple[int, int]:
        left = left_start
        while left > 0 and text[left - 1] not in _SENTENCE_BOUNDARIES:
            left -= 1
        right = end
        while right < len(text) and text[right] not in _SENTENCE_BOUNDARIES:
            right += 1
        if right < len(text):
            right += 1
        return left, right

    left, right = extract(start)
    for _ in range(3):
        candidate = text[left:right]
        prose = re.sub(r"\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\$[^$\n]*\$|\\\(.*?\\\)", " ", candidate)
        prose = _CITE_RE.sub("", prose).strip(" \t，,、;；:：。.")
        if len(prose) >= 8 or left == 0:
            break
        left, right = extract(max(0, left - 1))
    return text[left:right].strip(" \t，,、;；:：")


def render_markdown(text: str, citations: bool = True) -> str:
    """Markdown + math + code → safe, self-contained HTML.

    `citations=True` wraps `[n]` in <span class="cite" data-n data-q>, with
    `data-q` holding the sentence the citation belongs to (used to locate the
    passage in the source article).
    """
    raw = (text or "").replace("\r\n", "\n")
    if not raw.strip():
        return ""

    stash: list[tuple[str, str]] = []

    def _stash(rendered: str) -> str:
        token = f"CLAPXSTASH{len(stash)}XEND"
        stash.append((token, rendered))
        return token

    # 1) fenced code blocks (before Markdown so their content is never parsed)
    def _fence(m: re.Match) -> str:
        return "\n\n" + _stash(_render_code(m.group(3), m.group(2))) + "\n\n"

    working = _FENCE_RE.sub(_fence, raw)

    # 2) citations — extracted BEFORE math so the sentence keeps the real TeX
    #    (the stash placeholder must never leak into the citation's data-q).
    cite_meta: list[tuple[int, int, str, str]] = []  # (start, end, nums, sentence)
    if citations:

        def _cite(m: re.Match) -> str:
            nums = m.group(1)
            sentence = _sentence_at(working, m.start(), m.end())
            # strip TeX from the locating query: the reader matches prose,
            # not \hbar/\frac tokens
            qtext = re.sub(r"\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\$[^$\n]*\$|\\\(.*?\\\)", " ", sentence)
            qtext = re.sub(r"\s+", " ", qtext).strip()
            idx = len(cite_meta)
            cite_meta.append((m.start(), m.end(), nums, qtext))
            return f"CLAPXCITE{idx}XEND"

        working = _CITE_RE.sub(_cite, working)

    # 3) display math, then inline math
    working = _DISPLAY_MATH_RE.sub(lambda m: "\n\n" + _stash(_render_math(m.group(1), True)) + "\n\n", working)
    working = _BRACKET_MATH_RE.sub(lambda m: "\n\n" + _stash(_render_math(m.group(1), True)) + "\n\n", working)
    working = _INLINE_DOLLAR_RE.sub(lambda m: _stash(_render_math(m.group(1), False)), working)
    working = _INLINE_PAREN_RE.sub(lambda m: _stash(_render_math(m.group(1), False)), working)

    md = mistune.create_markdown(
        escape=True,
        plugins=["table", "strikethrough", "url", "task_lists"],
    )
    out = md(working)

    # 4) restore placeholders
    for token, rendered in stash:
        out = out.replace(token, rendered)
    for idx, (_s, _e, nums, sentence) in enumerate(cite_meta):
        label = "[" + nums.replace(" ", "") + "]"
        out = out.replace(
            f"CLAPXCITE{idx}XEND",
            f'<span class="cite" data-n="{_esc(nums)}" data-q="{_esc(sentence)}">{_esc(label)}</span>',
        )

    # 5) link hygiene: no script-ish URLs (target=_blank is added by mistune's url plugin
    #    only for bare URLs; markdown links keep their href)
    out = _DANGEROUS_HREF.sub('href="#"', out)
    return out


def render_plain(text: str) -> str:
    """Escaped text with line breaks (used for user messages)."""
    return _esc(text or "").replace("\n", "<br>\n")
