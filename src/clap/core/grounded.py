"""Grounded answering: retrieval-first context, closed-world prompt, audit.

Design (matching the user's "有依据" methodology):

1. **Retrieval first** — every question is answered against numbered
   passages retrieved from the ZIM external library (or the local KB).
2. **Closed world** — the system prompt allows ONLY the supplied material;
   every factual sentence must carry a `[n]` citation; missing knowledge is
   answered with 「来源未提及」 instead of model invention.
3. **Sentence audit** — after generation, each sentence that cites a source
   is checked against that source's text (character-bigram overlap plus
   term recall). Sentences that cite nothing, or cite material that does
   not support them, are marked ungrounded and the score is reported.

The audit is deliberately lexical (no extra model call): it is cheap,
deterministic and testable. It cannot prove semantic entailment, so the UI
presents it as 「依据度」 (evidence coverage), not as a truth certificate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CITATION_RE = re.compile(r"\[(\d+(?:\s*[,，]\s*\d+)*)\]")
MATH_RE = re.compile(r"\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\$[^$\n]*\$|\\\(.*?\\\)")
SPLIT_RE = re.compile(r"(?<=[。！？!?；;])\s*|\n+")
LATIN_SPLIT_RE = re.compile(r"(?<=[.])\s+")
PUNCT_RE = re.compile(r"[\s\u3000，。！？、；：,.!?;:\"'“”‘’（）()\[\]【】《》<>—\-…·]+")

EMPTY_ANSWER = "来源未提及。"

# Phrases that count as an honest "the material does not say" answer. Both
# languages are accepted because the answer language follows the question.
ABSTENTION_PREFIXES = (
    "来源未提及",
    "（来源未提及",
    "(来源未提及",
    "the material does not mention",
    "the material doesn't mention",
    "not mentioned in the material",
    "no source material",
    # the same honest answer in the other interface languages
    "資料には言及されていません",
    "le document n'en parle pas",
    "в материале это не упоминается",
    "das material erwähnt das nicht",
    "el material no lo menciona",
    "o material não menciona isto",
    "il materiale non lo menziona",
    "자료에 언급되지 않았습니다",
)

SYSTEM_PROMPT = """你是 CLAP 的「有依据」答题助手。请严格遵守下面的规则：

1. 只使用下方编号材料回答，不要使用材料以外的知识。
2. 每个陈述事实的句子末尾都要标注来源编号，例如 [1] 或 [2,3]。
3. 材料没有提到答案时，直接写「来源未提及。」并说明缺少什么，不要猜测，也不要补充常识。
4. 不要编造材料里没有的数字、人名、论文标题或结论。
5. 引用原文时保持原意，可以压缩，但不要改变事实。
6. 用与用户提问相同的语言回答。

材料：
{context}
"""

MIN_SENTENCE_CHARS = 6
GROUNDING_THRESHOLD = 0.30


@dataclass
class Source:
    n: int
    zim: str
    path: str
    title: str
    text: str = ""
    score: float = 0.0
    kind: str = "zim"

    def as_dict(self, with_text: bool = False) -> dict:
        d = {
            "n": self.n,
            "kind": self.kind,
            "zim": self.zim,
            "path": self.path,
            "title": self.title,
            "score": round(self.score, 4),
        }
        if with_text:
            d["text"] = self.text
        return d


@dataclass
class SentenceAudit:
    text: str
    cited: list[int] = field(default_factory=list)
    grounded: bool = False
    verified: bool = True
    overlap: float = 0.0
    reason: str = ""
    code: str = ""

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "cited": self.cited,
            "grounded": self.grounded,
            "verified": self.verified,
            "overlap": round(self.overlap, 3),
            "reason": self.reason,
            "code": self.code,
        }


def build_context(hits, max_chars_per: int = 3500) -> tuple[list[Source], str]:
    """Turn retrieval hits into numbered sources + the prompt material block."""
    sources: list[Source] = []
    blocks: list[str] = []
    for i, h in enumerate(hits, start=1):
        text = (getattr(h, "text", "") or getattr(h, "snippet", "") or "").strip()
        if not text:
            continue
        src = Source(
            n=i,
            zim=getattr(h, "zim", getattr(h, "source", "")),
            path=getattr(h, "path", ""),
            title=getattr(h, "title", "") or (getattr(h, "path", "") or "").rsplit("/", 1)[-1],
            text=text[:max_chars_per],
            score=float(getattr(h, "score", 0.0) or 0.0),
            kind=getattr(h, "kind", "zim") or "zim",
        )
        sources.append(src)
        blocks.append(f"[{src.n}] {src.title}（{src.zim} · {src.path}）\n{src.text}")
    return sources, "\n\n".join(blocks)


def render_system_prompt(sources: list[Source]) -> str:
    _, context = build_context_from_sources(sources)
    return SYSTEM_PROMPT.format(context=context)


def build_context_from_sources(sources: list[Source], max_chars_per: int = 3500) -> tuple[list[Source], str]:
    blocks = [
        f"[{s.n}] {s.title}（{s.zim} · {s.path}）\n{s.text[:max_chars_per]}" for s in sources if s.text
    ]
    return sources, "\n\n".join(blocks)


def split_sentences(answer: str) -> list[str]:
    text = (answer or "").strip()
    if not text:
        return []
    # Formulas are not prose: drop TeX before splitting, otherwise each line
    # of a display block becomes a "sentence" that fails the lexical check.
    text = MATH_RE.sub(" ", text)
    parts: list[str] = []
    for chunk in SPLIT_RE.split(text):
        chunk = (chunk or "").strip()
        if not chunk:
            continue
        # long Latin sentences often use ". " rather than "。"
        if len(chunk) > 240 and SPLIT_RE.search(chunk) is None:
            parts.extend(p.strip() for p in LATIN_SPLIT_RE.split(chunk) if p.strip())
        else:
            parts.append(chunk)
    # A clause ending in 「：」/「；」 is a lead-in ("…关系是：") whose citation
    # arrives on the following clause; merging avoids a false "uncited" row.
    merged: list[str] = []
    buf = ""
    for part in parts:
        buf = (buf + " " + part) if buf else part
        if re.search(r"[：:；;]\s*$", buf) and len(buf) < 160:
            continue
        merged.append(buf.strip())
        buf = ""
    if buf:
        merged.append(buf.strip())
    # A citation that sits on its own line («[1]。») belongs to the sentence
    # above it, not to a new empty sentence. Citations that end up at the
    # START of the next piece («。 [1] 下一句») move back to the previous.
    out: list[str] = []
    for piece in merged:
        lead = re.match(r"^((?:\[\d+(?:\s*[,，]\s*\d+)*\]\s*)+)", piece)
        if lead and out:
            cites = lead.group(1).strip()
            rest = piece[lead.end():].strip()
            out[-1] = (out[-1] + " " + cites).strip()
            if rest:
                out.append(rest)
            continue
        bare = strip_citations(piece).strip(" \t，,。.；;：:")
        if not bare and out:
            out[-1] = (out[-1] + " " + piece).strip()
            continue
        out.append(piece)
    return out


def citations_of(sentence: str) -> list[int]:
    out: list[int] = []
    for m in CITATION_RE.finditer(sentence):
        for token in re.split(r"[,，]", m.group(1)):
            token = token.strip()
            if token.isdigit():
                n = int(token)
                if n not in out:
                    out.append(n)
    return out


def strip_citations(sentence: str) -> str:
    return CITATION_RE.sub("", sentence)


def strip_math(sentence: str) -> str:
    """Remove TeX before lexical comparison (formulas have no Chinese prose)."""
    return MATH_RE.sub(" ", sentence or "")


def _grams(text: str) -> set[str]:
    clean = PUNCT_RE.sub("", text.lower())
    if len(clean) < 2:
        return set()
    return {clean[i : i + 2] for i in range(len(clean) - 1)}


def _terms(text: str) -> set[str]:
    """Distinctive lexical tokens: CJK runs / Latin words of length >= 2."""
    out: set[str] = set()
    for run in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        # Chinese has no spaces: use 2-4 char n-grams of the run
        for n in (2, 3, 4):
            for i in range(len(run) - n + 1):
                out.add(run[i : i + n])
    for word in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", text):
        out.add(word.lower())
    return out


_LATIN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{1,}|\d+(?:\.\d+)?")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_ASCII_LETTER_RE = re.compile(r"[A-Za-z]")


def _cjk_share(text: str) -> float:
    return len(_CJK_RE.findall(text)) / max(len(text), 1) if text else 0.0


def _ascii_share(text: str) -> float:
    return len(_ASCII_LETTER_RE.findall(text)) / max(len(text), 1) if text else 0.0


def is_cross_lingual(sentence: str, source_text: str) -> bool:
    """CJK claim vs mostly-Latin source (a translated paraphrase)."""
    return _cjk_share(sentence) > 0.2 and _ascii_share(source_text) > 0.5


def latin_tokens(text: str) -> set[str]:
    return {t.lower() for t in _LATIN_TOKEN_RE.findall(text or "")}


def support_score(sentence: str, source_text: str) -> float:
    """How well does `source_text` support `sentence` (0..1, lexical).

    For cross-lingual pairs (Chinese claim, English source) character n-grams
    cannot align, so named entities / numbers / acronyms carry the check.
    """
    body = strip_citations(sentence)
    body = strip_math(body)
    grams = _grams(body)
    if not grams:
        return 0.0
    src_grams = _grams(source_text)
    gram_overlap = len(grams & src_grams) / len(grams)
    terms = _terms(body)
    term_recall = (len(terms & _terms(source_text)) / len(terms)) if terms else 0.0
    if is_cross_lingual(body, source_text):
        lat = latin_tokens(body)
        if not lat:
            return 0.0
        return len(lat & latin_tokens(source_text)) / len(lat)
    return max(gram_overlap, term_recall)


def audit(answer: str, sources: list[Source]) -> dict:
    """Per-sentence evidence check; returns the score and the sentence table.

    Three states per sentence: grounded / ungrounded / unverifiable. A claim
    translated from a mostly-Latin source without any shared entity cannot be
    checked lexically, so it is reported as unverifiable and excluded from the
    score denominator (honest coverage instead of a false accusation).
    """
    by_n = {s.n: s for s in sources}
    rows: list[SentenceAudit] = []
    counted = 0
    grounded = 0
    unverified = 0

    # A plain «the material does not mention it» answer is one honest
    # abstention, not a list of uncited sentences — check it before splitting
    # (translations may contain parentheses with full stops).
    stripped = strip_math(strip_citations(answer or "")).strip().lower()
    if stripped and any(stripped.startswith(p.lower()) for p in ABSTENTION_PREFIXES):
        row = SentenceAudit(
            text=(answer or "").strip(), grounded=True, overlap=1.0,
            reason="来源未提及", code="abstained",
        )
        return {
            "score": 1.0,
            "sentences_total": 1,
            "sentences_grounded": 1,
            "sentences_unverified": 0,
            "sentences": [row.as_dict()],
            "threshold": GROUNDING_THRESHOLD,
        }

    for sent in split_sentences(answer):
        cited = citations_of(sent)
        body = strip_citations(sent).strip()
        prose = strip_math(body).strip()
        if len(prose) < MIN_SENTENCE_CHARS:
            continue
        body_l = body.lower()
        if any(body_l.startswith(p.lower()) for p in ABSTENTION_PREFIXES):
            # Saying "the material does not mention it" is correct behaviour.
            rows.append(
                SentenceAudit(text=sent, cited=cited, grounded=True, overlap=1.0,
                              reason="来源未提及", code="abstained")
            )
            counted += 1
            grounded += 1
            continue
        if not cited:
            rows.append(SentenceAudit(text=sent, cited=[], grounded=False, reason="未标注来源", code="uncited"))
            counted += 1
            continue
        valid = [n for n in cited if n in by_n]
        if not valid:
            rows.append(
                SentenceAudit(text=sent, cited=cited, grounded=False,
                              reason="引用了不存在的编号", code="bad_number")
            )
            counted += 1
            continue
        best = 0.0
        cross = False
        for n in valid:
            score = support_score(body, by_n[n].text)
            cross = cross or is_cross_lingual(body, by_n[n].text)
            if score > best:
                best = score
        if best >= GROUNDING_THRESHOLD:
            rows.append(
                SentenceAudit(
                    text=sent, cited=cited, grounded=True, overlap=best,
                    reason="有据（跨语言，按术语核对）" if cross else "有据",
                    code="grounded_cross" if cross else "grounded",
                )
            )
            counted += 1
            grounded += 1
        elif cross and not latin_tokens(body):
            rows.append(
                SentenceAudit(
                    text=sent, cited=cited, grounded=False, verified=False,
                    overlap=best, reason="跨语言转述，无法逐字核对", code="unverified_cross",
                )
            )
            unverified += 1
        else:
            rows.append(
                SentenceAudit(
                    text=sent, cited=cited, grounded=False, overlap=best,
                    reason="引用材料不支持该句", code="unsupported",
                )
            )
            counted += 1
    score = (grounded / counted) if counted else 0.0
    return {
        "score": round(score, 4),
        "sentences_total": counted,
        "sentences_grounded": grounded,
        "sentences_unverified": unverified,
        "sentences": [r.as_dict() for r in rows],
        "threshold": GROUNDING_THRESHOLD,
    }


def cited_sources(answer: str, sources: list[Source]) -> list[dict]:
    """Only the sources the answer actually cites (for the UI panel)."""
    used = {n for sent in split_sentences(answer) for n in citations_of(sent)}
    return [s.as_dict() for s in sources if s.n in used]
