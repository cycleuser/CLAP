"""ZIM external knowledge base (read-only, offline, zero-copy).

A `.zim` archive is a compressed, pre-indexed collection (Kiwix format).
Rather than ingesting its articles into our own vector store, CLAP queries
the archive's built-in full-text index directly. That gives us:

- no network access at query time (works fully offline),
- no duplicated storage (a 1.3 GB AAAI bundle stays one file),
- the archive remains the single source of truth (immutable sources).

Roots: `CLAP_ZIM_ROOTS` (os.pathsep separated) or `~/Documents/重点文献`.

Notes
-----
* `libzim` prints "No stemming for language 'zh'" for Chinese archives; that
  is informational, not a failure.
* Article content is HTML; `html_to_text` strips tags while keeping block
  boundaries so retrieved passages stay readable for the model.
"""

from __future__ import annotations

import html
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ROOTS = [str(Path("~").expanduser() / "Documents" / "重点文献")]
MAX_ARTICLE_CHARS = 60_000  # hard cap per article handed to callers
_ARCHIVE_CACHE_SIZE = 8


def roots() -> list[str]:
    """Configured ZIM root directories (existing ones only)."""
    raw = os.environ.get("CLAP_ZIM_ROOTS", "")
    candidates = [p for p in raw.split(os.pathsep) if p.strip()] if raw.strip() else []
    candidates += DEFAULT_ROOTS
    out: list[str] = []
    for c in candidates:
        c = os.path.abspath(os.path.expanduser(c))
        if os.path.isdir(c) and c not in out:
            out.append(c)
    return out


def _libzim():
    try:
        from libzim.reader import Archive  # noqa: F401
        from libzim.search import Query, Searcher  # noqa: F401
    except Exception as exc:  # pragma: no cover - env dependent
        raise RuntimeError("libzim 未安装：pip install libzim") from exc
    return "ok"


@dataclass
class ZimInfo:
    name: str
    path: str
    title: str
    language: str
    entries: int
    size_bytes: int
    mtime: float
    error: str = ""

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "title": self.title,
            "language": self.language,
            "entries": self.entries,
            "size_mb": round(self.size_bytes / 1048576, 1),
            "mtime": int(self.mtime),
            "error": self.error,
        }


@dataclass
class ZimHit:
    zim: str
    path: str
    title: str
    snippet: str = ""
    score: float = 1.0
    tf: float = 0.0
    estimated: int = 0
    text: str = field(default="", repr=False)

    def as_dict(self, with_text: bool = False) -> dict:
        d = {
            "zim": self.zim,
            "path": self.path,
            "title": self.title,
            "snippet": self.snippet,
            "score": round(self.score, 4),
        }
        if with_text:
            d["text"] = self.text
        return d


class ZimLibrary:
    """Thread-safe facade over the configured `.zim` archives."""

    def __init__(self, zim_roots: list[str] | None = None):
        self._roots = [os.path.abspath(os.path.expanduser(p)) for p in (zim_roots or roots())]
        self._lock = threading.RLock()
        self._archives: OrderedDict[str, object] = OrderedDict()
        self._infos: dict[str, ZimInfo] | None = None
        self._text_cache: OrderedDict[tuple[str, str], str] = OrderedDict()

    @property
    def roots(self) -> list[str]:
        return list(self._roots)

    # -- discovery ------------------------------------------------------
    def list_zims(self) -> list[ZimInfo]:
        with self._lock:
            if self._infos is not None:
                return list(self._infos.values())
            infos: dict[str, ZimInfo] = {}
            for root in self._roots:
                try:
                    names = sorted(os.listdir(root))
                except OSError:
                    continue
                for name in names:
                    if not name.lower().endswith(".zim"):
                        continue
                    path = os.path.join(root, name)
                    try:
                        st = os.stat(path)
                        a = self._archive(path)
                        info = ZimInfo(
                            name=name,
                            path=path,
                            title=_decode(a.get_metadata("Title")),
                            language=_decode(a.get_metadata("Language")),
                            entries=int(a.entry_count),
                            size_bytes=st.st_size,
                            mtime=st.st_mtime,
                        )
                    except Exception as exc:  # unreadable archive: surfaced, not fatal
                        info = ZimInfo(
                            name=name,
                            path=path,
                            title=name[:-4],
                            language="",
                            entries=0,
                            size_bytes=os.path.getsize(path) if os.path.exists(path) else 0,
                            mtime=os.path.getmtime(path) if os.path.exists(path) else 0.0,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    infos[name] = info
            self._infos = infos
            return list(infos.values())

    def names(self) -> list[str]:
        return [z.name for z in self.list_zims()]

    def _archive(self, path: str):
        _libzim()
        from libzim.reader import Archive

        with self._lock:
            if path in self._archives:
                self._archives.move_to_end(path)
                return self._archives[path]
            arch = Archive(path)
            self._archives[path] = arch
            while len(self._archives) > _ARCHIVE_CACHE_SIZE:
                self._archives.popitem(last=False)
            return arch

    def _archive_by_name(self, name: str):
        for z in self.list_zims():
            if z.name == name or name in (z.name[:-4], z.title):
                return z, self._archive(z.path)
        raise KeyError(f"未找到 ZIM：{name}")

    # -- search ---------------------------------------------------------
    def search(self, query: str, zim_names: list[str] | None = None, top_k: int = 10) -> list[ZimHit]:
        """Cross-archive search, merged by normalised relevance.

        Questions are not queries: Xapian's index chokes on punctuation and
        question words («量子力学研究什么？它和波函数是什么关系？» finds
        nothing, «波函数» finds three chapters). We therefore search the full
        normalised string first and fall back to its content terms, merging
        the result sets.
        """
        query = (query or "").strip()
        if not query:
            return []
        wanted = set(zim_names or [])
        infos = [
            z for z in self.list_zims() if not z.error and (not wanted or z.name in wanted or z.name[:-4] in wanted)
        ]
        if not infos:
            return []

        merged: dict[tuple[str, str], ZimHit] = {}
        candidates = _query_candidates(query)
        for candidate in candidates:
            got = self._search_one(candidate, infos, max(top_k, 5))
            # longer terms are more specific: “薛定谔方程” should outrank “公式”
            weight = max(1.0, len(candidate) / 2.0) ** 2.0
            for h in got:
                key = (h.zim, h.path)
                if key in merged:
                    merged[key].score += h.score * weight
                else:
                    h.score *= weight
                    merged[key] = h
        if not merged:
            # Questions built from long phrases can miss the index entirely;
            # distinctive character bigrams still find the right chapter.
            for term in _bigram_fallback(query):
                for h in self._search_one(term, infos, 3):
                    key = (h.zim, h.path)
                    if key in merged:
                        merged[key].score += h.score * 0.5
                    else:
                        h.score *= 0.5
                        merged[key] = h
                if len(merged) >= top_k:
                    break

        hits = sorted(merged.values(), key=lambda h: h.score, reverse=True)
        # re-rank by how often the question's content words actually appear in
        # each article (an index estimate counts a hit once; a real chapter
        # repeats its key term many times)
        terms = [
            t
            for c in candidates
            for t in c.split()
            if len(t) >= 2 and not any(ch in t for ch in "\\{}$^_=<>")
        ]
        uniq_terms = list(dict.fromkeys(terms))
        for h in hits[:20]:
            text = self._text(h.zim, h.path).lower()
            tf = 0.0
            for term in uniq_terms:
                # cap per-term occurrences: a generic word repeated 100× must
                # not drown a precise term (Schrödinger vs. equation)
                count = min(text.count(term.lower()), 8)
                if count:
                    tf += count * max(1.0, len(term) / 2.0) ** 2
            h.tf = tf
        hits.sort(key=lambda h: (getattr(h, "tf", 0.0), h.score), reverse=True)

        ql = _normalise(query).lower()
        for h in hits:
            if ql and ql in h.title.lower():
                h.score += 50.0
        hits.sort(key=lambda h: (getattr(h, "tf", 0.0), h.score), reverse=True)

        out: list[ZimHit] = []
        for h in hits:
            try:
                text = self._text(h.zim, h.path)
                h.snippet = _snippet(text, query)
                h.text = text[:MAX_ARTICLE_CHARS]
            except Exception:
                h.snippet = ""
            out.append(h)
            if len(out) >= top_k:
                break
        return out

    def _text(self, zim_name: str, entry_path: str) -> str:
        """Article text with a small LRU cache (search + snippets + reader)."""
        key = (zim_name, entry_path)
        with self._lock:
            if key in self._text_cache:
                self._text_cache.move_to_end(key)
                return self._text_cache[key]
        text = self.article(zim_name, entry_path)["text"]
        with self._lock:
            self._text_cache[key] = text
            while len(self._text_cache) > 64:
                self._text_cache.popitem(last=False)
        return text

    def _search_one(self, query: str, infos: list[ZimInfo], per_archive: int) -> list[ZimHit]:
        from libzim.search import Query, Searcher

        hits: list[ZimHit] = []
        for info in infos:
            try:
                arch = self._archive(info.path)
                result = Searcher(arch).search(Query().set_query(query))
                estimated = int(result.getEstimatedMatches() or 0)
                for path in result.getResults(0, per_archive):
                    hits.append(
                        ZimHit(
                            zim=info.name,
                            path=path,
                            title=_title_of(arch, path),
                            score=float(estimated),
                            estimated=estimated,
                        )
                    )
            except Exception:
                continue
        return hits

    # -- article --------------------------------------------------------
    def article(self, zim_name: str, entry_path: str, max_chars: int = MAX_ARTICLE_CHARS) -> dict:
        info, arch = self._archive_by_name(zim_name)
        try:
            entry = arch.get_entry_by_path(entry_path)
        except Exception:
            # some bundles address articles with a leading slash
            entry = arch.get_entry_by_path(entry_path.lstrip("/"))
        item = entry.get_item()
        raw = item.content.tobytes()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "ignore")
        is_html = entry_path.lower().endswith((".html", ".htm")) or "<html" in raw[:400].lower()
        text = html_to_text(raw) if is_html else raw
        return {
            "zim": info.name,
            "path": entry_path,
            "title": _title_of(arch, entry_path),
            "text": text[:max_chars],
            "chars": len(text),
            "is_html": is_html,
        }

    def suggestions(self, prefix: str, zim_name: str = "", limit: int = 8) -> list[str]:
        prefix = (prefix or "").strip()
        if not prefix:
            return []
        from libzim.suggestion import SuggestionSearcher

        out: list[str] = []
        infos = self.list_zims()
        if zim_name:
            infos = [z for z in infos if z.name == zim_name or z.name[:-4] == zim_name]
        for info in infos:
            if info.error:
                continue
            try:
                arch = self._archive(info.path)
                res = SuggestionSearcher(arch).suggest(prefix, limit)
                out.extend(list(res.getResults(0, limit)))
            except Exception:
                continue
            if len(out) >= limit:
                break
        return out[:limit]


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", "ignore")
    return str(value or "")


def _title_of(arch, entry_path: str) -> str:
    try:
        entry = arch.get_entry_by_path(entry_path)
        return _decode(entry.title) or entry_path.rsplit("/", 1)[-1]
    except Exception:
        return entry_path.rsplit("/", 1)[-1]


_BLOCK_RE = re.compile(r"</?(p|div|br|li|ul|ol|h[1-6]|tr|table|section|article)\b[^>]*>", re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style)\b.*?</\1>", re.I | re.S)
_WS_RE = re.compile(r"[ \t\u00a0]+")
_NL_RE = re.compile(r"\n{3,}")


def html_to_text(raw: str) -> str:
    """HTML → readable text keeping block boundaries (no external parser)."""
    body = _SCRIPT_RE.sub(" ", raw)
    body = _BLOCK_RE.sub("\n", body)
    body = _TAG_RE.sub("", body)
    body = html.unescape(body)
    body = _WS_RE.sub(" ", body)
    body = "\n".join(line.strip() for line in body.splitlines())
    body = _NL_RE.sub("\n\n", body)
    return body.strip()


_PUNCT_QUERY_RE = re.compile(r"[？?！!。，,；;：:、（）()\[\]【】《》<>\"“”‘’…\n\r\t]+")
_QUERY_STOPWORDS = (
    "研究", "什么", "关系", "如何", "怎样", "为什么", "请问", "介绍", "一下", "以及",
    "它们", "我们", "你们", "他们", "这个", "那个", "哪些", "是否", "可以", "能够",
    "说明", "写出", "列出", "给出", "计算", "解释", "描述", "总结", "概括", "举例",
    "比较", "分析", "回答", "帮我", "看看", "用于", "使用", "基于", "关于",
    "代表", "表示", "意味", "一个", "一些", "这个", "那个", "并", "给", "用", "写",
    "是", "的", "了", "和", "与", "它", "这", "那", "在", "有", "吗", "呢", "吧",
    "我", "你", "他", "她", "请", "讲", "说", "问", "短", "长",
)
_STOP_RE = re.compile("|".join(sorted((re.escape(w) for w in _QUERY_STOPWORDS), key=len, reverse=True)))

# English function words are dropped at the token level (never as substrings:
# stripping "in" inside a word would corrupt it).
_EN_STOP = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so", "to", "of",
    "in", "on", "at", "by", "for", "from", "with", "without", "about", "into",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "what", "which", "who", "whom", "whose", "when", "where", "why", "how",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "you", "your", "he", "she", "his", "her", "i", "me", "my",
    "please", "explain", "describe", "tell", "give", "show", "answer",
    "mean", "means", "meaning",
}


def _normalise(query: str) -> str:
    """Strip punctuation and collapse whitespace (index-safe phrase form)."""
    return re.sub(r"\s+", " ", _PUNCT_QUERY_RE.sub(" ", query)).strip()


def _query_candidates(query: str) -> list[str]:
    """Search phrases in priority order: a short phrase, then content terms."""
    base = _normalise(query)
    if not base:
        return []
    tokens = base.split(" ")
    terms: list[str] = []
    for token in tokens:
        # mixed scripts («CAR-Transformerは何を…») must be split: the Latin
        # model name is the searchable part, the CJK run is not glued to it
        for latin in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-z0-9À-ÖØ-öø-ÿ_.+-]*", token):
            if latin.lower() not in _EN_STOP and len(latin) >= 2 and latin not in terms:
                terms.append(latin)
        for run in re.findall(r"[\u4e00-\u9fff]+", token):
            cleaned = _STOP_RE.sub(" ", run)
            for part in cleaned.split():
                part = part.strip()
                if len(part) >= 2 and part not in terms:
                    terms.append(part)
    # a long natural-language question is not a phrase the index can match;
    # searching it whole would let common words pull unrelated documents
    latin_only = all(
        re.search(r"[^\W\d_]", t, re.UNICODE) and not re.search(r"[\u4e00-\u9fff]", t)
        for t in tokens
    )
    short_phrase = (" " not in base and len(base) <= 24) or (
        latin_only and len(tokens) <= 3 and len(base) <= 40
    )
    candidates = ([base] if short_phrase else []) + terms[:6]
    return candidates


def _bigram_fallback(query: str, limit: int = 6) -> list[str]:
    """Last-resort terms for questions whose phrases match nothing.

    Chinese archives are indexed in character n-grams, so a distinctive
    bigram («方程» from «薛定谔方程») still finds the right chapter.
    """
    base = _normalise(query)
    out: list[str] = []
    for run in re.findall(r"[\u4e00-\u9fff]{3,}", base):
        for i in range(len(run) - 1):
            gram = run[i : i + 2]
            if gram in _STOPWORD_BIGRAMS or gram in out:
                continue
            out.append(gram)
            if len(out) >= limit * 2:
                break
    # prefer longer words first (already ordered by position); cap the fan-out
    return out[:limit]


_STOPWORD_BIGRAMS = {"什么", "怎么", "为什", "一个", "可以", "我们", "你们", "这个", "那个", "以及", "是否", "能够"}


def _snippet(text: str, query: str, radius: int = 80) -> str:
    """Context window around the first query term occurrence."""
    if not text:
        return ""
    low = text.lower()
    terms: list[str] = []
    for cand in _query_candidates(query):
        terms.extend(t for t in cand.split() if t)
    pos = -1
    for t in sorted(set(terms), key=len, reverse=True):
        pos = low.find(t.lower())
        if pos >= 0:
            break
    if pos < 0:
        pos = 0
    start = max(0, pos - radius)
    end = min(len(text), pos + radius)
    return ("…" if start else "") + text[start:end].replace("\n", " ") + ("…" if end < len(text) else "")


def locate_passage(text: str, query: str, max_span: int = 1400) -> dict | None:
    """Find the paragraph of `text` that best matches `query`.

    Used by the citation panel: clicking a source opens the original article
    at the passage that actually talks about the cited claim, instead of
    dumping the reader at the top of a 10k-character chapter.
    """
    text = text or ""
    query = (query or "").strip()
    if not text or not query:
        return None

    terms: list[str] = []
    for cand in _query_candidates(query):
        for token in cand.split():
            token = token.strip()
            if len(token) < 2 or token in terms:
                continue
            # skip TeX / code-like fragments: they never appear in prose
            if any(ch in token for ch in "\\{}$^_=<>"):
                continue
            terms.append(token)
    if not terms:
        return None

    # paragraphs = blank-line separated blocks (fallback: the whole text)
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in re.finditer(r"\n\s*\n", text):
        if m.start() > pos:
            spans.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    if not spans:
        spans = [(0, len(text))]

    low = text.lower()
    best: tuple[float, int, int] | None = None
    scored: list[tuple[float, int, int]] = []
    for start, end in spans:
        block = low[start:end]
        score = 0.0
        for term in terms:
            hits = block.count(term.lower())
            if hits:
                score += hits * min(len(term), 6)
        if score:
            scored.append((score, start, end))
    if scored:
        top = max(s[0] for s in scored)
        # prefer the earliest paragraph that is still strongly relevant:
        # citations open at the abstract/intro, not deep inside a methods
        # section that merely repeats the term most often
        threshold = top * 0.5
        for score, start, end in scored:
            if score >= threshold:
                best = (score, start, end)
                break

    if best is None:
        first = -1
        for term in sorted(terms, key=len, reverse=True):
            first = low.find(term.lower())
            if first >= 0:
                break
        if first < 0:
            return None
        start = max(0, first - 200)
        end = min(len(text), first + 600)
        return {"start": start, "end": end}

    _, start, end = best
    if end - start > max_span:
        # trim long paragraphs around the best-matching term
        block = low[start:end]
        centre = start
        for term in sorted(terms, key=len, reverse=True):
            idx = block.find(term.lower())
            if idx >= 0:
                centre = start + idx
                break
        start = max(start, centre - max_span // 3)
        end = min(end, start + max_span)
    return {"start": start, "end": end}


_LIBRARY: ZimLibrary | None = None
_LIBRARY_LOCK = threading.Lock()


def library() -> ZimLibrary:
    """Process-wide singleton (the archive cache must be shared)."""
    global _LIBRARY
    with _LIBRARY_LOCK:
        if _LIBRARY is None:
            _LIBRARY = ZimLibrary()
        return _LIBRARY


def reset_library() -> None:
    global _LIBRARY
    with _LIBRARY_LOCK:
        _LIBRARY = None
