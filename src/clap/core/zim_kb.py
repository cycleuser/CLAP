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
        for candidate in _query_candidates(query):
            got = self._search_one(candidate, infos, max(top_k, 5))
            for h in got:
                key = (h.zim, h.path)
                if key in merged:
                    merged[key].score += h.score
                else:
                    merged[key] = h
            if got and candidate == _normalise(query):
                break  # the full phrase hit: no need for the fallbacks
            if len(merged) >= top_k * 2:
                break

        hits = sorted(merged.values(), key=lambda h: h.score, reverse=True)
        ql = _normalise(query).lower()
        for h in hits:
            if ql and ql in h.title.lower():
                h.score += 50.0
        hits.sort(key=lambda h: h.score, reverse=True)

        out: list[ZimHit] = []
        for h in hits:
            try:
                text = self.article(h.zim, h.path, max_chars=MAX_ARTICLE_CHARS)["text"]
                h.snippet = _snippet(text, query)
                h.text = text
            except Exception:
                h.snippet = ""
            out.append(h)
            if len(out) >= top_k:
                break
        return out

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
    "是", "的", "了", "和", "与", "它", "这", "那", "在", "有", "吗", "呢", "吧",
    "我", "你", "他", "她", "请", "讲", "说", "问", "关于",
)
_STOP_RE = re.compile("|".join(sorted((re.escape(w) for w in _QUERY_STOPWORDS), key=len, reverse=True)))


def _normalise(query: str) -> str:
    """Strip punctuation and collapse whitespace (index-safe phrase form)."""
    return re.sub(r"\s+", " ", _PUNCT_QUERY_RE.sub(" ", query)).strip()


def _query_candidates(query: str) -> list[str]:
    """Search phrases in priority order: full question, then content terms."""
    base = _normalise(query)
    if not base:
        return []
    candidates = [base]
    terms: list[str] = []
    for token in base.split(" "):
        cleaned = _STOP_RE.sub(" ", token)
        for part in cleaned.split():
            part = part.strip()
            # CJK runs shorter than 2 chars and Latin words shorter than 2 are noise
            if len(part) >= 2 and part not in terms:
                terms.append(part)
    # a long CJK run keeps its leading np-gram as an extra candidate
    for token in base.split(" "):
        if re.fullmatch(r"[\u4e00-\u9fff]{6,}", token):
            head = _STOP_RE.sub(" ", token).strip().split(" ")[0] if _STOP_RE.sub(" ", token).strip() else ""
            if len(head) >= 2 and head not in terms:
                terms.insert(0, head)
    for t in terms[:4]:
        if t != base and t not in candidates:
            candidates.append(t)
    return candidates


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
