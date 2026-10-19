"""Plain-text evidence library (only text — no PDF/DOCX and friends).

Users add `.txt` / `.md` / `.markdown` / `.text` files either by uploading or
by pointing at a file/folder. Everything is copied into a managed directory so
the evidence set is stable:

    ~/.clap/text_sources/          (override: CLAP_TEXT_ROOTS, os.pathsep list)

The library is deliberately narrow: binary formats are rejected (415-style)
because they cannot be quoted and checked word for word. The grounded mode
searches ZIM archives and this library together; citations from here open the
matching paragraph in the reader.
"""

from __future__ import annotations

import hashlib
import os
import threading
from dataclasses import dataclass
from pathlib import Path

from clap.core import zim_kb as ZK

PLAIN_EXTS = (".txt", ".md", ".markdown", ".text")
MAX_BYTES = 20 * 1024 * 1024  # 20 MB per file
MAX_ADD_PATH_FILES = 200
MAX_ARTICLE_CHARS = 200_000


def roots() -> list[str]:
    """Configured text roots: CLAP_TEXT_ROOTS replaces the managed default."""
    raw = os.environ.get("CLAP_TEXT_ROOTS", "")
    if raw.strip():
        candidates = [p for p in raw.split(os.pathsep) if p.strip()]
    else:
        candidates = [str(Path.home() / ".clap" / "text_sources")]
    out: list[str] = []
    for c in candidates:
        c = os.path.abspath(os.path.expanduser(c))
        if c not in out:
            out.append(c)
    return out


def managed_root() -> Path:
    """Where uploaded files are copied (first configured root)."""
    root = Path(roots()[0])
    root.mkdir(parents=True, exist_ok=True)
    return root


def is_plain_name(name: str) -> bool:
    return Path(name).suffix.lower() in PLAIN_EXTS


def decode_text(data: bytes) -> str:
    if b"\x00" in data[:4096]:
        raise ValueError("二进制文件不能作为文本资料（只接受纯文本）")
    for enc in ("utf-8", "utf-8-sig", "gb18030", "big5"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("无法按纯文本解码（请转成 UTF-8 文本）")


@dataclass
class TextHit:
    name: str
    path: str
    title: str
    text: str = ""
    snippet: str = ""
    score: float = 0.0
    kind: str = "text"
    zim: str = "文本资料"

    def as_dict(self, with_text: bool = False) -> dict:
        d = {
            "kind": "text",
            "zim": self.zim,
            "name": self.name,
            "path": self.path,
            "title": self.title,
            "snippet": self.snippet,
            "score": round(self.score, 4),
        }
        if with_text:
            d["text"] = self.text
        return d


class TextLibrary:
    def __init__(self, text_roots: list[str] | None = None):
        self._roots = [os.path.abspath(os.path.expanduser(p)) for p in (text_roots or roots())]
        self._lock = threading.RLock()
        self._cache: dict[str, str] = {}

    @property
    def roots(self) -> list[str]:
        return list(self._roots)

    def _managed_root(self) -> Path:
        root = Path(self._roots[0])
        root.mkdir(parents=True, exist_ok=True)
        return root

    # -- discovery ------------------------------------------------------
    def list_files(self) -> list[dict]:
        out: list[dict] = []
        seen: set[str] = set()
        for root in self._roots:
            p = Path(root)
            if not p.is_dir():
                continue
            for f in sorted(p.rglob("*")):
                if not f.is_file() or not is_plain_name(f.name):
                    continue
                key = str(f.resolve())
                if key in seen:
                    continue
                seen.add(key)
                try:
                    st = f.stat()
                except OSError:
                    continue
                out.append(
                    {
                        "kind": "text",
                        "name": f.name,
                        "path": str(f),
                        "title": f.stem,
                        "size": st.st_size,
                        "size_kb": round(st.st_size / 1024, 1),
                        "mtime": int(st.st_mtime),
                    }
                )
        out.sort(key=lambda x: x["name"].lower())
        return out

    def _find(self, name: str) -> Path | None:
        name = os.path.basename(name or "")
        for root in self._roots:
            p = Path(root) / name
            if p.is_file() and is_plain_name(p.name):
                return p
        # allow lookup by exact path inside a configured root
        cand = Path(name)
        if cand.is_file() and is_plain_name(cand.name):
            resolved = str(cand.resolve())
            for root in self._roots:
                if resolved.startswith(os.path.abspath(root) + os.sep):
                    return cand
        return None

    def read(self, name: str) -> str:
        p = self._find(name)
        if p is None:
            raise KeyError(f"文本资料不存在或已被删除：{name}")
        key = str(p)
        with self._lock:
            if key in self._cache:
                return self._cache[key]
        try:
            text = decode_text(p.read_bytes())
        except ValueError:
            raise
        text = text[:MAX_ARTICLE_CHARS]
        with self._lock:
            self._cache[key] = text
        return text

    # -- adding ---------------------------------------------------------
    def add_bytes(self, filename: str, data: bytes) -> dict:
        base = os.path.basename(filename or "note.txt")
        if not is_plain_name(base):
            raise ValueError("只接受纯文本：.txt / .md / .markdown / .text")
        if len(data) > MAX_BYTES:
            raise ValueError("文件超过 20 MB")
        decode_text(data)  # validates encoding/binary up front
        root = self._managed_root()
        digest = hashlib.sha256(data).hexdigest()[:10]
        target = root / base
        stem, ext = os.path.splitext(base)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() == hashlib.sha256(data).hexdigest():
                return {"name": target.name, "path": str(target), "duplicate": True}
            target = root / f"{stem}-{digest}{ext}"
        target.write_bytes(data)
        with self._lock:
            self._cache.pop(str(target), None)
        return {"name": target.name, "path": str(target), "duplicate": False}

    def add_path(self, path: str) -> dict:
        p = Path(os.path.expanduser(path))
        if not p.exists():
            raise ValueError(f"路径不存在：{path}")
        added: list[str] = []
        skipped: list[str] = []
        if p.is_file():
            if not is_plain_name(p.name):
                raise ValueError("只接受纯文本：.txt / .md / .markdown / .text")
            added.append(self.add_bytes(p.name, p.read_bytes())["name"])
        elif p.is_dir():
            files = [f for f in sorted(p.rglob("*")) if f.is_file()]
            for f in files:
                if len(added) >= MAX_ADD_PATH_FILES:
                    skipped.append("…")
                    break
                if not is_plain_name(f.name):
                    skipped.append(f.name)
                    continue
                try:
                    added.append(self.add_bytes(f.name, f.read_bytes())["name"])
                except ValueError:
                    skipped.append(f.name)
        else:
            raise ValueError("路径既不是文件也不是目录")
        return {"added": added, "skipped": skipped, "count": len(added)}

    def delete(self, name: str) -> bool:
        p = self._find(name)
        if p is None:
            return False
        try:
            p.unlink()
        except OSError:
            return False
        with self._lock:
            self._cache.pop(str(p), None)
        return True

    # -- search ---------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> list[TextHit]:
        query = (query or "").strip()
        if not query:
            return []
        terms = [
            t for c in ZK._query_candidates(query) for t in c.split()
            if len(t) >= 2 and not any(ch in t for ch in "\\{}$^_=<>")
        ]
        uniq = list(dict.fromkeys(terms))
        if not uniq:
            return []
        ql = query.lower()
        scored: list[tuple[float, dict]] = []
        for f in self.list_files():
            try:
                text = self.read(f["name"])
            except Exception:
                continue
            low = text.lower()
            score = 0.0
            for term in uniq:
                count = low.count(term.lower())
                if count:
                    score += count * max(1.0, len(term) / 2.0) ** 2
            if not score:
                continue
            if ql in f["title"].lower():
                score += 40.0
            scored.append((score, f))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: list[TextHit] = []
        for score, f in scored[:top_k]:
            text = self.read(f["name"])
            out.append(
                TextHit(
                    name=f["name"],
                    path=f["path"],
                    title=f["title"],
                    text=text,
                    snippet=ZK._snippet(text, query),
                    score=score,
                )
            )
        return out

    def article(self, name: str, query: str = "") -> dict:
        text = self.read(name)
        p = self._find(name)
        return {
            "kind": "text",
            "name": p.name if p else name,
            "path": str(p) if p else name,
            "title": p.stem if p else name,
            "text": text,
            "chars": len(text),
            "match": ZK.locate_passage(text, query) if query else None,
        }


_LIBRARY: TextLibrary | None = None
_LOCK = threading.Lock()


def library() -> TextLibrary:
    global _LIBRARY
    with _LOCK:
        if _LIBRARY is None:
            _LIBRARY = TextLibrary()
        return _LIBRARY


def reset_library() -> None:
    global _LIBRARY
    with _LOCK:
        _LIBRARY = None
