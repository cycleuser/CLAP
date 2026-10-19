"""CLAP session core: `.clap` file format v2, v1 migration, validation, tree ops.

Implements the CLAP-NEXT session format (spec-session-features.md §0):

- v2 = node tree. `messages` is a flat array; the tree is expressed by
  `id` / `parent` links (diff-able, merge-able). Each message carries
  provenance metadata: provider, model, latency_ms, token counts, error.
- `manifest` is the evidence chain: how this session was produced
  (chat/embed model, provider, document fingerprint, CLAP version,
  annotations schema, privacy and routing policy, purpose, event history).
- `branches` name linearized paths through the tree. Each branch has
  `{id, name, root, fork_at, model, created}` where
    * `fork_at`  = message the branch was forked from (None for the main line),
    * `root`     = first *own* message of the branch (None until appended),
    * `model`    = optional model override for that branch (what-if forks),
    * `created`  = ISO timestamp of branch creation.
- `annotations` are structured teacher annotations (F5) attached to message
  ids; `message["annotations"]` mirrors the annotation ids.

Backward compatibility: opening a v1 file (flat `{version: "1.0",
timestamp, model, embed_model, document, messages, language}`) migrates it
to v2 on load, losslessly, with no semantic change.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

SESSION_FORMAT_VERSION = "2.0"
ANNOTATIONS_SCHEMA = "2026.10"
CLAP_VERSION = "2.0.0"

ROLES = ("user", "assistant", "system")

# F5 annotation taxonomy: stable machine names + Chinese UI labels.
ANNOTATION_KINDS = (
    "keypoint",
    "question",
    "generated",
    "reviewed",
    "error_fix",
    "knowledge_point",
)
KIND_LABELS = {
    "keypoint": "重点",
    "question": "疑问",
    "generated": "生成",
    "reviewed": "讲评",
    "error_fix": "纠错",
    "knowledge_point": "知识点",
}

# F5 teaching-purpose presets chosen when a session is created.
TEACHING_PURPOSES = ("备课", "答疑", "作业讲评", "文献研讨")

# Annotation CSV columns (spec F5; `branch` is a documented superset column).
ANNOTATION_CSV_FIELDS = (
    "session_id",
    "turn_id",
    "kind",
    "tags",
    "note",
    "author",
    "ts",
    "model",
    "latency_ms",
    "tokens_out",
    "branch",
)


class SessionError(Exception):
    """Raised on malformed `.clap` payloads."""


class SessionValidationError(SessionError):
    """Raised when a v2 session fails structural validation."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def now_ms() -> int:
    """Wall-clock epoch milliseconds (message/branch timestamps)."""
    return int(time.time() * 1000)


def now_iso() -> str:
    """ISO-8601 local timestamp (manifest/branch creation)."""
    return datetime.now().isoformat(timespec="seconds")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(obj) -> str:
    """Deterministic JSON used for digests (sorted keys, no spaces)."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def session_digest(session: dict) -> str:
    """sha256 over the canonical JSON of the whole session dict."""
    return _sha256(canonical_json(session).encode("utf-8"))


def sha256_strings(**fields) -> str:
    """Combined digest for share manifests (per-field fingerprints)."""
    return _sha256(canonical_json(fields).encode("utf-8"))


def document_meta(document_path: str) -> dict:
    """Fingerprint a document path for the manifest evidence chain.

    `path_hash` is sha256 of the file bytes when the file exists on disk,
    otherwise sha256 of the path string itself (e.g. after the file moved);
    `bytes` is the on-disk size or 0 when unknown. `path` keeps the full
    local path only when the file exists (local convenience for re-opening;
    the sharing pack strips it before it travels — see sharing.py).
    """
    p = Path(document_path or "")
    if document_path and p.exists():
        digest = _sha256(p.read_bytes())
        meta = {"path_hash": digest, "name": p.name, "bytes": p.stat().st_size, "path": str(p)}
    else:
        digest = _sha256((document_path or "").encode("utf-8"))
        meta = {"path_hash": digest, "name": p.name if document_path else "", "bytes": 0}
    return meta


def _next_id(session: dict, prefix: str = "m") -> str:
    """Short sequential id unique within the session (`m1`, `m2`, ...)."""
    used: set[str] = set()
    for m in session.get("messages", []):
        i = m.get("id", "")
        if isinstance(i, str) and i.startswith(prefix):
            used.add(i[len(prefix):])
    if prefix == "b":
        for b_item in session.get("branches", []):
            i = b_item.get("id", "")
            if isinstance(i, str) and i.startswith("b"):
                used.add(i[1:])
    n = 1
    while str(n) in used:
        n += 1
    return f"{prefix}{n}"


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def new_message(
    role: str,
    content: str,
    parent: str | None = None,
    model: str | None = None,
    provider: str | None = None,
    latency_ms: int | None = None,
    tokens: dict | None = None,
    error: str | None = None,
    ts: int | None = None,
    **extra,
) -> dict:
    """Create a v2 message node. `id` is assigned when appended to a session.

    `extra` carries optional provenance extensions used by the web layer
    (retrieval `sources`, `audit` table, `grounded` flag, `zims` filter).
    Unknown keys are preserved so exports can quote them verbatim.
    """
    if role not in ROLES:
        raise SessionError(f"unknown role {role!r}; expected one of {ROLES}")
    tok = tokens if tokens is not None else {"in": None, "out": None}
    msg = {
        "id": None,  # filled by append helpers
        "parent": parent,
        "role": role,
        "content": content,
        "ts": ts if ts is not None else now_ms(),
        "provider": provider,
        "model": model,
        "latency_ms": latency_ms,
        "tokens": tok,
        "error": error,
        "annotations": [],
    }
    for key, value in extra.items():
        if value is not None:
            msg[key] = value
    return msg


def new_session(
    chat_model: str = "",
    embed_model: str = "",
    provider: str = "ollama-local",
    document: str = "",
    language: str = "zh",
    purpose: str = "备课",
    privacy_mode: bool = True,
    routing: str = "manual",
) -> dict:
    """Build an empty v2 session. `purpose` is one of TEACHING_PURPOSES (F5)."""
    if purpose not in TEACHING_PURPOSES:
        purpose = TEACHING_PURPOSES[0]
    return {
        "version": SESSION_FORMAT_VERSION,
        "created": now_iso(),
        "manifest": {
            "session_id": uuid.uuid4().hex[:12],
            "chat_model": chat_model,
            "embed_model": embed_model,
            "provider": provider,
            "purpose": purpose,
            "document": document_meta(document),
            "clap_version": CLAP_VERSION,
            "annotations_schema": ANNOTATIONS_SCHEMA,
            "privacy_mode": privacy_mode,
            "routing": routing,
            "history": [_event("create", detail={"provider": provider, "purpose": purpose})],
        },
        "messages": [],
        "branches": [],
        "annotations": [],
        "language": language,
    }


def _event(event: str, detail: dict | None = None) -> dict:
    ev = {"event": event, "ts": now_ms()}
    if detail:
        ev["detail"] = detail
    return ev


# --------------------------------------------------------------------------
# io / migration
# --------------------------------------------------------------------------


def load_session(path) -> dict:
    """Load a `.clap` file; v1 payloads are migrated to v2 transparently."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return normalize(data)


def save_session(session: dict, path, indent: int = 2) -> str:
    """Persist a v2 session as UTF-8 JSON; returns the written path."""
    assert_valid(session)
    p = Path(path)
    if p.parent and str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(session, f, ensure_ascii=False, indent=indent)
        f.write("\n")
    return str(p)


def normalize(data) -> dict:
    """Accept v1 or v2 dicts and return a validated v2 session."""
    if not isinstance(data, dict):
        raise SessionError("session root must be a JSON object")
    version = str(data.get("version", "1.0"))
    if version == SESSION_FORMAT_VERSION:
        if "messages" not in data:
            raise SessionError("v2 session missing 'messages'")
        return data
    return migrate_v1(data)


def migrate_v1(data: dict) -> dict:
    """Migrate a v1 flat session to the v2 node-tree format, losslessly.

    - messages become a parent chain in original order with synthetic ids;
    - per-message provenance is unknown in v1 (`provider`/`model`/tokens are
      null; `model` is recorded once in the manifest — never invented);
    - `ts` stays null (v1 stored no per-message timestamps);
    - document path is fingerprinted by `document_meta`.
    """
    if not isinstance(data, dict):
        raise SessionError("v1 session root must be a JSON object")
    legacy_msgs = data.get("messages") or []
    if not isinstance(legacy_msgs, list):
        raise SessionError("v1 'messages' must be a list")

    session = new_session(
        chat_model=str(data.get("model") or ""),
        embed_model=str(data.get("embed_model") or ""),
        provider="ollama-local",
        document=str(data.get("document") or ""),
        language=str(data.get("language") or "zh"),
    )
    session["created"] = str(data.get("timestamp") or session["created"])

    prev: str | None = None
    for raw in legacy_msgs:
        msg = new_message(
            role=str(raw.get("role", "user")),
            content=str(raw.get("content", "")),
            parent=prev,
        )
        msg["ts"] = None  # unknown in v1; never fabricated
        # carry already-known provenance (live-chat meta) if present; v1 did
        # not record it, so absent fields stay null — nothing invented.
        for src, dst in (
            ("provider", "provider"),
            ("model", "model"),
            ("latency_ms", "latency_ms"),
            ("tokens", "tokens"),
            ("error", "error"),
        ):
            if src in raw and raw[src] is not None:
                msg[dst] = raw[src]
        msg["id"] = _next_id(session, "m")
        session["messages"].append(msg)
        prev = msg["id"]

    if session["messages"]:
        root_id = session["messages"][0]["id"]
        session["branches"] = [
            {
                "id": "b1",
                "name": "主线",
                "root": root_id,
                "fork_at": None,
                "model": None,
                "created": session["created"],
            }
        ]

    first_msg_preview = legacy_msgs[0].get("content", "")[:50] if legacy_msgs else ""
    session["manifest"]["history"] = [
        _event(
            "migrate_v1",
            detail={
                "from_version": "1.0",
                "messages": len(legacy_msgs),
                "document": first_msg_preview and session["manifest"]["document"]["name"],
            },
        )
    ]
    return session


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def validate_session(session: dict) -> list[str]:
    """Return a list of structural issues (empty list = valid)."""
    issues: list[str] = []
    if session.get("version") != SESSION_FORMAT_VERSION:
        issues.append(f"version must be {SESSION_FORMAT_VERSION!r}")
    if not isinstance(session.get("messages"), list):
        issues.append("messages must be a list")
        return issues

    seen: set[str] = set()
    for m in session["messages"]:
        mid = m.get("id")
        if not mid:
            issues.append("message without id")
        elif mid in seen:
            issues.append(f"duplicate message id {mid!r}")
        seen.add(mid)
    for m in session["messages"]:
        parent = m.get("parent")
        if parent is not None and parent not in seen:
            issues.append(f"message {m.get('id')!r} has unknown parent {parent!r}")
        if m.get("role") not in ROLES:
            issues.append(f"message {m.get('id')!r} bad role {m.get('role')!r}")
        tok = m.get("tokens")
        if tok is not None and not isinstance(tok, dict):
            issues.append(f"message {m.get('id')!r} tokens must be object|null")

    for b in session.get("branches", []):
        for key in ("root", "fork_at"):
            ref = b.get(key)
            if ref is not None and ref not in seen:
                issues.append(f"branch {b.get('id')!r} {key} {ref!r} not in messages")
    for a in session.get("annotations", []):
        target = a.get("target")
        if target not in seen:
            issues.append(f"annotation {a.get('id')!r} target {target!r} not in messages")
        if a.get("kind") not in ANNOTATION_KINDS:
            issues.append(f"annotation {a.get('id')!r} bad kind {a.get('kind')!r}")
    man = session.get("manifest")
    if not isinstance(man, dict) or not man.get("session_id"):
        issues.append("manifest.session_id missing")
    return issues


def assert_valid(session: dict) -> None:
    issues = validate_session(session)
    if issues:
        raise SessionValidationError("; ".join(issues))


# --------------------------------------------------------------------------
# tree operations
# --------------------------------------------------------------------------


def message(session: dict, msg_id: str) -> dict | None:
    for m in session.get("messages", []):
        if m.get("id") == msg_id:
            return m
    return None


def children_map(session: dict) -> dict[str, list[str]]:
    """parent id -> child ids, in append (creation) order."""
    out: dict[str, list[str]] = {}
    for m in session.get("messages", []):
        out.setdefault(m.get("parent"), []).append(m["id"])
    return out


def roots(session: dict) -> list[str]:
    return [m["id"] for m in session.get("messages", []) if m.get("parent") is None]


def main_branch(session: dict) -> dict:
    for b in session.get("branches", []):
        if b.get("fork_at") is None:
            return b
    return {"id": "b1", "name": "主线", "root": None, "fork_at": None, "model": None}


def get_branch(session: dict, branch_id: str) -> dict | None:
    for b in session.get("branches", []):
        if b.get("id") == branch_id:
            return b
    return None


def branch_root_ids(session: dict, exclude: str | None = None) -> set[str]:
    """Ids of branches' first own messages (lineage junctions)."""
    out: set[str] = set()
    for b in session.get("branches", []):
        r = b.get("root")
        if r is not None and b.get("id") != exclude:
            out.add(r)
    return out


def walk_from(session: dict, start: str, skip_root_ids: set[str] | None = None) -> list[str]:
    """Linearized walk `start -> tip`.

    At each node choose the earliest child that is NOT the root of another
    branch (skip_root_ids). Fork children of *other* branches are skipped so
    the walk stays inside one lineage; our own branch root is never in the
    skip set. Deterministic: first-created child wins ties.
    """
    cmap = children_map(session)
    skip = skip_root_ids or set()
    path = [start]
    cur = start
    while True:
        nxt = None
        for c in cmap.get(cur, []):
            if c in skip:
                continue
            nxt = c
            break
        if nxt is None:
            break
        path.append(nxt)
        cur = nxt
    return path


def parent_chain(session: dict, msg_id: str) -> list[str]:
    """Ancestor chain of `msg_id` (exclusive), from root down to its parent."""
    out: list[str] = []
    cur = message(session, msg_id)
    while cur is not None and cur.get("parent"):
        out.append(cur["parent"])
        cur = message(session, cur["parent"])
    out.reverse()
    return out


def branch_path(session: dict, branch_id: str) -> list[dict]:
    """Message objects of a branch's OWN lineage (`root -> tip`).

    Empty for a stub branch (created but no own message yet).
    """
    b = get_branch(session, branch_id)
    if b is None or not b.get("root"):
        return []
    ids = walk_from(session, b["root"], skip_root_ids=branch_root_ids(session, exclude=branch_id))
    return [m for mid in ids if (m := message(session, mid)) is not None]


def branch_full_path(session: dict, branch_id: str) -> list[dict]:
    """Full replay context of a branch: shared spine + own lineage.

    The spine is the exact parent chain from the session root down to
    `fork_at` (parent links never cross lineages, so no filtering is
    needed); the own lineage starts at `branch.root` and walks to the tip,
    skipping sibling branch roots. For the main branch this is a plain walk
    from the session root skipping all other branch roots.
    """
    b = get_branch(session, branch_id)
    if b is None:
        return []
    if b.get("fork_at") is None:
        start = b.get("root") or (roots(session)[0] if roots(session) else None)
        if start is None:
            return []
        ids = walk_from(session, start, skip_root_ids=branch_root_ids(session, exclude=branch_id))
        return [m for mid in ids if (m := message(session, mid)) is not None]
    fork_at = b["fork_at"]
    spine_ids = parent_chain(session, fork_at) + [fork_at]
    out = [m for mid in spine_ids if (m := message(session, mid)) is not None]
    if b.get("root") and b["root"] != fork_at:
        own_ids = walk_from(session, b["root"], skip_root_ids=branch_root_ids(session, exclude=branch_id))
        out += [m for mid in own_ids if (m := message(session, mid)) is not None]
    return out


def append_message(
    session: dict,
    role: str,
    content: str,
    parent: str | None,
    branch_id: str | None = None,
    **meta,
) -> dict:
    """Append a message node; assigns ids and maintains branch roots.

    Meta kwargs (model/provider/latency_ms/tokens/error) are message
    provenance; only values that are actually known are stored.
    """
    msg = new_message(role=role, content=content, parent=parent, **meta)
    msg["id"] = _next_id(session, "m")
    session["messages"].append(msg)
    if branch_id:
        b = get_branch(session, branch_id)
        if b is not None and not b.get("root"):
            b["root"] = msg["id"]
    return msg


def _add_history(session: dict, event: str, detail: dict | None = None) -> None:
    session.setdefault("manifest", {}).setdefault("history", []).append(_event(event, detail))


# --------------------------------------------------------------------------
# P1: fork (session splitting at any node)
# --------------------------------------------------------------------------


def fork_branch(
    session: dict,
    at_msg_id: str,
    name: str = "",
    model: str | None = None,
    note: str = "",
    author: str = "",
) -> dict:
    """Fork a new branch at `at_msg_id` (the tree itself is untouched).

    The branch records `fork_at` (the node it split from) and an optional
    `model` override for what-if continuations. `root` is set when the
    branch's first own message is appended. The fork event is appended to
    manifest.history.
    """
    if message(session, at_msg_id) is None:
        raise SessionError(f"fork point {at_msg_id!r} not in messages")
    ensure_main_branch(session)
    branch = {
        "id": _next_id(session, "b"),
        "name": name or "",
        "root": None,
        "fork_at": at_msg_id,
        "model": model,
        "created": now_iso(),
    }
    session["branches"].append(branch)
    _add_history(
        session,
        "fork",
        detail={"branch": branch["id"], "at": at_msg_id, "model": model, "name": name, "note": note, "author": author},
    )
    return branch


def append_branch_message(session: dict, branch_id: str, role: str, content: str, **meta) -> dict:
    """Append a message that continues `branch_id` from its current tip."""
    b = get_branch(session, branch_id)
    if b is None:
        raise SessionError(f"branch {branch_id!r} not found")
    own = branch_path(session, branch_id)
    if b.get("fork_at") is None:
        if own:
            parent = own[-1]["id"]
        elif session["messages"]:
            parent = None
            b["root"] = _ensure_main_root(session)
        else:
            parent = None
    else:
        parent = own[-1]["id"] if own else b["fork_at"]
    msg = append_message(session, role, content, parent=parent, branch_id=branch_id, **meta)
    return msg


def _ensure_main_root(session: dict) -> str | None:
    b = main_branch(session)
    if session["messages"] and not b.get("root"):
        b["root"] = session["messages"][0]["id"]
    return b.get("root")


def ensure_main_branch(session: dict) -> dict:
    """Guarantee a main (fork_at=None) branch entry; returns it."""
    for b in session.get("branches", []):
        if b.get("fork_at") is None:
            return b
    mroot = roots(session)
    b = {
        "id": "b1",
        "name": "主线",
        "root": mroot[0] if mroot else None,
        "fork_at": None,
        "model": None,
        "created": session.get("created", now_iso()),
    }
    session.setdefault("branches", []).append(b)
    return b


# --------------------------------------------------------------------------
# P5 support: annotations (F5)
# --------------------------------------------------------------------------


def add_annotation(
    session: dict,
    target: str,
    kind: str,
    tags: list[str] | None = None,
    note: str = "",
    author: str = "teacher",
    ts: int | None = None,
) -> dict:
    """Attach a structured annotation to a message id (F5 taxonomy)."""
    if kind not in ANNOTATION_KINDS:
        raise SessionError(f"unknown annotation kind {kind!r}; expected {ANNOTATION_KINDS}")
    if message(session, target) is None:
        raise SessionError(f"annotation target {target!r} not in messages")
    ann = {
        "id": f"a{_next_id(session, 'a')[1:]}",
        "target": target,
        "kind": kind,
        "tags": list(tags or []),
        "note": note,
        "author": author,
        "ts": ts if ts is not None else now_ms(),
    }
    session.setdefault("annotations", []).append(ann)
    if message(session, target).get("annotations") is not None:
        message(session, target)["annotations"].append(ann["id"])
    return ann


def annotation_branch(session: dict, ann: dict) -> str:
    """Branch id that contains an annotation's target message ('' if main)."""
    for b in session.get("branches", []):
        if b.get("fork_at") is None:
            continue
        ids = {m["id"] for m in branch_path(session, b["id"])}
        if ann.get("target") in ids:
            return b["id"]
    bid = main_branch(session).get("id", "")
    return bid


def annotation_rows(session: dict) -> list[dict]:
    """Annotation rows for CSV/JSON export (the learning-analytics unit)."""
    rows = []
    for ann in session.get("annotations", []):
        m = message(session, ann.get("target", ""))
        tok = (m or {}).get("tokens") or {}
        ts = ann.get("ts")
        iso = datetime.fromtimestamp(ts / 1000).isoformat(timespec="seconds") if ts else ""
        rows.append(
            {
                "session_id": session.get("manifest", {}).get("session_id", ""),
                "turn_id": ann.get("target", ""),
                "kind": ann.get("kind", ""),
                "tags": ";".join(ann.get("tags") or []),
                "note": ann.get("note", ""),
                "author": ann.get("author", ""),
                "ts": iso,
                "model": (m or {}).get("model") or "",
                "latency_ms": (m or {}).get("latency_ms"),
                "tokens_out": tok.get("out"),
                "branch": annotation_branch(session, ann),
            }
        )
    return rows


def annotations_csv(session: dict) -> str:
    """Render annotation rows as CSV text (spec F5 column order)."""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(ANNOTATION_CSV_FIELDS))
    w.writeheader()
    w.writerows(annotation_rows(session))
    return buf.getvalue()


def export_annotations(session: dict, path, fmt: str = "csv") -> str:
    """Write annotations to `path` as csv (default) or json; returns path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        with open(p, "w", encoding="utf-8") as f:
            json.dump(annotation_rows(session), f, ensure_ascii=False, indent=2)
    else:
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(annotations_csv(session))
    return str(p)


def count_kinds(session: dict) -> dict[str, int]:
    """kind -> count over all annotations (learning-analytics summary)."""
    out = {k: 0 for k in ANNOTATION_KINDS}
    for ann in session.get("annotations", []):
        if ann.get("kind") in out:
            out[ann["kind"]] += 1
    return out
