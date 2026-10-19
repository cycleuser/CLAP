"""CLAP-NEXT F1: shareable session packages + central session library.

A share package (`.clapshare`) is a zipped pair:

- `session.clap` — the v2 session, with the local document path stripped
  (share hygiene: names/hashes travel, full local paths do not);
- `MANIFEST.json` — evidence chain digests (sha256 of the session bytes and
  per-field canonical digests) so adopt can prove field-by-field fidelity.

The central library (`~/.clap/sessions/` + `index.json`) stores adopted and
own sessions with a recent-list; adopt supports rename/overwrite/fork
policies for session-id conflicts.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

from clap.core import session as session_mod

PACKAGE_FORMAT = "clapshare/1"
SESSION_MEMBER = "session.clap"
MANIFEST_MEMBER = "MANIFEST.json"

STRIPPED_DOC_PATH = "<stripped-for-share>"


def default_library_dir() -> Path:
    return Path.home() / ".clap" / "sessions"


def _field_digests(session: dict) -> dict[str, str]:
    return {
        key: session_mod.sha256_strings(key=key, value=session.get(key))
        for key in ("manifest", "messages", "branches", "annotations", "language", "created")
    }


def pack(source, out_dir=None) -> Path:
    """Zip a .clap session + MANIFEST into `<stem>.clapshare`; returns path."""
    session = session_mod.load_session(source)
    # share hygiene: strip local document path so it does not travel
    doc = session.get("manifest", {}).get("document") or {}
    if doc.get("path"):
        doc["path"] = STRIPPED_DOC_PATH
    issues = session_mod.validate_session(session)
    if issues:
        raise session_mod.SessionValidationError("; ".join(issues))

    source_path = Path(source)
    out_dir = Path(out_dir) if out_dir else source_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pkg = out_dir / f"{source_path.stem}.clapshare"

    session_bytes = json.dumps(session, ensure_ascii=False, indent=2).encode("utf-8")
    manifest = {
        "format": PACKAGE_FORMAT,
        "session_id": session["manifest"]["session_id"],
        "session_file": SESSION_MEMBER,
        "session_sha256": session_mod._sha256(session_bytes),
        "packed_at": session_mod.now_iso(),
        "clap_version": session["manifest"].get("clap_version", ""),
        "counts": {
            "messages": len(session.get("messages", [])),
            "branches": len(session.get("branches", [])),
            "annotations": len(session.get("annotations", [])),
        },
        "fields": _field_digests(session),
    }
    with zipfile.ZipFile(pkg, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(SESSION_MEMBER, session_bytes)
        z.writestr(MANIFEST_MEMBER, json.dumps(manifest, ensure_ascii=False, indent=2))
    return pkg


def read_manifest(package) -> dict:
    with zipfile.ZipFile(package) as z:
        return json.loads(z.read(MANIFEST_MEMBER))


def verify_package(package) -> dict:
    """Verify zip structure + digests; returns {ok, manifest, reasons}."""
    reasons: list[str] = []
    try:
        with zipfile.ZipFile(package) as z:
            names = set(z.namelist())
        if SESSION_MEMBER not in names or MANIFEST_MEMBER not in names:
            reasons.append(f"missing members; have {sorted(names)}")
            return {"ok": False, "reasons": reasons, "manifest": None}
        man = read_manifest(package)
        if man.get("format") != PACKAGE_FORMAT:
            reasons.append(f"format {man.get('format')!r} != {PACKAGE_FORMAT}")
        with zipfile.ZipFile(package) as z:
            session_bytes = z.read(SESSION_MEMBER)
        if man.get("session_sha256") != session_mod._sha256(session_bytes):
            reasons.append("session_sha256 mismatch")
        session = session_mod.normalize(json.loads(session_bytes.decode("utf-8")))
        issues = session_mod.validate_session(session)
        reasons.extend(issues)
        if man.get("fields"):
            for key, dig in man["fields"].items():
                if _field_digests(session).get(key) != dig:
                    reasons.append(f"field digest mismatch: {key}")
        return {"ok": not reasons, "reasons": reasons, "manifest": man, "session": session}
    except zipfile.BadZipFile as e:
        return {"ok": False, "reasons": [f"bad zip: {e}"], "manifest": None}


def _register(session: dict, path: Path, library: Path, shared: bool) -> dict:
    first_user = next((m.get("content", "") for m in session.get("messages", []) if m.get("role") == "user"), "")
    entry = {
        "session_id": session["manifest"]["session_id"],
        "name": first_user[:40] or (session["manifest"].get("document") or {}).get("name", ""),
        "purpose": session["manifest"].get("purpose", ""),
        "chat_model": session["manifest"].get("chat_model", ""),
        "messages": len(session.get("messages", [])),
        "file": str(path),
        "shared": shared,
        "adopted": session_mod.now_iso(),
    }
    idx_path = library / "index.json"
    index = []
    if idx_path.exists():
        try:
            index = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            index = []
    index = [e for e in index if e.get("file") != str(path)]
    index.insert(0, entry)
    idx_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    return entry


def library_path(library=None) -> Path:
    return Path(library) if library else default_library_dir()


def adopt(package, policy: str = "rename", library=None) -> dict:
    """Verify and store a share package into the library; returns the record.

    Policies for session_id conflicts: rename (new unique file name +
    suffix), overwrite (replace existing), fork (keep file, issue a new
    session_id, record adopt_fork in manifest history).
    """
    result = verify_package(package)
    if not result["ok"]:
        raise session_mod.SessionError(f"package failed verification: {result['reasons']}")
    session: dict = result["session"]

    lib_dir = library_path(library)
    lib_dir.mkdir(parents=True, exist_ok=True)
    target = lib_dir / f"{session['manifest']['session_id']}.clap"

    policy_applied = "first-adopt"
    if target.exists():
        existing = session_mod.load_session(target)
        if session_mod.session_digest(existing) == session_mod.session_digest(session):
            policy_applied = "identical-noop"
            _register(existing, target, lib_dir, True)
            return {"session_id": target.stem, "path": str(target), "policy": policy_applied, "verified": True}
        if policy == "overwrite":
            policy_applied = "overwrite"
        elif policy == "fork":
            old_id = session["manifest"]["session_id"]
            session["manifest"]["session_id"] = _unique_session_id(lib_dir)
            session_mod._add_history(
                session,
                "adopt_fork",
                detail={"from_session_id": old_id, "policy": "fork"},
            )
            target = lib_dir / f"{session['manifest']['session_id']}.clap"
            policy_applied = "fork"
        else:  # rename default
            n = 1
            while target.exists():
                target = lib_dir / f"{session['manifest']['session_id']}-{n}.clap"
                n += 1
            policy_applied = "rename"

    session_mod.save_session(session, target)
    entry = _register(session, target, lib_dir, True)
    return {"session_id": target.stem, "path": str(target), "policy": policy_applied, "verified": True, "entry": entry}


def _unique_session_id(lib_dir: Path) -> str:
    import uuid

    while True:
        sid = uuid.uuid4().hex[:12]
        if not (lib_dir / f"{sid}.clap").exists():
            return sid


def save_to_library(session: dict, library=None) -> dict:
    """Register a locally-owned session in the central library (P4)."""
    lib_dir = library_path(library)
    lib_dir.mkdir(parents=True, exist_ok=True)
    path = lib_dir / f"{session['manifest']['session_id']}.clap"
    session_mod.save_session(session, path)
    entry = _register(session, path, lib_dir, False)
    return {"path": str(path), "entry": entry}


def list_library(library=None) -> list[dict]:
    idx_path = library_path(library) / "index.json"
    if not idx_path.exists():
        return []
    try:
        return json.loads(idx_path.read_text(encoding="utf-8"))
    except Exception:
        return []


def load_from_library(session_id: str, library=None) -> dict | None:
    """Load a library session by stored filename stem (the adopted id)."""
    lib_dir = library_path(library)
    p = lib_dir / f"{session_id}.clap"
    if not p.exists():
        for e in list_library(lib_dir):
            if Path(e.get("file", "")).stem == session_id and Path(e["file"]).exists():
                p = Path(e["file"])
                break
        else:
            return None
    return session_mod.load_session(p)


def copy_package(package, dest_dir) -> Path:
    """Copy a package to another directory (simulating machine A -> B)."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    return Path(shutil.copy2(package, dest / Path(package).name))
