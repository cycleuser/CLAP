"""P4 tests: share package fidelity, adopt policies, central library."""

import json
import zipfile

import pytest

from clap.core import session as S
from clap.core import sharing as SH


@pytest.fixture()
def rich_session():
    """A session with branches + annotations (all shareable content)."""
    s = S.new_session(chat_model="gemma3:4b", document="/tmp/local/教案.docx", language="zh")
    S.ensure_main_branch(s)
    m1 = S.append_message(s, "user", "什么是兼语句？", parent=None, branch_id="b1", model="gemma3:4b")
    m2 = S.append_message(
        s, "assistant", "兼语句是……", parent=m1["id"], branch_id="b1",
        model="gemma3:4b", provider="ollama-local", latency_ms=900, tokens={"in": 5, "out": 9},
    )
    S.fork_branch(s, m2["id"], name="提高", model="deepseek-chat")
    S.append_branch_message(s, "b2", "user", "补一步")
    S.add_annotation(s, m2["id"], "keypoint", tags=["兼语句"], note="考点", author="teacher")
    return s


def test_pack_manifest_and_hygiene(tmp_path, rich_session):
    src = tmp_path / "lesson.clap"
    S.save_session(rich_session, src)
    pkg = SH.pack(src, out_dir=tmp_path / "out")
    assert pkg.name == "lesson.clapshare"
    man = SH.read_manifest(pkg)
    assert man["format"] == SH.PACKAGE_FORMAT
    assert man["counts"] == {"messages": 3, "branches": 2, "annotations": 1}
    assert len(man["session_sha256"]) == 64
    with zipfile.ZipFile(pkg) as z:
        packed = json.loads(z.read(SH.SESSION_MEMBER))
    # local absolute path must not travel
    assert packed["manifest"]["document"].get("path") in (None, SH.STRIPPED_DOC_PATH)
    assert "/tmp/local" not in json.dumps(packed)


def test_share_adopt_field_fidelity(tmp_path, rich_session):
    src = tmp_path / "lesson.clap"
    S.save_session(rich_session, src)
    pkg = SH.pack(src, out_dir=tmp_path)
    # simulate machine B: copy the package elsewhere
    pkg_b = SH.copy_package(pkg, tmp_path / "B")
    v = SH.verify_package(pkg_b)
    assert v["ok"] is True, v["reasons"]
    adopted = SH.adopt(pkg_b, library=tmp_path / "lib")
    assert adopted["verified"] is True

    # field-by-field diff must be zero
    original = S.load_session(src)
    received = S.load_session(adopted["path"])
    for key in ("manifest", "messages", "branches", "annotations", "language"):
        assert SH._field_digests(original)[key] == SH._field_digests(received)[key], key
        a = json.loads(json.dumps(original.get(key), ensure_ascii=False, sort_keys=True))
        b = json.loads(json.dumps(received.get(key), ensure_ascii=False, sort_keys=True))
        if key == "manifest":
            a["document"].pop("path", None)
            b["document"].pop("path", None)
        assert a == b, key  # literal equality too, not just digest


def test_adopt_policies(tmp_path, rich_session):
    src = tmp_path / "lesson.clap"
    S.save_session(rich_session, src)
    pkg = SH.pack(src, out_dir=tmp_path)
    lib = tmp_path / "lib"

    first = SH.adopt(pkg, library=lib)
    # re-adopt identical content -> noop
    same = SH.adopt(pkg, library=lib)
    assert same["policy"] == "identical-noop"

    # modify -> conflict rename (default)
    rich_session["messages"][0]["content"] = "改过的问题"
    modified = tmp_path / "modified.clap"
    S.save_session(rich_session, modified)
    pkg2 = SH.pack(modified, out_dir=tmp_path)
    renamed = SH.adopt(pkg2, library=lib, policy="rename")
    assert renamed["policy"] == "rename"
    assert renamed["path"] != first["path"]

    # fork policy: new session_id, history event
    forked = SH.adopt(pkg2, library=lib, policy="fork")
    assert forked["policy"] == "fork"
    s_f = S.load_session(forked["path"])
    evs = [h["event"] for h in s_f["manifest"]["history"]]
    assert "adopt_fork" in evs

    # overwrite policy: stale file at the target id gets replaced
    rich_session["messages"][0]["content"] = "再改"
    modified2 = tmp_path / "modified2.clap"
    S.save_session(rich_session, modified2)
    pkg3 = SH.pack(modified2, out_dir=tmp_path)
    man3 = SH.read_manifest(pkg3)
    target_id = man3["session_id"]
    (lib / f"{target_id}.clap").write_text("{}", encoding="utf-8")  # stale placeholder
    over = SH.adopt(pkg3, library=lib, policy="overwrite")
    assert over["session_id"] == target_id
    assert over["policy"] == "overwrite"
    with open(lib / f"{target_id}.clap", encoding="utf-8") as f:
        assert json.load(f)["version"] == "2.0"  # real content now, not the stale {}


def test_library_list_and_load(tmp_path, rich_session):
    lib = tmp_path / "lib"
    SH.save_to_library(rich_session, library=lib)
    entries = SH.list_library(lib)
    assert len(entries) == 1
    assert entries[0]["session_id"] == rich_session["manifest"]["session_id"]
    assert entries[0]["shared"] is False
    loaded = SH.load_from_library(rich_session["manifest"]["session_id"], library=lib)
    assert loaded is not None
    assert S.session_digest(loaded) == S.session_digest(rich_session)


def test_adopt_corrupt_package(tmp_path):
    bad = tmp_path / "bad.clapshare"
    bad.write_bytes(b"not a zip")
    with pytest.raises(S.SessionError):
        SH.adopt(bad, library=tmp_path / "lib")


def test_verify_detects_tampering(tmp_path, rich_session):
    src = tmp_path / "lesson.clap"
    S.save_session(rich_session, src)
    pkg = SH.pack(src, out_dir=tmp_path)
    # tamper: rewrite session member, keep old manifest digest
    with zipfile.ZipFile(pkg) as z:
        names = z.namelist()
        data = {n: z.read(n) for n in names}
    sess = json.loads(data[SH.SESSION_MEMBER])
    sess["messages"][0]["content"] = "tampered"
    data[SH.SESSION_MEMBER] = json.dumps(sess, ensure_ascii=False).encode("utf-8")
    with zipfile.ZipFile(pkg, "w") as z:
        for n in names:
            z.writestr(n, data[n])
    v = SH.verify_package(pkg)
    assert v["ok"] is False
    assert any("sha256" in r or "digest" in r for r in v["reasons"])
