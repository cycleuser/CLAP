"""P0 tests: session v2 format, save/load roundtrip, v1 migration, tree ops."""

import json

import pytest

from clap.core import session as S

# ---------------------------------------------------------------- fixtures


@pytest.fixture()
def v1_dict():
    """A realistic v1 session as old GUI save_chat_to_file wrote it."""
    return {
        "version": "1.0",
        "timestamp": "2026-10-19T10:00:00",
        "model": "gemma3:4b",
        "embed_model": "nomic-embed-text:latest",
        "document": "教案.docx",
        "messages": [
            {"role": "user", "content": "什么是兼语句？"},
            {"role": "assistant", "content": "兼语句是……"},
            {"role": "user", "content": "再举一个例子。"},
            {"role": "assistant", "content": "例句：请他来。"},
        ],
        "language": "zh",
    }


@pytest.fixture()
def v2_session():
    s = S.new_session(chat_model="gemma3:4b", document="教案.docx", language="zh")
    S.ensure_main_branch(s)
    m1 = S.append_message(s, "user", "什么是兼语句？", parent=None, branch_id="b1", model="gemma3:4b")
    m2 = S.append_message(
        s,
        "assistant",
        "兼语句是……",
        parent=m1["id"],
        branch_id="b1",
        model="gemma3:4b",
        provider="ollama-local",
        latency_ms=1234,
        tokens={"in": 10, "out": 20},
    )
    return s, m1, m2


# ------------------------------------------------------ construction/v2


def test_new_session_shape():
    s = S.new_session(chat_model="m", document="教案.docx")
    assert s["version"] == "2.0"
    required = {
        "version",
        "created",
        "manifest",
        "messages",
        "branches",
        "annotations",
        "language",
    }
    assert required <= set(s)
    man = s["manifest"]
    for key in (
        "session_id",
        "chat_model",
        "embed_model",
        "provider",
        "document",
        "clap_version",
        "annotations_schema",
        "privacy_mode",
        "routing",
        "history",
    ):
        assert key in man, key
    assert man["privacy_mode"] is True  # privacy is the default (F6)
    doc_meta = man["document"]
    assert doc_meta is not None
    assert doc_meta["name"] == "教案.docx"
    assert len(doc_meta["path_hash"]) == 64
    assert man["purpose"] in S.TEACHING_PURPOSES


def test_new_message_defaults():
    m = S.new_message("user", "hi", parent=None)
    assert m["provider"] is None and m["tokens"]["in"] is None
    assert m["annotations"] == []
    with pytest.raises(S.SessionError):
        S.new_message("robot", "hi")


# ------------------------------------------------------ save/load roundtrip


def test_save_load_roundtrip(tmp_path, v2_session):
    s, _, _ = v2_session
    path = tmp_path / "s.clap"
    S.save_session(s, path)
    loaded = S.load_session(path)
    # field-by-field identical
    assert json.loads(json.dumps(loaded, ensure_ascii=False, sort_keys=True)) == json.loads(
        json.dumps(s, ensure_ascii=False, sort_keys=True)
    )
    # digest stable across io
    assert S.session_digest(loaded) == S.session_digest(s)
    # reload from the written bytes again -> idempotent, same digest
    again = S.load_session(path)
    assert S.session_digest(again) == S.session_digest(s)


def test_load_rejects_garbage(tmp_path):
    p = tmp_path / "bad.clap"
    p.write_text("[1,2,3]", encoding="utf-8")
    with pytest.raises(S.SessionError):
        S.load_session(p)


# ------------------------------------------------------ v1 migration


def test_migrate_v1_chain_and_manifest(v1_dict):
    v2 = S.migrate_v1(v1_dict)
    assert v2["version"] == "2.0"
    ids = [m["id"] for m in v2["messages"]]
    parents = [m["parent"] for m in v2["messages"]]
    assert parents[0] is None
    assert parents[1:] == ids[:-1]  # linear chain preserved
    assert len(ids) == len(set(ids))
    man = v2["manifest"]
    assert man["chat_model"] == "gemma3:4b"
    assert man["embed_model"] == "nomic-embed-text:latest"
    assert man["provider"] == "ollama-local"  # v1 era: local ollama only
    assert man["document"]["name"] == "教案.docx"
    assert man["document"]["bytes"] == 0  # non-existent path, bytes unknown
    assert man["privacy_mode"] is True
    assert v2["annotations"] == []
    assert v2["branches"][0]["root"] == ids[0]
    assert v2["branches"][0]["name"] == "主线"
    hist = man["history"][0]
    assert hist["event"] == "migrate_v1"
    # unknown provenance stays null, never fabricated
    for m in v2["messages"]:
        assert m["provider"] is None
        assert m["model"] is None
        assert m["ts"] is None
        assert m["tokens"] == {"in": None, "out": None}


def test_v1_file_loads_migrated(tmp_path, v1_dict):
    p = tmp_path / "legacy.clap"
    p.write_text(json.dumps(v1_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    s = S.load_session(p)
    assert s["version"] == "2.0"
    assert len(s["messages"]) == 4
    text = p.read_text(encoding="utf-8")
    assert json.loads(text)["version"] == "1.0"  # file on disk untouched until saved
    # save through new code, reload, stable
    S.save_session(s, p)
    s2 = S.load_session(p)
    assert S.session_digest(s2) == S.session_digest(s)
    # semantic equality: contents preserved in order
    assert [m["content"] for m in s2["messages"]] == [
        m["content"] for m in v1_dict["messages"]
    ]


def test_migrate_missing_fields(v1_dict):
    del v1_dict["model"]
    del v1_dict["document"]
    v2 = S.migrate_v1(v1_dict)
    assert v2["manifest"]["chat_model"] == ""
    assert v2["manifest"]["document"]["name"] == ""
    assert v2["manifest"]["document"]["bytes"] == 0


# ------------------------------------------------------ validation


def test_validate_ok(v2_session):
    s, _, _ = v2_session
    assert S.validate_session(s) == []


def test_validate_reports_issues(v2_session):
    s, m1, _ = v2_session
    bad = json.loads(json.dumps(s, ensure_ascii=False))
    bad["version"] = "9"
    bad["messages"].append({"id": "dup", "parent": "nope", "role": "user"})  # unknown parent + bad fields
    bad["messages"].append(dict(bad["messages"][-1], id="dup", role="alien"))
    bad["branches"].append({"id": "bx", "root": "ghost"})
    bad["annotations"].append({"id": "a1", "target": "ghost", "kind": "not-a-kind"})
    del bad["manifest"]["session_id"]
    issues = S.validate_session(bad)
    joined = " ".join(issues)
    for expect in ("version", "unknown parent", "bad role", "duplicate message id", "root", "target", "bad kind", "session_id"):
        assert expect in joined, (expect, joined)
    with pytest.raises(S.SessionValidationError):
        S.assert_valid(bad)


# ------------------------------------------------------ tree walk (P0 basics)


def test_children_map_and_walk(v2_session):
    s, m1, m2 = v2_session
    cm = S.children_map(s)
    assert cm[None] == [m1["id"]]
    assert cm[m1["id"]] == [m2["id"]]
    walk = S.walk_from(s, m1["id"])
    assert walk == [m1["id"], m2["id"]]
    assert S.parent_chain(s, m2["id"]) == [m1["id"]]


def test_walk_skips_other_branch_roots(v2_session):
    s, m1, m2 = v2_session
    S.ensure_main_branch(s)
    b2 = S.fork_branch(s, m2["id"], name="提高")
    assert b2["root"] is None
    u = S.append_branch_message(s, b2["id"], "user", "支线第一问")
    assert s["branches"][1]["root"] == u["id"]
    # main walk never steps into the fork's root
    main_ids = S.walk_from(s, m1["id"], skip_root_ids=S.branch_root_ids(s, exclude="b1"))
    assert u["id"] not in main_ids
    # branch own path
    own = S.branch_path(s, b2["id"])
    assert [m["id"] for m in own] == [u["id"]]
    # full replay context = spine (m1,m2) + own
    full = S.branch_full_path(s, b2["id"])
    assert [m["id"] for m in full] == [m1["id"], m2["id"], u["id"]]


def test_fork_at_works(v2_session):
    s, m1, m2 = v2_session
    S.ensure_main_branch(s)
    with pytest.raises(S.SessionError):
        S.fork_branch(s, "ghost")
    b = S.fork_branch(s, m2["id"], name="拓展", model="deepseek-chat", note="拓展支线")
    rec = s["manifest"]["history"][-1]
    assert rec["event"] == "fork" and rec["detail"]["at"] == m2["id"]
    assert b["model"] == "deepseek-chat"


def test_append_sequential_ids_unique(v2_session):
    s, _, _ = v2_session
    n0 = len(s["messages"])
    for _ in range(5):
        S.append_branch_message(s, "b1", "user", "x")
    ids = [m["id"] for m in s["messages"]]
    assert len(ids) == n0 + 5 and len(set(ids)) == len(ids)
