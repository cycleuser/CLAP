"""P1 tests: fork semantics — main-line invariance, lineage isolation, model override."""

import json

import pytest

from clap.core import session as S


@pytest.fixture()
def teaching_session():
    """An 8-turn teaching main line (user/assistant alternating)."""
    s = S.new_session(chat_model="gemma3:4b", document="教案.docx", language="zh")
    S.ensure_main_branch(s)
    parent = None
    for i in range(8):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"主线 turn{i} 内容"
        node = S.append_message(
            s, role, content, parent=parent, branch_id="b1",
            model="gemma3:4b" if role == "assistant" else None,
            provider="ollama-local" if role == "assistant" else None,
            latency_ms=100 * i if role == "assistant" else None,
            tokens={"in": i, "out": i + 1} if role == "assistant" else None,
        )
        parent = node["id"]
    return s


def _main_snapshot(s: dict) -> dict:
    main = S.main_branch(s)
    own = S.branch_path(s, main["id"])
    return {
        "ids": [m["id"] for m in own],
        "parents": [m["parent"] for m in own],
        "contents": [m["content"] for m in own],
        "roles": [m["role"] for m in own],
        "latencies": [m["latency_ms"] for m in own],
        "model": main.get("model"),
    }


def test_fork_main_line_invariance(teaching_session):
    s = teaching_session
    before = _main_snapshot(s)
    S.fork_branch(s, "m6", name="基础", model="deepseek-chat")
    S.append_branch_message(s, "b2", "user", "基础支线先补一步")
    S.append_branch_message(s, "b2", "assistant", "好的，我们补充讲一遍。", model="deepseek-chat")
    after = _main_snapshot(s)
    # main-line snapshot diff must be empty
    assert before == after
    # fork event recorded in manifest history
    rec = s["manifest"]["history"][-1]
    assert rec["event"] == "fork" and rec["detail"]["at"] == "m6" and rec["detail"]["model"] == "deepseek-chat"


def test_three_forks_independent_lineages(teaching_session):
    s = teaching_session
    b_basic = S.fork_branch(s, "m6", name="基础", model="deepseek-chat")
    b_adv = S.fork_branch(s, "m6", name="提高")
    b_ext = S.fork_branch(s, "m6", name="拓展", model="gemma4:31b")
    for b, first in zip((b_basic, b_adv, b_ext), ("u1?", "u2?", "u3?"), strict=True):
        S.append_branch_message(s, b["id"], "user", first)
    # each branch's own path is disjoint
    paths = [tuple(m["id"] for m in S.branch_path(s, b["id"])) for b in (b_basic, b_adv, b_ext)]
    assert len(paths[0]) == len(paths[1]) == len(paths[2]) == 1
    assert len(set(paths)) == 3  # sibling fork roots are distinct nodes
    # all parents point at the fork point
    for b in (b_basic, b_adv, b_ext):
        own = S.branch_path(s, b["id"])
        assert own[0]["parent"] == "m6"
    # validation still clean
    assert S.validate_session(s) == []


def test_branch_context_includes_shared_spine(teaching_session):
    s = teaching_session
    b = S.fork_branch(s, "m6", name="提高")
    S.append_branch_message(s, b["id"], "user", "支线问题")
    full = S.branch_full_path(s, b["id"])
    ids = [m["id"] for m in full]
    # spine m1..m6 + own (m8+)
    assert ids[:6] == ["m1", "m2", "m3", "m4", "m5", "m6"]
    assert ids[-1] == "支线问题" or s["messages"][-1]["content"] == "支线问题"
    # the own part does not leak into the main path
    main_ids = [m["id"] for m in S.branch_path(s, "b1")]
    assert s["messages"][-1]["id"] not in main_ids


def test_subfork_within_branch(teaching_session):
    s = teaching_session
    b2 = S.fork_branch(s, "m6", name="提高")
    u = S.append_branch_message(s, b2["id"], "user", "提高问1")
    a = S.append_branch_message(s, b2["id"], "assistant", "提高答1")
    # sub-fork from inside b2
    b3 = S.fork_branch(s, a["id"], name="拓展")
    u3 = S.append_branch_message(s, b3["id"], "user", "拓展问1")
    # b2's walk skips b3's root and keeps its own tip = a's chain
    b2_ids = [m["id"] for m in S.branch_path(s, b2["id"])]
    assert b3["root"] not in b2_ids
    assert u3["id"] not in b2_ids
    # b3 full path: m1..m6 (spine) + u,a (b2 own) + u3
    full_ids = [m["id"] for m in S.branch_full_path(s, b3["id"])]
    assert full_ids == ["m1", "m2", "m3", "m4", "m5", "m6", u["id"], a["id"], u3["id"]]
    assert S.validate_session(s) == []


def test_fork_persists_roundtrip(tmp_path, teaching_session):
    s = teaching_session
    S.fork_branch(s, "m4", name="提高", model="deepseek-chat")
    p = tmp_path / "forked.clap"
    S.save_session(s, p)
    s2 = S.load_session(p)
    assert len(s2["branches"]) == 2
    assert json.loads(json.dumps(s2, sort_keys=True)) == json.loads(json.dumps(s, sort_keys=True))
