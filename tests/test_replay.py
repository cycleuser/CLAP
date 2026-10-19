"""P2 tests: replay engine (context modes, checkpoint/resume, failure retry)."""


import pytest

from clap.core import replay as R
from clap.core import session as S


class FakeProvider:
    """Deterministic offline provider for engine tests (unit-level only)."""

    id = "fake"

    def __init__(self, outputs, fail_on=None):
        self.outputs = list(outputs)
        self.fail_on = fail_on or set()
        self.received: list[list[dict]] = []
        self.calls = 0
        self.last_meta: dict = {}

    def chat(self, messages, model, stream=True):
        self.calls += 1
        self.received.append([dict(m) for m in messages])
        if len(self.received) in self.fail_on:
            self.last_meta = {"provider": self.id, "model": model, "tokens": {"in": None, "out": None}}
            raise RuntimeError("simulated provider failure")

        def gen():
            text = self.outputs[len(self.received) - 1]
            self.last_meta = {
                "provider": self.id,
                "model": model,
                "latency_ms": 10,
                "tokens": {"in": 7, "out": len(text)},
            }
            yield text
            self.last_meta = {
                "provider": self.id,
                "model": model,
                "latency_ms": 11,
                "tokens": {"in": 7, "out": len(text)},
            }

        return gen()


@pytest.fixture()
def three_turn_session():
    """user/assistant x2 + user (7 messages, replay has 3 turns)."""
    s = S.new_session(chat_model="orig-model", document="", language="zh")
    S.ensure_main_branch(s)
    p = None
    seq = [
        ("user", "问1"),
        ("assistant", "原答1"),
        ("user", "问2"),
        ("assistant", "原答2"),
        ("user", "问3"),
    ]
    for role, content in seq:
        n = S.append_message(s, role, content, parent=p, branch_id="b1")
        p = n["id"]
    return s


def test_replay_basic_context_replayed(three_turn_session, tmp_path):
    s = three_turn_session
    prov = FakeProvider(["新答1", "新答2", "新答3"])
    engine = R.ReplayEngine(s, prov, model="fake-model")
    assert engine.total_turns() == 3

    steps = list(engine.run())
    assert [st.turn for st in steps] == [1, 2, 3]
    assert [st.output for st in steps] == ["新答1", "新答2", "新答3"]
    assert all(st.error is None for st in steps)
    assert steps[0].prompt_tokens == 7
    assert steps[2].completion_tokens == 3
    assert steps[0].resumed is False

    # turn 3 context: user1 + replayed a1 + user2 + replayed a2 + user3
    ctx3 = prov.received[2]
    assert ctx3 == [
        {"role": "user", "content": "问1"},
        {"role": "assistant", "content": "新答1"},
        {"role": "user", "content": "问2"},
        {"role": "assistant", "content": "新答2"},
        {"role": "user", "content": "问3"},
    ]
    # original assistant text must NOT be sent to the model
    assert all(m["content"] not in ("原答1", "原答2") for m in ctx3 if m["role"] != "user")


def test_replay_context_original(three_turn_session):
    s = three_turn_session
    prov = FakeProvider(["新1", "新2", "新3"])
    engine = R.ReplayEngine(s, prov, model="m", context_mode="original")
    list(engine.run())
    ctx2 = prov.received[1]
    assert ctx2[1] == {"role": "assistant", "content": "原答1"}


def test_resume_from_checkpoint(tmp_path, three_turn_session):
    s = three_turn_session
    log = tmp_path / "r.jsonl"

    # first run: only 2 turns
    R.replay_to_log(s, FakeProvider(["a1", "a2"]), log_path=log, to_turn=2)
    rows = R.load_replay_log(log)
    assert [r.get("turn") for r in rows if "turn" in r] == [1, 2]
    assert not any(r.get("event") == "replay_complete" for r in rows)

    # resume: completes turn 3, marked resumed, marker written
    summary = R.replay_to_log(s, FakeProvider(["a3"]), log_path=log)
    assert summary["completed_turns"] == 3
    assert summary["resumed_from"] == 3
    rows = R.load_replay_log(log)
    turns = [r["turn"] for r in rows if "turn" in r]
    assert turns == [1, 2, 3]  # deduped
    t3 = [r for r in rows if r.get("turn") == 3][0]
    assert t3["output"] == "a3" and t3["resumed"] is True
    assert any(r.get("event") == "replay_complete" for r in rows)


def test_failed_turn_retried_on_resume(tmp_path, three_turn_session):
    s = three_turn_session
    log = tmp_path / "r.jsonl"
    R.replay_to_log(s, FakeProvider(["a1", "x", "a3"], fail_on={2}), log_path=log)
    rows = R.load_replay_log(log)
    err_rows = [r for r in rows if r.get("error")]
    assert err_rows and err_rows[0]["turn"] == 2

    summary = R.replay_to_log(s, FakeProvider(["fixed", "a3"]), log_path=log)
    assert summary["completed_turns"] == 3
    rows = R.load_replay_log(log)
    t2 = [r for r in rows if r.get("turn") == 2][0]
    assert t2["output"] == "fixed" and not t2.get("error")  # retried row replaces the failed row
    assert any(r.get("event") == "replay_complete" for r in rows)


def test_replay_already_complete_noop(tmp_path, three_turn_session):
    s = three_turn_session
    log = tmp_path / "r.jsonl"
    R.replay_to_log(s, FakeProvider(["a", "b", "c"]), log_path=log)
    n_calls = 3
    summary = R.replay_to_log(s, FakeProvider(["x", "y", "z"]), log_path=log)
    assert summary.get("skipped") == "already complete"
    rows = R.load_replay_log(log)
    # no duplicate rows appended
    assert len([r for r in rows if "turn" in r]) == n_calls


def test_fork_branch_replay_context(tmp_path, teaching_like):
    s, p = teaching_like
    b2 = S.fork_branch(s, p["m6"], name="提高")
    S.append_branch_message(s, b2["id"], "user", "支线问1")
    prov = FakeProvider(["a1", "a2", "a3", "支线新答"])
    engine = R.ReplayEngine(s, prov, branch_id=b2["id"], model="fake-model")
    # replay of the forked branch contains the *full context* (spine + own)
    assert engine.total_turns() == 4  # m1,m3,m5 spine users + 支线问1
    steps = list(engine.run())
    assert steps[-1].output == "支线新答"
    ctx = prov.received[-1]
    assert ctx[-1] == {"role": "user", "content": "支线问1"}
    assert ctx[0]["content"] == "turn0"  # spine context included


@pytest.fixture()
def teaching_like():
    s = S.new_session(chat_model="gemma3:4b", language="zh")
    S.ensure_main_branch(s)
    out = {}
    parent = None
    for i in range(6):
        role = "user" if i % 2 == 0 else "assistant"
        node = S.append_message(s, role, f"turn{i}", parent=parent, branch_id="b1")
        if i % 2 == 0:
            out[node["id"]] = node
        parent = node["id"]
    return s, {"m6": parent}
