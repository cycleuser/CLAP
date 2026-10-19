"""CLAP-NEXT F3: whole-replay engine (what-if with a different model).

Semantics (spec-session-features.md F3): given a session tree, a branch and
a provider/model, every historical *user* message is re-sent in order and a
new assistant turn is produced each time — the original assistant text is
NOT concatenated as an answer. Two context modes:

- "replayed" (default): earlier assistant turns in the context are the
  replayed outputs (divergence propagates — what a different model would
  have done from the same prompts);
- "original": earlier assistant turns are the original file contents
  (cleaner per-turn comparison).

Every completed turn appends a line to the replay log (jsonl):
{turn, msg_id, branch, provider, model, prompt_tokens, completion_tokens,
 latency_ms, output, error, ts}. Failed turns are logged with `error` and
do NOT count as progress. The log itself is the checkpoint: a rerun resumes
after the last successfully completed turn by reading the existing log.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass

from clap.core import session as session_mod


@dataclass
class ReplayStep:
    """One replayed turn (one jsonl line in the replay log)."""

    turn: int  # 1-based
    msg_id: str  # source user message id
    branch: str
    provider: str | None
    model: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_ms: int | None
    output: str
    error: str | None = None
    resumed: bool = False
    ts: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_row(cls, row: dict) -> ReplayStep:
        return cls(
            turn=int(row["turn"]),
            msg_id=row.get("msg_id", ""),
            branch=row.get("branch", ""),
            provider=row.get("provider"),
            model=row.get("model"),
            prompt_tokens=row.get("prompt_tokens"),
            completion_tokens=row.get("completion_tokens"),
            latency_ms=row.get("latency_ms"),
            output=row.get("output", ""),
            error=row.get("error"),
            resumed=bool(row.get("resumed")),
            ts=row.get("ts", 0),
        )


def load_replay_log(path) -> list[dict]:
    """Read a replay log; returns its rows (per-turn: last row wins), in order."""
    rows: list[dict] = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "turn" in obj:
                    rows = [r for r in rows if not ("turn" in r and r.get("turn") == obj["turn"])]
                    rows.append(obj)
                elif obj.get("event"):
                    rows.append(obj)
    except FileNotFoundError:
        return []
    return rows


class ReplayEngine:
    """Whole-replay over a branch with a chosen provider/model."""

    def __init__(
        self,
        session: dict,
        provider,
        branch_id: str | None = None,
        model: str | None = None,
        context_mode: str = "replayed",
    ):
        self.session = session
        self.provider = provider
        main = session_mod.main_branch(session)
        self.branch_id = branch_id or main.get("id") or "b1"
        if session_mod.get_branch(session, self.branch_id) is None:
            raise session_mod.SessionError(f"branch {self.branch_id!r} not found")
        branch = session_mod.get_branch(session, self.branch_id) or {}
        self.model = model or branch.get("model") or session.get("manifest", {}).get("chat_model", "")
        if not self.model:
            raise session_mod.SessionError("no model for replay (branch/manifest has none); pass --model")
        self.context_mode = context_mode if context_mode in ("replayed", "original") else "replayed"

        path = session_mod.branch_full_path(session, self.branch_id)
        self.path_messages = [m for m in path if m.get("role") in ("user", "assistant")]
        self.turns = [m for m in self.path_messages if m.get("role") == "user"]

    # ------------------------------------------------------------------

    def total_turns(self) -> int:
        return len(self.turns)

    def _context(self, turn_idx: int, prior_steps: list[ReplayStep]) -> list[dict]:
        """Message list to send for turn `turn_idx` (0-based)."""
        cur_user = self.turns[turn_idx]
        prior_out: dict[int, str] = {}
        for st in prior_steps:
            if st.output and not st.error:
                prior_out[st.turn - 1] = st.output

        turn_index_by_id = {t["id"]: i for i, t in enumerate(self.turns)}
        msgs: list[dict] = []
        for k, m in enumerate(self.path_messages):
            if m["id"] == cur_user["id"]:
                break
            if m.get("role") == "user":
                msgs.append({"role": "user", "content": m.get("content", "")})
                nxt = self.path_messages[k + 1] if k + 1 < len(self.path_messages) else None
                if nxt is not None and nxt.get("role") == "assistant":
                    tidx = turn_index_by_id.get(m["id"])
                    if tidx is not None and tidx in prior_out and self.context_mode == "replayed":
                        msgs.append({"role": "assistant", "content": prior_out[tidx]})
                    else:
                        msgs.append({"role": "assistant", "content": nxt.get("content", "")})
        msgs.append({"role": "user", "content": cur_user.get("content", "")})
        return msgs

    def run(
        self,
        start_turn: int = 1,
        to_turn: int | None = None,
        prior_steps: list[ReplayStep] | None = None,
        chunks_cb=None,
    ):
        """Yield one ReplayStep per attempt-turn (streaming via provider).

        `chunks_cb(turn, text)` receives each delta (UI streaming). A
        KeyboardInterrupt propagates immediately; already yielded (written)
        turns above remain the checkpoint. Failed turns yield with
        `error` set and are visible in the log but do not count as progress.
        """
        prior: list[ReplayStep] = list(prior_steps or [])
        limit = self.total_turns() if to_turn is None else min(to_turn, self.total_turns())
        for idx in range(start_turn - 1, limit):
            ctx = self._context(idx, prior)
            t0 = time.time()
            out_parts: list[str] = []
            err: str | None = None
            try:
                stream = self.provider.chat(messages=ctx, model=self.model, stream=True)
                for text in stream:
                    out_parts.append(text)
                    if chunks_cb is not None:
                        try:
                            chunks_cb(idx + 1, text)
                        except Exception:
                            pass
            except KeyboardInterrupt:
                raise
            except Exception as e:
                err = f"{type(e).__name__}: {e}"[:300]

            meta = getattr(self.provider, "last_meta", {}) or {}
            step = ReplayStep(
                turn=idx + 1,
                msg_id=self.turns[idx]["id"],
                branch=self.branch_id,
                provider=meta.get("provider") or getattr(self.provider, "id", None),
                model=meta.get("model") or self.model,
                prompt_tokens=(meta.get("tokens") or {}).get("in"),
                completion_tokens=(meta.get("tokens") or {}).get("out"),
                latency_ms=meta.get("latency_ms") or int((time.time() - t0) * 1000),
                output="".join(out_parts),
                error=err,
                resumed=start_turn > 1,
                ts=int(time.time() * 1000),
            )
            prior.append(step)
            yield step


def replay_to_log(
    session_path,
    provider,
    log_path,
    branch_id: str | None = None,
    model: str | None = None,
    context_mode: str = "replayed",
    to_turn: int | None = None,
    append: bool = True,
    force_from: int | None = None,
) -> dict:
    """Scenario/CLI driver: replay a session file, appending to the log.

    Resume: when `append` and the log already contains completed turns, the
    run continues after the last successful turn (its outputs feed context).
    Failed turns are retried from scratch by the resume logic (dedup by turn).
    """
    s = (
        session_mod.load_session(session_path)
        if isinstance(session_path, (str, os.PathLike))
        else session_path
    )
    engine = ReplayEngine(s, provider, branch_id=branch_id, model=model, context_mode=context_mode)
    total = engine.total_turns()
    limit = total if to_turn is None else min(to_turn, total)

    rows = load_replay_log(log_path)
    ok_rows = [r for r in rows if "turn" in r and not r.get("error")]
    ok_turns = {int(r["turn"]) for r in ok_rows}
    prior_steps: list[ReplayStep] = []
    start = 1
    if append and ok_turns:
        # resume from the FIRST missing/failed turn (holes are retried),
        # not from "number of completed rows + 1"
        missing = [t for t in range(1, limit + 1) if t not in ok_turns]
        start = force_from or (missing[0] if missing else limit + 1)
        start = max(1, min(start, limit + 1))
        prior_steps = [ReplayStep.from_row(r) for r in ok_rows if int(r["turn"]) < start]
    if start > limit:  # nothing left to do
        return {
            "total_turns": total,
            "completed_turns": len([r for r in ok_rows if int(r["turn"]) <= limit]),
            "resumed_from": None,
            "skipped": "already complete",
            "branch": engine.branch_id,
            "model": engine.model,
            "context_mode": context_mode,
            "log": str(log_path),
        }

    with open(log_path, "a", encoding="utf-8") as f:
        # progress = highest turn such that every turn in [start..turn] is ok
        progress = start - 1
        for step in engine.run(start_turn=start, prior_steps=prior_steps, to_turn=limit):
            f.write(step.to_json() + "\n")
            f.flush()
            if not step.error and step.turn == progress + 1:
                progress = step.turn
        if progress >= total:  # whole-branch replay finished (consecutive!)
            f.write(
                json.dumps(
                    {
                        "event": "replay_complete",
                        "branch": engine.branch_id,
                        "turns": total,
                        "ts": int(time.time() * 1000),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return {
        "total_turns": total,
        "completed_turns": progress,
        "resumed_from": start if start > 1 else None,
        "branch": engine.branch_id,
        "model": engine.model,
        "context_mode": context_mode,
        "log": str(log_path),
    }
