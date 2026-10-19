"""P5 tests: timeline HTML (self-contained, fidelity), SVG export, annotations."""

import json
import os
import re

import pytest

from clap.core import session as S
from clap.core import timeline as T
from clap.gui import export_timeline_svg as EX


@pytest.fixture()
def annotated_session():
    s = S.new_session(chat_model="gemma3:4b", document="/tmp/local/教案.docx", language="zh")
    S.ensure_main_branch(s)
    m1 = S.append_message(s, "user", "什么是兼语句？", parent=None, branch_id="b1", model="gemma3:4b")
    m2 = S.append_message(
        s, "assistant", "兼语句是……", parent=m1["id"], branch_id="b1",
        model="gemma3:4b", provider="ollama-local", latency_ms=900, tokens={"in": 5, "out": 9},
    )
    S.fork_branch(s, m2["id"], name="提高", model="deepseek-chat")
    b2 = [b for b in s["branches"] if b["fork_at"]][0]
    u3 = S.append_branch_message(s, b2["id"], "user", "提高问1", model="deepseek-chat")
    S.add_annotation(s, m2["id"], "keypoint", tags=["兼语句"], note="考点", author="teacher")
    S.add_annotation(s, u3["id"], "question", tags=[], note="待讲评", author="teacher")
    return s


def test_timeline_html_fidelity(annotated_session):
    html = T.render_timeline_html(annotated_session)
    # contents present
    for frag in ("什么是兼语句？", "兼语句是……", "提高问1", "重点", "疑问", "考点", "待讲评"):
        assert frag in html
    # branch section and fork marker
    assert "fork" in html
    assert "提高" in html and "主线" in html
    # header facts
    assert annotated_session["manifest"]["session_id"] in html
    assert "备课" in html
    # no external links (self-contained)
    assert not re.search(r'href="http|src="http|@import|<script src', html)
    # template applied (wrap present) and no leftover marker
    assert "{{CONTENT}}" not in html


def test_timeline_null_rendering():
    s = S.new_session(chat_model="m", language="zh")
    S.ensure_main_branch(s)
    S.append_message(s, "user", "q", parent=None, branch_id="b1")  # no provenance
    html = T.render_timeline_html(s)
    assert html.count(T.NULL_MARK) >= 3  # model/provider/latency/tokens unknown → "—"


def test_svg_export(annotated_session, tmp_path):
    p = EX.write_timeline_svg(annotated_session, tmp_path / "tl.svg")
    data = (tmp_path / "tl.svg").read_text(encoding="utf-8")
    assert p.endswith("tl.svg")
    assert "<svg" in data
    assert data.index("<svg") < data.index("</svg>")
    # palette discipline: blue/grey tokens in the svg style
    assert ("#1d4ed8" in data) or ("#2563eb" in data) or ("#93c5fd" in data)


def test_distribution_png(annotated_session, tmp_path):
    p = EX.write_annotation_distribution(annotated_session, tmp_path / "dist.png")
    assert os.path.exists(p) and os.path.getsize(p) > 1000


def test_annotate_cli_roundtrip(annotated_session, tmp_path):
    f = tmp_path / "s.clap"
    S.save_session(annotated_session, f)
    rows = S.annotation_rows(S.load_session(f))
    assert len(rows) == 2
    assert set(rows[0].keys()) == set(S.ANNOTATION_CSV_FIELDS)
    csv_text = S.annotations_csv(S.load_session(f))
    header = csv_text.splitlines()[0]
    assert header == "session_id,turn_id,kind,tags,note,author,ts,model,latency_ms,tokens_out,branch"
    row0 = csv_text.splitlines()[1]
    assert "keypoint" in row0 and "b1" in row0


def test_annotation_json_export(annotated_session, tmp_path):
    f = tmp_path / "s.clap"
    S.save_session(annotated_session, f)
    out = S.export_annotations(S.load_session(f), tmp_path / "ann.json", fmt="json")
    rows = json.loads((tmp_path / "ann.json").read_text(encoding="utf-8"))
    assert len(rows) == 2 and out.endswith("ann.json")


def test_branch_tree_spec(annotated_session):
    spec = T.branch_tree_spec(annotated_session)
    assert len(spec["branches"]) == 2
    main = spec["branches"][0]
    forked = spec["branches"][1]
    assert main["fork_at"] is None and main["own_len"] == 2
    assert forked["fork_at"] == "m2" and forked["own_len"] == 1
    assert spec["annotations"] == 2


def test_gui_annotate_dialog_payload(qapp, annotated_session):
    from clap.core import session as session_mod
    from clap.gui.annotate_dialog import AnnotateDialog

    window_msgs = session_mod.branch_path(annotated_session, "b1")
    dlg = AnnotateDialog(window_msgs)
    dlg.msg_combo.setCurrentIndex(1)  # assistant message
    dlg.kind_combo.setCurrentIndex(2)  # generated
    dlg.tags_edit.setText("a, b")
    dlg.note_edit.setPlainText("n")
    payload = dlg.payload()
    assert payload["target"] == window_msgs[1]["id"]
    assert payload["kind"] == "generated"
    assert payload["tags"] == ["a", "b"]
