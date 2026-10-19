"""Offscreen GUI smoke tests (thin shells over clap.core.session).

The GUI cannot be run interactively here; these tests construct the
MainWindow on the offscreen platform and exercise save/load behaviour with
QFileDialog monkeypatched. No network calls are made.
"""

import json
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication, QFileDialog  # noqa: E402

from clap.core import session as S  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def window(qapp):
    from clap.gui.main_window import MainWindow

    w = MainWindow()
    yield w
    w.close()


def test_window_constructs_offscreen(window):
    assert window.session is None
    assert window.active_branch is None


def test_gui_save_then_load_roundtrip(tmp_path, window, monkeypatch):
    # seed a conversation as the GUI would have after send/finish
    from clap.core import session as session_mod

    window.session = window._make_session()
    m_user = session_mod.append_message(
        window.session, "user", "什么是兼语句？", parent=None, branch_id=window.active_branch,
        model=window.config.chat_model, provider="ollama-local",
    )
    session_mod.append_message(
        window.session, "assistant", "兼语句是主谓短语作宾语的句子。", parent=m_user["id"], branch_id=window.active_branch,
        model=window.config.chat_model, provider="ollama-local",
        latency_ms=1500, tokens={"in": 12, "out": 30},
    )
    window._refresh_view()
    assert len(window.messages) == 2

    p = tmp_path / "out.clap"
    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(p), "")))
    window.save_chat_to_file()

    s = json.loads(p.read_text(encoding="utf-8"))
    assert s["version"] == "2.0"
    dig = S.session_digest(json.loads(p.read_text(encoding="utf-8")))

    # load back through the thin shell
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(p), "")))
    window2_messages = None
    window.load_chat_from_file()
    window2_messages = window.messages
    assert [m["content"] for m in window2_messages] == [
        "什么是兼语句？",
        "兼语句是主谓短语作宾语的句子。",
    ]
    loaded = window.session
    assert S.session_digest(loaded) == dig  # byte-level field fidelity
    assert window.active_branch == S.main_branch(loaded).get("id")


def test_gui_loads_v1_file(tmp_path, window, monkeypatch):
    v1 = {
        "version": "1.0",
        "timestamp": "2026-10-19T10:00:00",
        "model": "gemma3:4b",
        "embed_model": "nomic-embed-text:latest",
        "document": "教案.docx",
        "messages": [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ],
        "language": "zh",
    }
    p = tmp_path / "legacy.clap"
    p.write_text(json.dumps(v1, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(p), "")))
    window.load_chat_from_file()
    assert window.session["version"] == "2.0"
    assert len(window.messages) == 2
    assert window.messages[0]["content"] == "q1"