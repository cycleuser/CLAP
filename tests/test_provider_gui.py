"""P3 tests: config extension, provider policy plumbing (offscreen GUI)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtWidgets import QApplication  # noqa: E402


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


def test_config_backward_compatible_extension(tmp_path, monkeypatch):
    """Older config.json (no F6 fields) loads; missing keys take defaults."""
    from clap.utils import config as C

    monkeypatch.setattr(C.CLAPConfig, "get_config_file", staticmethod(lambda: tmp_path / "config.json"))
    (tmp_path / "config.json").write_text(
        '{"chat_model": "old-model", "embed_model": "emb", "font_size": 12}',
        encoding="utf-8",
    )
    cfg = C.load_config()
    assert cfg.chat_model == "old-model"
    assert cfg.privacy_mode is True  # F6 default
    assert cfg.routing == "manual"
    assert cfg.online_provider == ""


def test_resolve_selection_local_default(window):
    window.config.privacy_mode = True
    provider, pname, model, router = window.resolve_model_selection("deepseek:deepseek-chat")
    assert provider is None and router is None
    assert pname == "ollama-local" and model == "deepseek:deepseek-chat"


def test_resolve_selection_online_by_prefix(window):
    window.config.privacy_mode = False
    window.config.routing = "manual"
    if window.session is None:
        window.session = window._make_session()
    provider, pname, model, router = window.resolve_model_selection("deepseek:deepseek-chat")
    assert pname == "deepseek" and model == "deepseek-chat"
    assert provider is not None and router is None
    # privacy on: online selection collapses to local
    window.config.privacy_mode = True
    p2, n2, m2, r2 = window.resolve_model_selection("deepseek:deepseek-chat")
    assert n2 == "ollama-local"


def test_local_model_selection_never_wraps(window):
    window.config.privacy_mode = False
    provider, pname, model, router = window.resolve_model_selection("gemma3:4b")
    assert pname == "ollama-local" and provider is None and model == "gemma3:4b"


def test_settings_dialog_providers_tab_offscreen(qapp):
    from clap.gui.settings_dialog import SettingsDialog

    dlg = SettingsDialog()
    assert dlg.privacy_check.isChecked() is True
    assert dlg.routing_combo.currentText() == "manual"
    # the provider table must not contain raw key material
    rows = getattr(dlg, "_providers_rows", [])
    for r in rows:
        mk = r["masked_key"]
        assert mk == "" or mk == "<none>" or mk.startswith("***len")
    dlg.close()


def test_provider_channel_meta_capture(qapp, window):
    """ChatThread consumes a provider-channel stream and records real meta."""
    from clap.core.chat_thread import ChatThread

    class Local:
        id = "fake-online"

        def __init__(self):
            self.last_meta = {}

        def chat(self, messages, model, stream=True):
            self.last_meta = {"provider": "fake-online", "model": model, "latency_ms": 7,
                              "tokens": {"in": 5, "out": 2}}
            return iter(["你", "好"])

    window.config.privacy_mode = False
    if window.session is None:
        window.session = window._make_session()
    window.session["manifest"]["privacy_mode"] = False
    window.session["manifest"]["routing"] = "manual"
    from clap.core import session as session_mod

    session_mod.append_message(window.session, "user", "hi", parent=None, branch_id=window.active_branch)
    window.session["manifest"]["chat_model"] = "m"
    window._refresh_view()
    api_messages = [{"role": "user", "content": "hi"}]
    prov = Local()
    th = ChatThread(messages=api_messages, model="fake-model", provider=prov, provider_name="fake-online")
    th._process_chat()  # direct call (no event loop needed for meta capture)
    assert th.last_meta["provider"] == "fake-online"
    assert th.last_meta["tokens"] == {"in": 5, "out": 2}
    assert th.last_meta["latency_ms"] == 7


def test_no_secrets_in_provider_cli():
    """`clap providers` output carries no plaintext credentials."""
    import io
    from contextlib import redirect_stdout

    from clap.cli.main import providers_command

    buf = io.StringIO()
    with redirect_stdout(buf):
        providers_command(None)
    out = buf.getvalue()
    import re

    assert not re.search(r"sk-[A-Za-z0-9]{10,}", out)
    assert "***" in out
    assert "Bearer " not in out
