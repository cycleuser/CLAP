"""Main window for CLAP with PDF reader and chat interface."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAction, QKeySequence, QPixmap
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTextBrowser,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from clap.core.chat_thread import ChatThread
from clap.core.knowledge_base import KnowledgeBase
from clap.gui.settings_dialog import SettingsDialog
from clap.utils.config import get_available_models, is_first_run, load_config, save_config
from clap.utils.i18n import set_language, t


def _library_index_snapshot() -> list:
    """History-list entries from the central session library (F1)."""
    try:
        from clap.core.sharing import list_library

        entries = list_library()
        return [
            {
                "timestamp": e.get("adopted", ""),
                "preview": e.get("name", ""),
                "document": "",
                "model": e.get("chat_model", ""),
                "language": "",
                "messages": [],
                "file": e.get("file", ""),
            }
            for e in entries[:20]
        ]
    except Exception:
        return []


class MainWindow(QMainWindow):
    """Main window with PDF reader (left) and chat (right)."""

    def __init__(self):
        super().__init__()
        self.config = load_config()
        set_language(self.config.language)

        self.chat_thread: ChatThread | None = None
        self.knowledge_base: KnowledgeBase | None = None
        self.messages: list[dict] = []
        self.current_file: str = ""
        self.current_response: str = ""
        self.response_browser: QTextBrowser | None = None
        self.history_path = Path.home() / ".clap" / "history.json"
        # session v2 state (source of truth); `messages` is the linear view
        self.session: dict | None = None
        self.active_branch: str | None = None

        self.setWindowTitle(t("app_name"))
        self.setMinimumSize(1200, 800)

        self.setup_toolbar()
        self.setup_ui()
        self.load_history()

    def setup_ui(self):
        """Setup the main UI."""
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(4, 4, 4, 4)

        main_splitter = QSplitter(Qt.Horizontal)

        left_panel = self._create_document_panel()
        main_splitter.addWidget(left_panel)

        right_panel = self._create_chat_panel()
        main_splitter.addWidget(right_panel)

        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([600, 600])

        layout.addWidget(main_splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.update_status()

    def _create_document_panel(self) -> QWidget:
        """Create the document reader panel (left side)."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.pdf_viewer = QWebEngineView()
        self.pdf_viewer.settings().setAttribute(
            self.pdf_viewer.settings().WebAttribute.PluginsEnabled, True
        )
        self.pdf_viewer.settings().setAttribute(
            self.pdf_viewer.settings().WebAttribute.PdfViewerEnabled, True
        )
        layout.addWidget(self.pdf_viewer, 1)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)
        self.image_label.hide()
        layout.addWidget(self.image_label, 1)

        self.doc_label = QLabel(t("no_document"))
        self.doc_label.setAlignment(Qt.AlignCenter)
        self.doc_label.setWordWrap(True)
        layout.addWidget(self.doc_label)

        return panel

    def _create_chat_panel(self) -> QWidget:
        """Create the chat panel (right side)."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.history_label = QLabel(t("chat_history"))
        layout.addWidget(self.history_label)

        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(120)
        self.history_list.itemClicked.connect(self._load_conversation)
        layout.addWidget(self.history_list)

        history_btn_layout = QHBoxLayout()
        self.new_chat_btn = QPushButton(t("new_chat"))
        self.new_chat_btn.clicked.connect(self.new_chat)
        self.delete_chat_btn = QPushButton(t("delete"))
        self.delete_chat_btn.clicked.connect(self._delete_conversation)
        history_btn_layout.addWidget(self.new_chat_btn)
        history_btn_layout.addWidget(self.delete_chat_btn)
        layout.addLayout(history_btn_layout)

        self.chat_label = QLabel(t("conversation"))
        layout.addWidget(self.chat_label)

        self.chat_browser = QTextBrowser()
        self.chat_browser.setOpenExternalLinks(True)
        layout.addWidget(self.chat_browser, 1)

        input_layout = QHBoxLayout()
        self.input_edit = QPlainTextEdit()
        self.input_edit.setPlaceholderText(t("type_message"))
        self.input_edit.setMaximumHeight(60)
        input_layout.addWidget(self.input_edit, 1)

        btn_layout = QVBoxLayout()
        self.send_btn = QPushButton(t("send"))
        self.send_btn.clicked.connect(self.send_message)
        self.send_btn.setDefault(True)
        btn_layout.addWidget(self.send_btn)

        self.clear_btn = QPushButton(t("clear"))
        self.clear_btn.clicked.connect(self.clear_chat)
        btn_layout.addWidget(self.clear_btn)
        input_layout.addLayout(btn_layout)

        layout.addLayout(input_layout)
        return panel

    def setup_toolbar(self):
        """Setup toolbar."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        self.open_action = QAction(t("open_document"), self)
        self.open_action.setShortcut(QKeySequence("Ctrl+O"))
        self.open_action.triggered.connect(self.open_document)
        toolbar.addAction(self.open_action)

        toolbar.addSeparator()

        self.save_chat_action = QAction(t("save_chat"), self)
        self.save_chat_action.setShortcut(QKeySequence("Ctrl+S"))
        self.save_chat_action.triggered.connect(self.save_chat_to_file)
        toolbar.addAction(self.save_chat_action)

        self.load_chat_action = QAction(t("load_chat"), self)
        self.load_chat_action.setShortcut(QKeySequence("Ctrl+L"))
        self.load_chat_action.triggered.connect(self.load_chat_from_file)
        toolbar.addAction(self.load_chat_action)

        self.export_action = QAction(t("export_md"), self)
        self.export_action.triggered.connect(self.export_chat_to_markdown)
        toolbar.addAction(self.export_action)

        self.share_action = QAction(t("share_chat"), self)
        self.share_action.triggered.connect(self.share_session_package)
        toolbar.addAction(self.share_action)

        self.adopt_action = QAction(t("adopt_chat"), self)
        self.adopt_action.triggered.connect(self.adopt_session_package)
        toolbar.addAction(self.adopt_action)

        toolbar.addSeparator()

        self.model_label = QLabel(f"{t('model')}:")
        toolbar.addWidget(self.model_label)
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(150)
        self.refresh_models()
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        toolbar.addWidget(self.model_combo)

        toolbar.addSeparator()

        self.settings_action = QAction(t("settings"), self)
        self.settings_action.setShortcut(QKeySequence("Ctrl+,"))
        self.settings_action.triggered.connect(self.show_settings)
        toolbar.addAction(self.settings_action)

        toolbar.addSeparator()

        self.kb_label = QLabel(f"{t('kb')}: 0 {t('chunks')}")
        toolbar.addWidget(self.kb_label)

    def refresh_models(self):
        """Refresh model list: local ollama models + available online provider
        models grouped with a `provider:model` prefix (F6)."""
        current = self.model_combo.currentText()
        self.model_combo.clear()
        local_models = get_available_models()
        self.model_combo.addItems(local_models)

        online_items: list[str] = []
        try:
            from clap.core.providers import discover_providers

            for pid, prov in discover_providers().items():
                if pid == "ollama-local":
                    continue
                # privacy default: online models shown but flagged; chat-time
                # routing enforces privacy_mode
                if self.config.privacy_mode:
                    continue
                if prov.is_available():
                    for m in prov.list_models()[:12]:
                        online_items.append(f"{pid}:{m}")
        except Exception:
            pass

        if online_items:

            self.model_combo.insertSeparator(self.model_combo.count())
            self.model_combo.addItems(online_items)

        if self.config.chat_model in local_models:
            self.model_combo.setCurrentText(self.config.chat_model)
        elif current and self.model_combo.findText(current) >= 0:
            self.model_combo.setCurrentText(current)

    def resolve_model_selection(self, selection: str):
        """Split a combo selection into (ChatThread kwargs, model label).

        `provider:model` selects an online provider; bare names stay local
        (privacy_mode enforced here: online requests are rewritten to local).
        """
        from clap.core.providers import (
            FallbackRouter,
            OllamaProvider,
            discover_providers,
        )

        provider_name = "ollama-local"
        provider = None
        model = selection
        router = None
        if self.config.privacy_mode:
            provider_name, model = "ollama-local", selection
        else:
            online_ids = set()
            try:
                online_ids = {
                    pid
                    for pid, prov in discover_providers().items()
                    if pid != "ollama-local"
                }
            except Exception:
                pass
            head = selection.split(":", 1)[0] if ":" in selection else ""
            if head in online_ids and head not in ("ollama",):  # ollama tags contain colons too
                pid, _, model = selection.partition(":")
                try:
                    registry = discover_providers()
                    prov = registry.get(pid)
                    if prov is not None:
                        provider = prov
                        provider_name = pid
                        local = registry.get("ollama-local", OllamaProvider())
                        if self.config.routing == "fallback":
                            router = FallbackRouter(prov, local, privacy_mode=False)
                except Exception:
                    provider, provider_name, model = None, "ollama-local", selection
        return provider, provider_name, model, router

    def on_model_changed(self, model: str):
        """Handle model change."""
        if model:
            self.config.chat_model = model
            save_config(self.config)
            self.update_status()

    def update_status(self):
        """Update status bar."""
        model = self.model_combo.currentText() or self.config.chat_model
        doc = os.path.basename(self.current_file) if self.current_file else t("no_document")
        self.status_bar.showMessage(f"{t('model')}: {model} | {t('open_document')}: {doc}")

    def show_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self)
        if dialog.exec():
            self.config = dialog.get_config()
            save_config(self.config)
            set_language(self.config.language)
            self._retranslate_ui()
            self.refresh_models()
            self.update_status()

    def _retranslate_ui(self):
        """Retranslate UI after language change."""
        self.setWindowTitle(t("app_name"))

        self.open_action.setText(t("open_document"))
        self.save_chat_action.setText(t("save_chat"))
        self.load_chat_action.setText(t("load_chat"))
        self.export_action.setText(t("export_md"))
        self.settings_action.setText(t("settings"))

        self.model_label.setText(f"{t('model')}:")
        chunks = (
            self.kb_label.text().split(":")[-1].strip()
            if ":" in self.kb_label.text()
            else f"0 {t('chunks')}"
        )
        self.kb_label.setText(f"{t('kb')}: {chunks}")

        self.history_label.setText(t("chat_history"))
        self.chat_label.setText(t("conversation"))

        self.new_chat_btn.setText(t("new_chat"))
        self.delete_chat_btn.setText(t("delete"))
        self.send_btn.setText(t("send"))
        self.clear_btn.setText(t("clear"))

        self.input_edit.setPlaceholderText(t("type_message"))
        self.doc_label.setText(t("no_document"))

        self.update_status()

    def open_document(self):
        """Open a document."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            t("open_document"),
            "",
            "Documents (*.pdf *.doc *.docx *.md *.markdown *.txt);;"
            "Images (*.png *.jpg *.jpeg *.gif *.bmp *.webp *.svg);;"
            "All Files (*)",
        )
        if path:
            self.load_document(path)

    def load_document(self, path: str):
        """Load a document."""
        self.current_file = path
        self._hide_all_viewers()

        if path.lower().endswith(".pdf"):
            self.pdf_viewer.show()
            self.pdf_viewer.load(QUrl.fromLocalFile(path))
            self._index_document(path)
        elif path.lower().endswith((".md", ".markdown")):
            self.pdf_viewer.show()
            self._load_markdown_viewer(path)
            self._index_document(path)
        elif path.lower().endswith(".svg"):
            self.pdf_viewer.show()
            self._load_svg_viewer(path)
            self.status_bar.showMessage(f"{t('loaded')}: {os.path.basename(path)}")
        elif path.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
            self._load_image_viewer(path)
            self.status_bar.showMessage(f"{t('loaded')}: {os.path.basename(path)}")
        else:
            self.doc_label.setText(os.path.basename(path))
            self.doc_label.show()
            self._index_document(path)

    def _hide_all_viewers(self):
        """Hide all document viewers."""
        self.pdf_viewer.hide()
        self.image_label.hide()
        self.doc_label.hide()

    def _load_image_viewer(self, path: str):
        """Load and display an image file."""
        pixmap = QPixmap(path)
        if pixmap.isNull():
            self.doc_label.setText(f"{t('error')}: {os.path.basename(path)}")
            self.doc_label.show()
            return

        self.image_label.setPixmap(pixmap)
        self.image_label.show()

    def _index_document(self, path: str):
        """Index a document into knowledge base."""
        self.status_bar.showMessage(t("indexing"))
        self.knowledge_base = KnowledgeBase(
            persist_directory=self.config.persist_directory, embedding_model=self.config.embed_model
        )
        result = self.knowledge_base.index_document(
            path, self.config.chunk_size, self.config.chunk_overlap
        )

        if result.get("success"):
            chunks = result.get("chunks", 0)
            self.kb_label.setText(f"{t('kb')}: {chunks} {t('chunks')}")
            self.status_bar.showMessage(
                f"{t('loaded')}: {os.path.basename(path)} ({chunks} {t('chunks')})"
            )
        else:
            self.status_bar.showMessage(f"{t('error')}: {result.get('error', 'Unknown')}")

    def _load_markdown_viewer(self, path: str):
        """Load and display markdown file as HTML."""
        try:
            with open(path, encoding="utf-8") as f:
                content = f.read()
            html = self._markdown(content)
            self.pdf_viewer.setHtml(html)
        except Exception:
            self.doc_label.setText(os.path.basename(path))
            self.doc_label.show()

    def _load_svg_viewer(self, path: str):
        """Load and display SVG file."""
        try:
            with open(path, encoding="utf-8") as f:
                svg_content = f.read()
            html = f"<!DOCTYPE html><html><head><style>body{{margin:0;padding:20px;display:flex;justify-content:center;align-items:center;min-height:100vh;}}</style></head><body>{svg_content}</body></html>"
            self.pdf_viewer.setHtml(html)
        except Exception:
            self.doc_label.setText(os.path.basename(path))
            self.doc_label.show()

    def new_chat(self):
        """Start a new chat session."""
        self.save_history()
        self.session = None
        self.active_branch = None
        self.messages.clear()
        self.current_response = ""
        self.chat_browser.clear()
        self.status_bar.showMessage(t("new_chat_started"))

    def clear_chat(self):
        """Clear current chat display."""
        self.chat_browser.clear()

    def save_chat_to_file(self):
        """Save current conversation to a file (thin shell over session.py)."""
        if not self.messages:
            QMessageBox.information(self, t("info"), t("no_conversation"))
            return

        path, _ = QFileDialog.getSaveFileName(
            self, t("save_conversation"), "", "CLAP Chat (*.clap);;JSON (*.json);;All Files (*)"
        )
        if not path:
            return

        if not path.endswith((".clap", ".json")):
            path += ".clap"

        from clap.core import session as session_mod

        try:
            if self.session is None:
                self.session = self._make_session()
                parent = None
                for m in self.messages:
                    node = session_mod.append_message(
                        self.session, m.get("role", "user"), m.get("content", ""),
                        parent=parent, branch_id=self.active_branch,
                        model=m.get("model"), provider=m.get("provider"),
                        latency_ms=m.get("latency_ms"), tokens=m.get("tokens"),
                        error=m.get("error"),
                    )
                    parent = node["id"]
                session_mod.ensure_main_branch(self.session)
            session_mod.save_session(self.session, path)
            self.status_bar.showMessage(f"{t('saved')}: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, t("error"), f"{t('failed_save')}: {e}")

    def load_chat_from_file(self):
        """Load conversation from a file (thin shell over session.py; v1 is
        migrated to v2 transparently)."""
        path, _ = QFileDialog.getOpenFileName(
            self, t("load_conversation"), "", "CLAP Chat (*.clap);;JSON (*.json);;All Files (*)"
        )
        if not path:
            return

        try:
            self._apply_loaded_session(path)
            self.status_bar.showMessage(f"{t('loaded')}: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, t("error"), f"{t('failed_load')}: {e}")

    def _make_session(self) -> dict:
        """Create a fresh v2 session from current GUI config/state."""
        from clap.core import session as session_mod

        s = session_mod.new_session(
            chat_model=self.config.chat_model,
            embed_model=self.config.embed_model,
            document=self.current_file,
            language=self.config.language,
        )
        session_mod.ensure_main_branch(s)
        self.active_branch = session_mod.main_branch(s).get("id")
        return s

    def _refresh_view(self):
        """Refresh the linear view to the active branch's own messages."""
        from clap.core import session as session_mod

        if self.session is None:
            return
        bid = self.active_branch or session_mod.main_branch(self.session).get("id")
        self.active_branch = bid
        self.messages = session_mod.branch_path(self.session, bid)

    # ---------------------------------------------------------------- F1

    def share_session_package(self):
        """Export the current/selected .clap as a shareable package."""
        path, _ = QFileDialog.getOpenFileName(
            self, t("load_conversation"), "", "CLAP Chat (*.clap);;All Files (*)"
        )
        if not path:
            return
        try:
            from clap.core import sharing

            pkg = sharing.pack(path)
            self.status_bar.showMessage(f"{t('shared')}: {pkg.name}")
        except Exception as e:
            QMessageBox.warning(self, t("error"), f"{t('share_failed')}: {e}")

    def adopt_session_package(self):
        """Adopt a .clapshare package into the local session library."""
        path, _ = QFileDialog.getOpenFileName(
            self, t("adopt_chat"), "", t("share_files") + ";;All Files (*)"
        )
        if not path:
            return
        try:
            from clap.core import sharing

            r = sharing.adopt(path)
            self.status_bar.showMessage(f"{t('adopted')}: {r['session_id']}")
            self._apply_loaded_session(r["path"])
        except Exception as e:
            QMessageBox.warning(self, t("error"), f"{t('adopt_failed')}: {e}")

    def _apply_loaded_session(self, path: str):
        """Common body of load_chat_from_file for a concrete path."""
        from clap.core import session as session_mod

        s = session_mod.load_session(path)
        self.session = s
        main = session_mod.main_branch(s)
        self.active_branch = main.get("id")
        self.messages = session_mod.branch_path(s, self.active_branch or "b1")

        try:
            doc = s.get("manifest", {}).get("document") or {}
            doc_path = doc.get("path", "")
            self.current_file = (
                doc_path if doc_path and os.path.exists(doc_path) and doc_path != "<stripped-for-share>" else ""
            )
        except Exception:
            self.current_file = ""

        model = s.get("manifest", {}).get("chat_model", "")
        if model:
            self.config.chat_model = model
            self.model_combo.setCurrentText(model)

        lang = s.get("language", "")
        if lang:
            self.config.language = lang
            set_language(lang)

        self._update_history_list(_library_index_snapshot())
        self._render_messages()
        self.doc_label.setText(
            os.path.basename(self.current_file) if self.current_file else t("no_document")
        )
        self.update_status()

    def export_chat_to_markdown(self):
        """Export conversation to a markdown file."""
        if not self.messages:
            QMessageBox.information(self, t("info"), t("no_conversation_export"))
            return

        path, _ = QFileDialog.getSaveFileName(
            self, t("export_markdown"), "", "Markdown (*.md);;Text (*.txt);;All Files (*)"
        )
        if not path:
            return

        if not path.endswith((".md", ".txt")):
            path += ".md"

        lines = [
            f"# {t('app_name')} {t('conversation')}",
            "",
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"{t('model')}: {self.config.chat_model}",
            "",
        ]
        if self.current_file:
            lines.append(f"{t('open_document')}: {os.path.basename(self.current_file)}")
            lines.append("")

        lines.append("---")
        lines.append("")

        for msg in self.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            role_display = "**You**" if role == "user" else "**Assistant**"
            lines.append(f"{role_display}:")
            lines.append("")
            lines.append(content)
            lines.append("")
            lines.append("---")
            lines.append("")

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            self.status_bar.showMessage(f"{t('exported')}: {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, t("error"), f"{t('failed_export')}: {e}")

    def _markdown(self, text: str) -> str:
        """Convert markdown to HTML."""
        try:
            import mistune

            return mistune.html(text)
        except Exception:
            return text.replace("\n", "<br>")

    def _render_messages(self):
        """Render all messages in the chat browser."""
        html = ""
        for msg in self.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            role_display = t("you") if role == "user" else t("assistant")
            html += f"<p><b>{role_display}:</b></p>{self._markdown(content)}<hr>"
        self.chat_browser.setHtml(html)

    def send_message(self):
        """Send message to LLM."""
        text = self.input_edit.toPlainText().strip()
        if not text:
            return

        model = self.model_combo.currentText()
        if not model:
            QMessageBox.warning(self, t("error"), t("select_model"))
            return

        from clap.core import session as session_mod

        provider, provider_name, model_label, router = self.resolve_model_selection(model)
        provider_obj = router if router is not None else provider

        if self.session is None:
            self.session = self._make_session()
        parent_id = self.messages[-1].get("id") if self.messages else None
        session_mod.append_message(
            self.session, "user", text, parent=parent_id, branch_id=self.active_branch,
            model=model_label, provider=provider_name,
        )
        self._refresh_view()
        self.input_edit.clear()
        self._render_messages()

        self.current_response = ""
        self.send_btn.setEnabled(False)
        self.status_bar.showMessage(t("thinking"))

        api_messages = [{"role": m.get("role"), "content": m.get("content")} for m in self.messages]
        self.chat_thread = ChatThread(
            messages=api_messages,
            model=model_label,
            path=self.current_file,
            embed_model=self.config.embed_model,
            provider=provider_obj,
            provider_name=provider_name,
        )
        self.chat_thread.new_text.connect(self.on_new_text)
        self.chat_thread.finished.connect(self.on_chat_finished)
        self.chat_thread.start()

    def on_new_text(self, text: str):
        """Handle new text from chat thread."""
        self.current_response += text
        temp_messages = self.messages + [{"role": "assistant", "content": self.current_response}]
        html = ""
        for msg in temp_messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            role_display = t("you") if role == "user" else t("assistant")
            html += f"<p><b>{role_display}:</b></p>{self._markdown(content)}<hr>"
        self.chat_browser.setHtml(html)

        scrollbar = self.chat_browser.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_chat_finished(self):
        """Handle chat completion."""
        self.send_btn.setEnabled(True)
        if self.current_response:
            from clap.core import session as session_mod

            meta = {}
            if self.chat_thread is not None:
                meta = dict(getattr(self.chat_thread, "last_meta", {}) or {})
            parent_id = self.messages[-1].get("id") if self.messages else None
            session_mod.append_message(
                self.session if self.session is not None else self._make_session(),
                "assistant",
                self.current_response,
                parent=parent_id,
                branch_id=self.active_branch,
                model=meta.get("model") or self.config.chat_model,
                provider=meta.get("provider") or "ollama-local",
                latency_ms=meta.get("latency_ms") if meta.get("latency_ms") is not None else None,
                tokens=meta.get("tokens"),
                error=meta.get("error"),
            )
            self._refresh_view()
        self.current_response = ""
        self.status_bar.showMessage(t("ready"))
        self.save_history()

    def save_history(self):
        """Save chat history to file + central session library (F1)."""
        if not self.messages:
            return

        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        history = []
        if self.history_path.exists():
            try:
                with open(self.history_path, encoding="utf-8") as f:
                    history = json.load(f)
            except Exception:
                pass

        session = {
            "timestamp": datetime.now().isoformat(),
            "messages": self.messages,
            "document": self.current_file,
            "model": self.config.chat_model,
            "language": self.config.language,
        }

        if self.messages:
            preview = self.messages[0].get("content", "")[:50]
            session["preview"] = preview

        existing = None
        for i, h in enumerate(history):
            if h.get("messages") and self.messages:
                if (
                    h["messages"][0].get("content") == self.messages[0].get("content")
                    if self.messages
                    else None
                ):
                    existing = i
                    break

        if existing is not None:
            history[existing] = session
        else:
            history.insert(0, session)

        with open(self.history_path, "w", encoding="utf-8") as f:
            json.dump(history[:50], f, ensure_ascii=False, indent=2)

        # central session library: register own session (best effort)
        try:
            from clap.core import session as session_mod
            from clap.core import sharing

            if self.session is None:
                self.session = self._make_session()
                parent = None
                for m in self.messages:
                    node = session_mod.append_message(
                        self.session, m.get("role", "user"), m.get("content", ""),
                        parent=parent, branch_id=self.active_branch,
                        model=m.get("model"), provider=m.get("provider"),
                        latency_ms=m.get("latency_ms"), tokens=m.get("tokens"),
                        error=m.get("error"),
                    )
                    parent = node["id"]
                session_mod.ensure_main_branch(self.session)
            # refresh the manifest's live chat model snapshot
            self.session["manifest"]["chat_model"] = self.session["manifest"].get("chat_model") or self.config.chat_model
            sharing.save_to_library(self.session)
            self._update_history_list(_library_index_snapshot() or history[:50])
            return
        except Exception:
            pass

        self._update_history_list(history)

    def load_history(self):
        """Load chat history from file."""
        if not self.history_path.exists():
            return

        try:
            with open(self.history_path, encoding="utf-8") as f:
                history = json.load(f)
            self._update_history_list(history)
        except Exception:
            pass

    def _update_history_list(self, history: list):
        """Update the history list widget."""
        self.history_list.clear()
        for i, session in enumerate(history[:20]):
            preview = session.get("preview", t("conversation"))
            timestamp = session.get("timestamp", "")[:10]
            self.history_list.addItem(f"{timestamp}: {preview}...")
            self.history_list.item(self.history_list.count() - 1).setData(Qt.UserRole, i)

    def _load_conversation(self, item):
        """Load a conversation from history."""
        idx = item.data(Qt.UserRole)
        if idx is None:
            return

        try:
            with open(self.history_path, encoding="utf-8") as f:
                history = json.load(f)

            if 0 <= idx < len(history):
                session = history[idx]
                # library entries carry a file: load the real session file
                if session.get("file") and Path(session["file"]).exists():
                    self._apply_loaded_session(session["file"])
                    self.status_bar.showMessage(f"{t('loaded')}: {os.path.basename(session['file'])}")
                    return
                self.session = None  # history.json is flat; rebuild on save
                self.active_branch = None
                self.messages = session.get("messages", [])
                self.current_file = session.get("document", "")
                model = session.get("model", "")
                if model:
                    self.config.chat_model = model
                    self.model_combo.setCurrentText(model)

                lang = session.get("language", "")
                if lang:
                    self.config.language = lang
                    set_language(lang)

                self._render_messages()
                self.doc_label.setText(
                    os.path.basename(self.current_file) if self.current_file else t("no_document")
                )
                self.update_status()
        except Exception as e:
            QMessageBox.warning(self, t("error"), f"{t('failed_load')}: {e}")

    def _delete_conversation(self):
        """Delete selected conversation from history."""
        item = self.history_list.currentItem()
        if not item:
            return

        idx = item.data(Qt.UserRole)
        if idx is None:
            return

        try:
            with open(self.history_path, encoding="utf-8") as f:
                history = json.load(f)

            if 0 <= idx < len(history):
                del history[idx]
                with open(self.history_path, "w", encoding="utf-8") as f:
                    json.dump(history, f, ensure_ascii=False, indent=2)
                self._update_history_list(history)
        except Exception:
            pass

    def closeEvent(self, event):
        """Handle window close."""
        self.save_history()
        if self.chat_thread and self.chat_thread.isRunning():
            self.chat_thread.terminate()
            self.chat_thread.wait()
        event.accept()


def main():
    """Main entry point for GUI."""
    app = QApplication(sys.argv)
    app.setApplicationName("CLAP")

    if is_first_run():
        dialog = SettingsDialog(first_run=True)
        if dialog.exec():
            save_config(dialog.get_config())
        else:
            sys.exit(0)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
