"""Main window for CLAP with PDF reader and chat interface."""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QAction, QKeySequence
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


class MainWindow(QMainWindow):
    """Main window with PDF reader (left) and chat (right)."""

    def __init__(self):
        super().__init__()
        self.config = load_config()
        self.chat_thread: ChatThread | None = None
        self.knowledge_base: KnowledgeBase | None = None
        self.messages: list[dict] = []
        self.current_file: str = ""
        self.current_response: str = ""
        self.response_browser: QTextBrowser | None = None
        self.history_path = Path.home() / ".clap" / "history.json"

        self.setWindowTitle("CLAP")
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

        self.doc_label = QLabel("Open a document to view")
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

        history_label = QLabel("Chat History")
        layout.addWidget(history_label)

        self.history_list = QListWidget()
        self.history_list.setMaximumHeight(120)
        self.history_list.itemClicked.connect(self._load_conversation)
        layout.addWidget(self.history_list)

        history_btn_layout = QHBoxLayout()
        self.new_chat_btn = QPushButton("New Chat")
        self.new_chat_btn.clicked.connect(self.new_chat)
        self.delete_chat_btn = QPushButton("Delete")
        self.delete_chat_btn.clicked.connect(self._delete_conversation)
        history_btn_layout.addWidget(self.new_chat_btn)
        history_btn_layout.addWidget(self.delete_chat_btn)
        layout.addLayout(history_btn_layout)

        chat_label = QLabel("Conversation")
        layout.addWidget(chat_label)

        self.chat_browser = QTextBrowser()
        self.chat_browser.setOpenExternalLinks(True)
        layout.addWidget(self.chat_browser, 1)

        input_layout = QHBoxLayout()
        self.input_edit = QPlainTextEdit()
        self.input_edit.setPlaceholderText("Type your message here...")
        self.input_edit.setMaximumHeight(60)
        input_layout.addWidget(self.input_edit, 1)

        btn_layout = QVBoxLayout()
        self.send_btn = QPushButton("Send")
        self.send_btn.clicked.connect(self.send_message)
        self.send_btn.setDefault(True)
        btn_layout.addWidget(self.send_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_chat)
        btn_layout.addWidget(self.clear_btn)
        input_layout.addLayout(btn_layout)

        layout.addLayout(input_layout)
        return panel

    def setup_toolbar(self):
        """Setup toolbar."""
        toolbar = QToolBar("Main Toolbar")
        self.addToolBar(toolbar)

        open_action = QAction("Open Document", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.open_document)
        toolbar.addAction(open_action)

        toolbar.addSeparator()

        toolbar.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(150)
        self.refresh_models()
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        toolbar.addWidget(self.model_combo)

        toolbar.addSeparator()

        settings_action = QAction("Settings", self)
        settings_action.setShortcut(QKeySequence("Ctrl+,"))
        settings_action.triggered.connect(self.show_settings)
        toolbar.addAction(settings_action)

        toolbar.addSeparator()

        self.kb_label = QLabel("KB: 0 chunks")
        toolbar.addWidget(self.kb_label)

    def refresh_models(self):
        """Refresh model list."""
        current = self.model_combo.currentText()
        self.model_combo.clear()
        models = get_available_models()
        self.model_combo.addItems(models)
        if self.config.chat_model in models:
            self.model_combo.setCurrentText(self.config.chat_model)
        elif current in models:
            self.model_combo.setCurrentText(current)

    def on_model_changed(self, model: str):
        """Handle model change."""
        if model:
            self.config.chat_model = model
            save_config(self.config)
            self.update_status()

    def update_status(self):
        """Update status bar."""
        model = self.model_combo.currentText() or self.config.chat_model
        doc = os.path.basename(self.current_file) if self.current_file else "No document"
        self.status_bar.showMessage(f"Model: {model} | Document: {doc}")

    def show_settings(self):
        """Show settings dialog."""
        dialog = SettingsDialog(self)
        if dialog.exec():
            self.config = dialog.get_config()
            save_config(self.config)
            self.refresh_models()
            self.update_status()

    def open_document(self):
        """Open a document."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Document", "", "Documents (*.pdf *.doc *.docx *.txt);;All Files (*)"
        )
        if path:
            self.load_document(path)

    def load_document(self, path: str):
        """Load a document."""
        self.current_file = path
        self.doc_label.hide()

        if path.lower().endswith(".pdf"):
            self.pdf_viewer.load(QUrl.fromLocalFile(path))
        else:
            self.doc_label.setText(os.path.basename(path))
            self.doc_label.show()

        self.status_bar.showMessage("Indexing document...")
        self.knowledge_base = KnowledgeBase(
            persist_directory=self.config.persist_directory, embedding_model=self.config.embed_model
        )
        result = self.knowledge_base.index_document(
            path, self.config.chunk_size, self.config.chunk_overlap
        )

        if result.get("success"):
            chunks = result.get("chunks", 0)
            self.kb_label.setText(f"KB: {chunks} chunks")
            self.status_bar.showMessage(f"Loaded: {os.path.basename(path)} ({chunks} chunks)")
        else:
            self.status_bar.showMessage(f"Error: {result.get('error', 'Unknown')}")

    def new_chat(self):
        """Start a new chat session."""
        self.save_history()
        self.messages.clear()
        self.current_response = ""
        self.chat_browser.clear()
        self.status_bar.showMessage("New chat started")

    def clear_chat(self):
        """Clear current chat display."""
        self.chat_browser.clear()

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
            role_display = "You" if role == "user" else "Assistant"
            html += f"<p><b>{role_display}:</b></p>{self._markdown(content)}<hr>"
        self.chat_browser.setHtml(html)

    def send_message(self):
        """Send message to LLM."""
        text = self.input_edit.toPlainText().strip()
        if not text:
            return

        model = self.model_combo.currentText()
        if not model:
            QMessageBox.warning(self, "Error", "Please select a model.")
            return

        self.messages.append({"role": "user", "content": text})
        self.input_edit.clear()
        self._render_messages()

        self.current_response = ""
        self.send_btn.setEnabled(False)
        self.status_bar.showMessage("Thinking...")

        self.chat_thread = ChatThread(
            messages=self.messages,
            model=model,
            path=self.current_file,
            embed_model=self.config.embed_model,
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
            role_display = "You" if role == "user" else "Assistant"
            html += f"<p><b>{role_display}:</b></p>{self._markdown(content)}<hr>"
        self.chat_browser.setHtml(html)

        scrollbar = self.chat_browser.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_chat_finished(self):
        """Handle chat completion."""
        self.send_btn.setEnabled(True)
        if self.current_response:
            self.messages.append({"role": "assistant", "content": self.current_response})
        self.current_response = ""
        self.status_bar.showMessage("Ready")
        self.save_history()

    def save_history(self):
        """Save chat history to file."""
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
            preview = session.get("preview", "Chat")
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
                self.messages = session.get("messages", [])
                self.current_file = session.get("document", "")
                model = session.get("model", "")
                if model:
                    self.config.chat_model = model
                    self.model_combo.setCurrentText(model)

                self._render_messages()
                self.doc_label.setText(
                    os.path.basename(self.current_file) if self.current_file else "No document"
                )
                self.update_status()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load conversation: {e}")

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
