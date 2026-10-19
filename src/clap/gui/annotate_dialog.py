"""CLAP-NEXT F5: annotation dialog (choose a message, kind, tags, note)."""

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
)

from clap.core.session import ANNOTATION_KINDS, KIND_LABELS


class AnnotateDialog(QDialog):
    """Pick a message + annotation kind; returns the annotation payload."""

    def __init__(self, messages: list[dict], parent=None, session=None):
        super().__init__(parent)
        self.setWindowTitle("Annotate (F5)")
        self.setMinimumWidth(460)
        self.messages = [m for m in messages if m.get("id")]
        self.session = session

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.msg_combo = QComboBox()
        for m in self.messages:
            preview = (m.get("content") or "").replace("\n", " ")[:38]
            self.msg_combo.addItem(f"{m['id']} · {m.get('role', '?')} · {preview}", m["id"])
        form.addRow("Message:", self.msg_combo)

        self.kind_combo = QComboBox()
        for k in ANNOTATION_KINDS:
            self.kind_combo.addItem(f"{KIND_LABELS[k]} ({k})", k)
        form.addRow("Kind:", self.kind_combo)

        self.tags_edit = QLineEdit()
        form.addRow("Tags ( , ): ", self.tags_edit)

        self.note_edit = QPlainTextEdit()
        self.note_edit.setFixedHeight(64)
        form.addRow("Note:", self.note_edit)

        self.author_edit = QLineEdit("teacher")
        form.addRow("Author:", self.author_edit)

        layout.addLayout(form)
        info = QLabel("kind: keypoint=重点 question=疑问 generated=生成 reviewed=讲评 error_fix=纠错 knowledge_point=知识点")
        info.setStyleSheet("color:#6b7280;font-size:11px;")
        layout.addWidget(info)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def payload(self) -> dict | None:
        rid = self.msg_combo.currentData()
        if not rid:
            return None
        return {
            "target": rid,
            "kind": self.kind_combo.currentData(),
            "tags": [x.strip() for x in self.tags_edit.text().split(",") if x.strip()],
            "note": self.note_edit.toPlainText().strip(),
            "author": self.author_edit.text().strip() or "teacher",
        }
