"""Settings dialog for CLAP."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from clap.utils.config import (
    CLAPConfig,
    get_chat_models,
    get_embedding_models,
    load_config,
    save_config,
)
from clap.utils.i18n import get_available_languages, get_language_name, set_language, t


class SettingsDialog(QDialog):
    """Settings dialog for configuring CLAP."""

    def __init__(self, parent=None, first_run=False):
        super().__init__(parent)
        self.first_run = first_run
        self.config = load_config()

        set_language(self.config.language)

        self.setWindowTitle(t("settings") if not first_run else t("welcome"))
        self.setMinimumWidth(400)

        self.setup_ui()
        if first_run:
            self.setWindowFlags(self.windowFlags() & ~Qt.WindowCloseButtonHint)

    def setup_ui(self):
        """Setup the settings UI."""
        layout = QVBoxLayout(self)

        tabs = QTabWidget()
        tabs.addTab(self._create_models_tab(), t("models"))
        tabs.addTab(self._create_general_tab(), t("general"))
        layout.addWidget(tabs)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()

        if self.first_run:
            btn = QPushButton(t("start"))
            btn.clicked.connect(self.save_and_close)
            btn.setDefault(True)
            btn_layout.addWidget(btn)
        else:
            cancel = QPushButton(t("cancel"))
            cancel.clicked.connect(self.reject)
            btn_layout.addWidget(cancel)

            save = QPushButton(t("settings"))
            save.clicked.connect(self.save_and_close)
            save.setDefault(True)
            btn_layout.addWidget(save)

        layout.addLayout(btn_layout)

    def _create_models_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        chat_group = QGroupBox(t("chat_model"))
        chat_layout = QFormLayout(chat_group)
        self.chat_model_combo = QComboBox()
        self.chat_model_combo.setEditable(True)
        self._refresh_chat_models()
        chat_layout.addRow(f"{t('model')}:", self.chat_model_combo)

        btn = QPushButton(t("refresh_models"))
        btn.clicked.connect(self._refresh_chat_models)
        chat_layout.addRow("", btn)
        layout.addWidget(chat_group)

        embed_group = QGroupBox(t("embed_model"))
        embed_layout = QFormLayout(embed_group)
        self.embed_model_combo = QComboBox()
        self.embed_model_combo.setEditable(True)
        self._refresh_embed_models()
        embed_layout.addRow(f"{t('model')}:", self.embed_model_combo)

        btn2 = QPushButton(t("refresh_models"))
        btn2.clicked.connect(self._refresh_embed_models)
        embed_layout.addRow("", btn2)
        layout.addWidget(embed_group)

        self.chat_model_combo.setCurrentText(self.config.chat_model)
        self.embed_model_combo.setCurrentText(self.config.embed_model)
        layout.addStretch()
        return widget

    def _create_general_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)

        lang_group = QGroupBox(t("language"))
        lang_layout = QFormLayout(lang_group)
        self.lang_combo = QComboBox()
        for lang in get_available_languages():
            self.lang_combo.addItem(get_language_name(lang), lang)
        idx = (
            get_available_languages().index(self.config.language)
            if self.config.language in get_available_languages()
            else 0
        )
        self.lang_combo.setCurrentIndex(idx)
        lang_layout.addRow(f"{t('language')}:", self.lang_combo)
        layout.addWidget(lang_group)

        appear = QGroupBox(t("appearance"))
        appear_layout = QFormLayout(appear)
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(8, 24)
        self.font_size_spin.setValue(self.config.font_size)
        appear_layout.addRow(f"{t('font_size')}:", self.font_size_spin)
        layout.addWidget(appear)

        chunk = QGroupBox(t("doc_processing"))
        chunk_layout = QFormLayout(chunk)
        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(500, 10000)
        self.chunk_size_spin.setSingleStep(500)
        self.chunk_size_spin.setValue(self.config.chunk_size)
        chunk_layout.addRow(f"{t('chunk_size')}:", self.chunk_size_spin)

        self.chunk_overlap_spin = QSpinBox()
        self.chunk_overlap_spin.setRange(0, 1000)
        self.chunk_overlap_spin.setSingleStep(50)
        self.chunk_overlap_spin.setValue(self.config.chunk_overlap)
        chunk_layout.addRow(f"{t('overlap')}:", self.chunk_overlap_spin)
        layout.addWidget(chunk)

        storage = QGroupBox(t("storage"))
        storage_layout = QFormLayout(storage)
        self.persist_dir_edit = QLineEdit(str(self.config.persist_directory))
        storage_layout.addRow(f"{t('path')}:", self.persist_dir_edit)

        browse = QPushButton(t("browse"))
        browse.clicked.connect(self._browse_dir)
        storage_layout.addRow("", browse)
        layout.addWidget(storage)

        layout.addStretch()
        return widget

    def _refresh_chat_models(self):
        current = self.chat_model_combo.currentText()
        self.chat_model_combo.clear()
        self.chat_model_combo.addItems(get_chat_models())
        if current:
            self.chat_model_combo.setCurrentText(current)

    def _refresh_embed_models(self):
        current = self.embed_model_combo.currentText()
        self.embed_model_combo.clear()
        self.embed_model_combo.addItems(get_embedding_models())
        if current:
            self.embed_model_combo.setCurrentText(current)

    def _browse_dir(self):
        path = QFileDialog.getExistingDirectory(self, t("browse"), self.persist_dir_edit.text())
        if path:
            self.persist_dir_edit.setText(path)

    def save_and_close(self):
        chat = self.chat_model_combo.currentText().strip()
        embed = self.embed_model_combo.currentText().strip()

        if not chat:
            QMessageBox.warning(self, t("error"), t("select_model"))
            return
        if not embed:
            QMessageBox.warning(self, t("error"), t("select_model"))
            return

        self.config.chat_model = chat
        self.config.embed_model = embed
        self.config.font_size = self.font_size_spin.value()
        self.config.chunk_size = self.chunk_size_spin.value()
        self.config.chunk_overlap = self.chunk_overlap_spin.value()
        self.config.persist_directory = self.persist_dir_edit.text()
        self.config.language = self.lang_combo.currentData()

        set_language(self.config.language)
        save_config(self.config)
        self.accept()

    def get_config(self) -> CLAPConfig:
        return self.config
