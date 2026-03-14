# CLAP (Chat Local And Persistent) Implementation Summary

## Overview
CLAP has been restructured and enhanced to become `clap-llm`, a comprehensive local LLM conversation tool. **Now compatible with NuoYi/marker-pdf**.

## Package Structure

```
CLAP/
├── pyproject.toml          # Package configuration
├── requirements.txt        # Dependencies
├── src/
│   └── clap/               # Main package
│       ├── __init__.py
│       ├── __main__.py
│       ├── core/           # Core functionality
│       │   ├── chat_thread.py
│       │   └── knowledge_base.py
│       ├── gui/            # PySide6 GUI
│       │   └── main_window.py
│       ├── cli/            # Rich CLI
│       │   └── main.py
│       └── web/            # Flask web interface
│           └── app.py
├── build.bat
└── upload_pypi.sh
```

## Installation

```bash
# From CLAP directory
pip install -e ./

# With optional features
pip install -e .[rag,data,web,cli]
```

## Features

### Core (Minimal Dependencies)
- **PySide6 GUI**: Dual-panel display
- **Ollama integration**: Streaming responses
- **ChromaDB + LangChain**: RAG support
- **Document processing**: PDF, DOCX, TXT (optional)

### Compatibility
- **No conflicts with NuoYi/marker-pdf**
- Removed `unstructured` dependency (source of conflicts)
- Uses `pypdf` and `python-docx` for document processing

## Usage

```bash
# CLI
clap --help
clap chat "Hello"
clap interactive

# GUI
clap-gui

# Web
clap-web --port 8080
```

## Dependencies

### Required (Minimal)
- PySide6, ollama, langchain-*, chromadb
- mistune, Pygments, Pillow, requests

### Optional
- `[rag]`: pypdf, python-docx, faiss-cpu
- `[data]`: numpy, pandas, matplotlib, scipy
- `[web]`: flask, flask-socketio
- `[cli]`: rich, prompt-toolkit

## Changes Made

1. **Moved pyproject.toml to root** for `pip install -e ./`
2. **Simplified dependencies** to avoid conflicts with marker-pdf
3. **Replaced unstructured** with pypdf/python-docx for document loading
4. **Updated imports** for newer langchain versions
5. **Reduced core dependencies** from 26 to 12 packages