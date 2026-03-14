# AGENTS.md - Coding Agent Guidelines for CLAP

## Project Overview

CLAP (Chat Local And Persistent) is a Python desktop application for local LLM conversations built on the Ollama framework. It uses PySide6 (Qt) for the GUI and BeeWare Briefcase for packaging.

## Build, Run, and Test Commands

### Development Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest
```

### Running the Application

```bash
# From project root, run directly
cd clap && python src/CLAP/app.py

# Or use Briefcase development mode
cd clap && briefcase dev
```

### Testing

```bash
# Run all tests
cd clap && pytest

# Run a single test file
cd clap && pytest tests/test_example.py

# Run a single test function
cd clap && pytest tests/test_example.py::test_function_name

# Run tests with verbose output
cd clap && pytest -v

# Run tests with coverage
cd clap && pytest --cov=src/CLAP tests/
```

### Linting and Type Checking

```bash
# Install linting tools if not present
pip install ruff mypy

# Run ruff linting
ruff check clap/src/

# Run ruff formatting check
ruff format --check clap/src/

# Run mypy type checking
mypy clap/src/
```

### Building for Distribution

```bash
# Create development environment
cd clap && briefcase dev

# Build application
cd clap && briefcase create
cd clap && briefcase build

# Package for distribution
cd clap && briefcase package
```

## Project Structure

```
CLAP/
├── clap/
│   ├── src/CLAP/           # Main application source
│   │   ├── app.py          # Main application code
│   │   ├── __init__.py
│   │   ├── __main__.py     # Entry point
│   │   └── resources/      # Application resources
│   └── pyproject.toml      # Briefcase project configuration
├── requirements.txt        # Python dependencies
├── images/                 # Documentation images
└── README.md
```

## Code Style Guidelines

### Imports

```python
# Standard library imports (alphabetically ordered)
import base64
import json
import os
import pickle
import re
import shutil
import stat
import sys
from datetime import datetime

# Third-party imports (grouped and alphabetically ordered)
import numpy as np
import ollama
import pandas as pd
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QAction, QFont
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget

# Local imports (last)
from .module import LocalClass
```

- Avoid duplicate imports
- Group imports by: standard library, third-party, local
- Sort alphabetically within each group
- Use explicit imports rather than wildcard imports

### Naming Conventions

```python
# Classes: PascalCase
class ChatLocalAndPersistent(QMainWindow):
    pass

class ChatThread(QThread):
    pass

# Functions and methods: snake_case
def send_message(self):
    pass

def open_chat(self):
    pass

# Variables: snake_case
file_loaded_path = ''
model_selector = QComboBox()

# Constants: UPPER_SNAKE_CASE
DEFAULT_WINDOW_WIDTH = 1024
DEFAULT_WINDOW_HEIGHT = 600

# Private methods: prefix with underscore
def _internal_helper(self):
    pass

# Qt signals: snake_case with Signal type
new_text = Signal(str)
```

### Formatting

- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Blank lines between method definitions
- Two blank lines between class definitions

```python
class ExampleClass:
    def method_one(self):
        pass
    
    def method_two(self):
        pass


class AnotherClass:
    pass
```

### Type Hints

Add type hints to function signatures:

```python
from typing import Optional, List, Dict, Any

def is_image(file_path: str) -> bool:
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except (IOError, SyntaxError):
        return False

def process_messages(
    self,
    messages: List[Dict[str, str]],
    model: str,
    path: Optional[str] = ''
) -> None:
    pass
```

### Error Handling

```python
# Use specific exception types
try:
    self.messages[-1]['content'] += 'And the Model reply is: ' + self.new_reply
except (KeyError, IndexError) as e:
    print(f"Error updating message: {e}")
    pass

# For file operations, use context managers
with open(file_path, 'rb') as f:
    data = pickle.load(f)

# For optional operations, use try/except
try:
    self.file_loaded_path = data['file_loaded_path']
except KeyError:
    self.file_loaded_path = ''
```

### Qt/PySide6 Patterns

```python
# Signal connections
self.chat_thread.new_text.connect(self.update_text_browser)
self.chat_thread.finished.connect(self.on_chat_finished)

# Widget styling
self.toolbar.setStyleSheet("font-size: 12px")

# Layout management
layout = QVBoxLayout()
layout.addWidget(widget)

# Shortcuts
self.new_action.setShortcut('Ctrl+N')
```

### Documentation

Use docstrings for public methods and classes:

```python
class ChatThread(QThread):
    """Thread for handling chat communication with Ollama.
    
    Attributes:
        messages: List of message dictionaries for the conversation.
        model: The model name to use for generation.
        path: Optional path to a file for processing.
    """
    
    def run(self) -> None:
        """Execute the chat thread, streaming responses from the model."""
        pass
```

## Dependencies

Key dependencies (see pyproject.toml and requirements.txt):
- PySide6: Qt GUI framework
- ollama: Ollama Python client
- langchain, langchain_ollama: LLM orchestration
- chromadb: Vector database for RAG
- matplotlib, numpy, pandas: Data visualization
- mistune: Markdown parsing

## Notes for Agents

1. The application uses Chinese comments in some places - maintain consistency with existing comment style
2. When modifying UI code, test on both light and dark themes
3. The app requires Ollama to be installed and running locally
4. File paths use OS-appropriate separators via os.path
5. ChromaDB persistence directories should be cleaned up appropriately