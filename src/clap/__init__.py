"""CLAP - Chat Local And Persistent.

A local LLM conversation tool with Ollama, featuring RAG, document import,
and dual-panel display.
"""

__version__ = "1.0.0"
__author__ = "Frederick"
__email__ = "wedonotuse@outlook.com"


def main():
    """Main entry point for CLAP."""
    from clap.__main__ import main as _main
    return _main()


# Lazy imports to avoid dependency issues during setup
def __getattr__(name):
    if name == "ChatThread":
        from clap.core.chat_thread import ChatThread
        return ChatThread
    elif name == "KnowledgeBase":
        from clap.core.knowledge_base import KnowledgeBase
        return KnowledgeBase
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["main", "ChatThread", "KnowledgeBase"]
