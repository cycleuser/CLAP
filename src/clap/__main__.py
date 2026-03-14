"""CLAP main entry point with unified command interface."""

import argparse
import sys


def main():
    """Main entry point for CLAP."""
    parser = argparse.ArgumentParser(
        prog="clap",
        description="CLAP - Chat Local And Persistent. Local LLM conversation tool with Ollama.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  clap                  Start interactive CLI mode
  clap --gui            Start GUI mode
  clap --web            Start web server
  clap chat "Hello"     Send a single message
  clap models           List available Ollama models
  clap kb index doc.pdf Index a document into knowledge base
        """,
    )

    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version="clap-llm 1.0.0",
    )

    parser.add_argument(
        "-g",
        "--gui",
        action="store_true",
        help="Start GUI mode",
    )

    parser.add_argument(
        "-w",
        "--web",
        action="store_true",
        help="Start web server mode",
    )

    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host for web server (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for web server (default: 5000)",
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # chat command
    chat_parser = subparsers.add_parser("chat", help="Chat with an LLM model")
    chat_parser.add_argument("message", nargs="*", help="Message to send")
    chat_parser.add_argument("--model", "-m", help="Model to use")

    # models command
    subparsers.add_parser("models", help="List available Ollama models")

    # interactive command
    subparsers.add_parser("interactive", help="Start interactive chat mode")

    # kb command
    kb_parser = subparsers.add_parser("kb", help="Knowledge base operations")
    kb_subparsers = kb_parser.add_subparsers(dest="kb_command", help="KB subcommands")

    kb_index = kb_subparsers.add_parser("index", help="Index a document")
    kb_index.add_argument("path", help="Path to document")

    kb_search = kb_subparsers.add_parser("search", help="Search knowledge base")
    kb_search.add_argument("query", help="Search query")
    kb_search.add_argument("-k", type=int, default=5, help="Number of results")

    kb_subparsers.add_parser("stats", help="Show knowledge base stats")
    kb_subparsers.add_parser("clear", help="Clear knowledge base")

    args = parser.parse_args()

    # Handle mode flags first
    if args.gui:
        return _start_gui()

    if args.web:
        return _start_web(args.host, args.port)

    # Handle subcommands
    if args.command == "chat":
        return _run_chat(args)
    elif args.command == "models":
        return _run_models()
    elif args.command == "interactive":
        return _run_interactive()
    elif args.command == "kb":
        return _run_kb(args)

    # Default: start interactive mode
    return _run_interactive()


def _start_gui():
    """Start GUI mode."""
    try:
        from clap.gui.main_window import main as gui_main

        return gui_main()
    except ImportError as e:
        print(f"Error importing GUI components: {e}")
        print("Please ensure PySide6 is installed: pip install PySide6")
        return 1


def _start_web(host, port):
    """Start web server mode."""
    try:
        import sys

        from clap.web.app import main as web_main

        sys.argv = ["clap", "--host", host, "--port", str(port)]
        return web_main()
    except ImportError as e:
        print(f"Error: Web mode requires Flask. {e}")
        print("Install with: pip install clap-llm[web]")
        return 1


def _run_chat(args):
    """Run chat command."""
    try:
        from clap.cli.main import chat_command

        return chat_command(args)
    except ImportError:
        print("Error: CLI features not available")
        return 1


def _run_models():
    """Run models command."""
    try:
        from clap.cli.main import models_command

        return models_command(None)
    except ImportError:
        print("Error: CLI features not available")
        return 1


def _run_interactive():
    """Run interactive mode."""
    try:
        from clap.cli.main import interactive_mode

        return interactive_mode()
    except ImportError:
        print("Error: CLI features not available")
        return 1


def _run_kb(args):
    """Run knowledge base command."""
    try:
        from clap.cli.main import kb_command

        return kb_command(args)
    except ImportError:
        print("Error: CLI features not available")
        return 1


if __name__ == "__main__":
    sys.exit(main())
