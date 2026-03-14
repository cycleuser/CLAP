"""CLI interface for CLAP."""

import argparse
import sys

import ollama

from clap.core.knowledge_base import KnowledgeBase


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(prog="clap", description="CLAP - Chat Local And Persistent")
    parser.add_argument("-V", "--version", action="version", version="clap-llm 1.0.0")
    subparsers = parser.add_subparsers(dest="cmd")

    chat_p = subparsers.add_parser("chat", help="Chat with LLM")
    chat_p.add_argument("message", nargs="*", help="Message")
    chat_p.add_argument("-m", "--model", help="Model")

    subparsers.add_parser("models", help="List models")

    kb_p = subparsers.add_parser("kb", help="Knowledge base")
    kb_sp = kb_p.add_subparsers(dest="kb_cmd")
    idx = kb_sp.add_parser("index", help="Index document")
    idx.add_argument("path", help="File path")
    sch = kb_sp.add_parser("search", help="Search")
    sch.add_argument("query", help="Query")
    sch.add_argument("-k", type=int, default=5)
    kb_sp.add_parser("stats", help="Stats")
    kb_sp.add_parser("clear", help="Clear")

    args = parser.parse_args()

    if args.cmd == "chat":
        return _chat(args)
    elif args.cmd == "models":
        return _models()
    elif args.cmd == "kb":
        return _kb(args)
    else:
        return _interactive()


def _chat(args):
    msg = " ".join(args.message) if args.message else None
    if not msg:
        print('Usage: clap chat "your message"')
        return 1

    try:
        models = [m["model"] for m in ollama.list().get("models", [])]
        if not models:
            print("No models found. Install Ollama.")
            return 1

        model = args.model if args.model in models else models[0]
        print(f"Model: {model}\nYou: {msg}\nAI: ", end="", flush=True)

        for chunk in ollama.chat(
            model=model, messages=[{"role": "user", "content": msg}], stream=True
        ):
            print(chunk["message"]["content"], end="", flush=True)
        print()
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def _models():
    try:
        models = [m["model"] for m in ollama.list().get("models", [])]
        if not models:
            print("No models found.")
            return 1
        print("Available models:")
        for m in models:
            print(f"  - {m}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def _kb(args):
    kb = KnowledgeBase()

    if args.kb_cmd == "index":
        r = kb.index_document(args.path)
        print(
            f"Indexed {r.get('chunks', 0)} chunks"
            if r.get("success")
            else f"Error: {r.get('error')}"
        )
    elif args.kb_cmd == "search":
        for i, r in enumerate(kb.search(args.query, args.k), 1):
            print(f"{i}. [{r['score']:.2f}] {r['content'][:100]}...")
    elif args.kb_cmd == "stats":
        s = kb.get_stats()
        print(f"Chunks: {s.get('chunks', 0)}")
    elif args.kb_cmd == "clear":
        kb.clear()
        print("Cleared.")
    return 0


def _interactive():
    try:
        models = [m["model"] for m in ollama.list().get("models", [])]
        if not models:
            print("No models found. Install Ollama.")
            return 1

        print(f"CLAP - Models: {', '.join(models)}")
        print("Type 'quit' to exit.\n")

        for i, m in enumerate(models, 1):
            print(f"  {i}. {m}")

        try:
            idx = int(input("\nSelect model [1]: ") or "1") - 1
            model = models[idx] if 0 <= idx < len(models) else models[0]
        except ValueError:
            model = models[0]

        print(f"\nUsing: {model}\n")
        messages = []

        while True:
            try:
                user = input("You: ").strip()
                if not user:
                    continue
                if user.lower() in ("quit", "exit"):
                    break

                messages.append({"role": "user", "content": user})
                print("AI: ", end="", flush=True)

                full = ""
                for chunk in ollama.chat(model=model, messages=messages, stream=True):
                    text = chunk["message"]["content"]
                    print(text, end="", flush=True)
                    full += text
                print()
                messages.append({"role": "assistant", "content": full})

            except KeyboardInterrupt:
                print("\n")
                break

        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
