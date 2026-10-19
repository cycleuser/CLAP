"""CLI interface for CLAP.

Public command functions (`*_command` / `*_mode`) are dispatched by
`clap.__main__` (the real `clap` entrypoint). Session-centric subcommands
(fork/tree/replay/share/adopt/sessions/visualize/annotate/...) are the
CLAP-NEXT session features (see spec-session-features.md).
"""

import argparse
import sys

from clap import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the full CLI parser (single source of truth for the entrypoint)."""
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
  clap fork s.clap --at m6 --name 提高 --model deepseek-chat
  clap tree s.clap      Show branches and turns
  clap replay s.clap --branch b2 --model deepseek-chat --log replay.jsonl
  clap share s.clap     Create a shareable session package
  clap adopt s.clapshare --policy rename
  clap sessions list    Show the local session library
  clap visualize s.clap --out timeline.html
  clap annotate s.clap --target m3 --kind keypoint --tags 兼语句
  clap annotate-export s.clap --format csv
        """,
    )
    parser.add_argument("-V", "--version", action="version", version=f"clap-llm {__version__}")
    parser.add_argument("-g", "--gui", action="store_true", help="Start GUI mode")
    parser.add_argument("-w", "--web", action="store_true", help="Start web server mode")
    parser.add_argument("--host", default="0.0.0.0", help="Host for web server")
    parser.add_argument("--port", type=int, default=5000, help="Port for web server")

    sp = parser.add_subparsers(dest="command", help="Available commands")

    chat_p = sp.add_parser("chat", help="Send a single message")
    chat_p.add_argument("message", nargs="*", help="Message")
    chat_p.add_argument("--model", "-m", help="Model")

    sp.add_parser("models", help="List available Ollama models")
    sp.add_parser("interactive", help="Start interactive chat mode")

    kb_p = sp.add_parser("kb", help="Knowledge base operations")
    kb_sp = kb_p.add_subparsers(dest="kb_command", help="KB subcommands")
    idx = kb_sp.add_parser("index", help="Index a document")
    idx.add_argument("path", help="Path to document")
    sch = kb_sp.add_parser("search", help="Search knowledge base")
    sch.add_argument("query", help="Query")
    sch.add_argument("-k", type=int, default=5, help="Number of results")
    kb_sp.add_parser("stats", help="Show knowledge base stats")
    kb_sp.add_parser("clear", help="Clear knowledge base")

    fork_p = sp.add_parser("fork", help="Fork a session at a message (F2)")
    fork_p.add_argument("file", help=".clap file")
    fork_p.add_argument("--at", required=True, help="Message id to fork at")
    fork_p.add_argument("--model", default=None, help="Model override for the branch")
    fork_p.add_argument("--name", default="", help="Branch name")
    fork_p.add_argument("--note", default="", help="Why this fork exists")
    fork_p.add_argument("--out", default=None, help="Write to a new file instead of in-place")

    tree_p = sp.add_parser("tree", help="Show branches and their paths (F2)")
    tree_p.add_argument("file", help=".clap file")

    replay_p = sp.add_parser("replay", help="Whole-replay a branch with a model (F3)")
    replay_p.add_argument("file", help=".clap file")
    replay_p.add_argument("--branch", default=None, help="Branch id (default main)")
    replay_p.add_argument("--model", default=None, help="Model for replay (default branch/manifest)")
    replay_p.add_argument("--provider", default=None, help="Provider id (default ollama-local)")
    replay_p.add_argument("--log", default=None, help="Replay log path (jsonl); default <stem>.replay.jsonl")
    replay_p.add_argument("--context", choices=["replayed", "original"], default="replayed")
    replay_p.add_argument("--to-turn", type=int, default=None)
    replay_p.add_argument("--from-turn", type=int, default=None, dest="force_from")
    replay_p.add_argument("--fresh", action="store_true", help="Ignore existing log, start over")

    sp.add_parser("providers", help="List configured providers (masked keys; F6)")

    share_p = sp.add_parser("share", help="Create a shareable session package (F1)")
    share_p.add_argument("file", help=".clap file")
    share_p.add_argument("--out", default=None, help="Output directory (default: alongside file)")

    adopt_p = sp.add_parser("adopt", help="Adopt a .clapshare package into the library (F1)")
    adopt_p.add_argument("package", help=".clapshare package file")
    adopt_p.add_argument("--policy", choices=["rename", "overwrite", "fork"], default="rename")
    adopt_p.add_argument("--dir", default=None, help="Library directory (default ~/.clap/sessions)")

    sess_p = sp.add_parser("sessions", help="Session library operations (F1)")
    sess_sp = sess_p.add_subparsers(dest="sessions_command")
    sess_sp.add_parser("list", help="List library sessions")

    return parser


def main(argv=None):
    """Dispatcher used by `clap` entrypoint (clap/__main__.py)."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.gui:
        return start_gui()
    if args.web:
        return start_web(args.host, args.port)

    if args.command == "chat":
        return chat_command(args)
    elif args.command == "models":
        return models_command(args)
    elif args.command == "interactive":
        return interactive_mode()
    elif args.command == "kb":
        return kb_command(args)
    elif args.command == "fork":
        return fork_command(args)
    elif args.command == "tree":
        return tree_command(args)
    elif args.command == "replay":
        return replay_command(args)
    elif args.command == "providers":
        return providers_command(args)
    elif args.command == "share":
        return share_command(args)
    elif args.command == "adopt":
        return adopt_command(args)
    elif args.command == "sessions":
        return sessions_command(args)
    return interactive_mode()


# ------------------------------------------------------------------ core


def chat_command(args):
    try:
        import ollama as ollama_lib

        msg = " ".join(args.message) if getattr(args, "message", None) else None
        if not msg:
            print('Usage: clap chat "your message"')
            return 1
        models = [m["model"] for m in ollama_lib.list().get("models", [])]
        if not models:
            print("No models found. Install Ollama.")
            return 1
        model = args.model if args.model in models else models[0]
        print(f"Model: {model}\nYou: {msg}\nAI: ", end="", flush=True)
        for chunk in ollama_lib.chat(model=model, messages=[{"role": "user", "content": msg}], stream=True):
            print(chunk["message"]["content"], end="", flush=True)
        print()
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def models_command(args):
    try:
        import ollama as ollama_lib

        models = [m["model"] for m in ollama_lib.list().get("models", [])]
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


def kb_command(args):
    from clap.core.knowledge_base import KnowledgeBase

    kb = KnowledgeBase()
    cmd = getattr(args, "kb_command", None)
    if cmd == "index":
        r = kb.index_document(args.path)
        print(f"Indexed {r.get('chunks', 0)} chunks" if r.get("success") else f"Error: {r.get('error')}")
    elif cmd == "search":
        for i, r in enumerate(kb.search(args.query, args.k), 1):
            print(f"{i}. [{r['score']:.2f}] {r['content'][:100]}...")
    elif cmd == "stats":
        s = kb.get_stats()
        print(f"Chunks: {s.get('chunks', 0)}")
    elif cmd == "clear":
        kb.clear()
        print("Cleared.")
    else:
        print("Usage: clap kb {index,search,stats,clear}")
        return 1
    return 0


def interactive_mode():
    try:
        import ollama as ollama_lib

        models = [m["model"] for m in ollama_lib.list().get("models", [])]
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
        except (ValueError, EOFError):
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
                for chunk in ollama_lib.chat(model=model, messages=messages, stream=True):
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


# ------------------------------------------------------------------ F2


def fork_command(args):
    from clap.core import session as session_mod

    try:
        s = session_mod.load_session(args.file)
        branch = session_mod.fork_branch(
            s, args.at, name=args.name, model=args.model, note=args.note, author="cli"
        )
        out = args.out or args.file
        session_mod.save_session(s, out)
        print(
            f"Forked branch {branch['id']} ({branch['name'] or 'unnamed'}) at {branch['fork_at']}"
            f"{' [model=' + branch['model'] + ']' if branch['model'] else ''} -> {out}"
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def tree_command(args):
    from clap.core import session as session_mod

    try:
        s = session_mod.load_session(args.file)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    man = s.get("manifest", {})
    print(
        f"session {man.get('session_id', '?')} | chat_model={man.get('chat_model', '?')}"
        f" | messages={len(s['messages'])} | branches={len(s.get('branches', []))}"
        f" | annotations={len(s.get('annotations', []))}"
    )
    for b in s.get("branches", []):
        own = session_mod.branch_path(s, b["id"])
        ctx = session_mod.branch_full_path(s, b["id"])
        model = b.get("model") or man.get("chat_model", "?")
        print(
            f"  {b['id']} {b['name'] or '(unnamed)'}: root={b.get('root')}"
            f" fork_at={b.get('fork_at')} model={model} turns={len(own)} (context {len(ctx)})"
        )
        for m in own:
            ann = m.get("annotations") or []
            prev = (m.get("content") or "")[:40].replace("\n", " ")
            ann_note = f" [{len(ann)} ann]" if ann else ""
            print(f"    {m['id']:<6} {m.get('role', '?'):<9} {prev}{ann_note}")
    return 0


# ------------------------------------------------------------------ F3


def replay_command(args):
    """Whole-replay a branch with a chosen provider/model (F3)."""
    import os

    from clap.core import replay as replay_mod
    from clap.core import session as session_mod
    from clap.core.providers import (
        FallbackRouter,
        discover_providers,
        provider_from_names,
    )

    log_path = args.log or os.path.splitext(args.file)[0] + ".replay.jsonl"
    if args.fresh and os.path.exists(log_path):
        os.remove(log_path)

    try:
        s = session_mod.load_session(args.file)
        man = s.get("manifest", {})
        privacy = bool(man.get("privacy_mode", True))

        registry = discover_providers()
        try:
            primary = provider_from_names(registry, args.provider)
        except Exception as e:
            print(f"Error: {e}")
            return 1
        local = registry.get("ollama-local")
        router = FallbackRouter(primary, local, privacy_mode=privacy)

        summary = replay_mod.replay_to_log(
            args.file,
            router,
            log_path=log_path,
            branch_id=args.branch,
            model=args.model,
            context_mode=args.context,
            to_turn=args.to_turn,
            append=not args.fresh,
            force_from=args.force_from,
        )
        for ev in router.events:
            session_mod._add_history(s, ev["event"], ev.get("detail"))

        print(
            f"Replay summary: total={summary['total_turns']} completed={summary['completed_turns']}"
            f" resumed_from={summary.get('resumed_from')} branch={summary['branch']}"
            f" model={summary['model']} log={summary['log']}"
        )
        for ev in router.events:
            print(f"  routing event: {ev['event']} {ev.get('detail', {})}")
        if router.events:
            # persist provider switches into the session's evidence chain
            session_mod.save_session(s, args.file)
        return 0
    except KeyboardInterrupt:
        print("\nReplay interrupted; completed turns are in the log (checkpoint=last completed turn).")
        return 130
    except Exception as e:
        print(f"Error: {e}")
        return 1


# ------------------------------------------------------------------ F6


def providers_command(args):
    """Endpoint inventory: ids, bases, kinds, masked keys (never key values)."""
    from clap.core.providers import discover_providers, masked_endpoint_table

    try:
        rows = masked_endpoint_table(discover_providers())
    except Exception as e:
        print(f"Error: {e}")
        return 1
    import re as _re

    for r in rows:
        masked = r.get("masked_key") or "<none>"
        if isinstance(masked, str) and not _re.fullmatch(r"(\*\*\*len\d+…\S{0,6})|<none>", masked):
            masked = "***"
        avail = "yes" if r["available"] else "no"
        print(f"{r['id']:<22} {r['kind']:<14} available={avail:<3} key={masked}")
        print(f"{'':<22} base={r['base']}")
        models = r.get("models") or []
        shown = ", ".join(models[:10])
        print(f"{'':<22} models[{len(models)}]: {shown}{' …' if len(models) > 10 else ''}")
    return 0


# ------------------------------------------------------------------ F1


def share_command(args):
    from clap.core import sharing

    try:
        pkg = sharing.pack(args.file, args.out)
        man = sharing.read_manifest(pkg)
        print(f"Package: {pkg}")
        print(f"  session_id: {man['session_id']}")
        print(f"  session_sha256: {man['session_sha256']}")
        print(f"  counts: {man['counts']}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def adopt_command(args):
    from clap.core import sharing

    try:
        r = sharing.adopt(args.package, policy=args.policy, library=args.dir)
        print(f"Adopted: session_id={r['session_id']} path={r['path']} policy={r['policy']} verified={r['verified']}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


def sessions_command(args):
    from clap.core import sharing

    if getattr(args, "sessions_command", None) == "list":
        entries = sharing.list_library(args.dir)
        if not entries:
            print("(session library empty)")
            return 0
        for e in entries[:20]:
            tag = "shared" if e.get("shared") else "own"
            print(f"  {e['session_id']}  [{tag}]  {e.get('purpose', ''):<6} turns={e.get('messages', 0)}  {e.get('name', '')}")
        return 0
    print("Usage: clap sessions list")
    return 1


def start_gui():
    try:
        from clap.gui.main_window import main as gui_main

        return gui_main()
    except ImportError as e:
        print(f"Error importing GUI components: {e}")
        print("Please ensure PySide6 is installed: pip install PySide6")
        return 1


def start_web(host, port):
    try:
        from clap.web.app import main as web_main

        sys.argv = ["clap", "--host", host, "--port", str(port)]
        return web_main()
    except ImportError as e:
        print(f"Error: Web mode requires Flask. {e}")
        print("Install with: pip install clap-llm[web]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
