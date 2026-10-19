"""CLAP main entry point with unified command interface.

All parsing/dispatch lives in `clap.cli.main`; this module stays the thin
entrypoint wired in pyproject (`clap = "clap:main"`).
"""

import sys

from clap.cli.main import main

if __name__ == "__main__":
    sys.exit(main())
