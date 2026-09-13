"""Entry point for ``python -m nvsh``."""

from __future__ import annotations

import sys

from nvsh.cli import main

if __name__ == "__main__":
    sys.exit(main())
