#!/usr/bin/env python3
"""CLI wrapper for paired GPU7 agentic and GPU6 pure-VLA BEHAVIOR Eval."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robots.behavior.paired_eval import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
