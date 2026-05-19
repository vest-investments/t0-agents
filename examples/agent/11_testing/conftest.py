"""Add the example directory to ``sys.path`` so tests can ``from agent import ...``.

Keeps the example self-contained — users can copy the directory to
their own project, run ``pytest .`` from inside it, and the imports
keep working regardless of whether pytest is launched from the repo
root or from the example directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
