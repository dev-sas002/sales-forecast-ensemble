"""
Pytest configuration: put the repo root on sys.path so `src` is importable.

The previous version computed `Path(__file__).resolve().parent`, which is the
tests/ directory — not the repo root its own comment claimed. It inserted a
path that makes nothing importable; `import src` worked only because pytest
happened to add the rootdir itself.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
