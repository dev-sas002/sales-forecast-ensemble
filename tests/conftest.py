"""
Pytest configuration: put the repo root (not tests/) on sys.path so `src` is
importable however pytest was invoked.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
