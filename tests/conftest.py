"""Ensure project root is on sys.path so that `dreaming_hawk`, `wordGraph`, and
`textUtils` are importable inside the test environment regardless of how pytest
is invoked.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
