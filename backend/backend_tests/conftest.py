"""Ensure project root is on sys.path so that `dreaming_hawk`, `wordGraph`, and
`textUtils` are importable inside the test environment regardless of how pytest
is invoked.
"""

import sys
from pathlib import Path

# This file lives at `<project>/backend/backend_tests/conftest.py`.
# The project root is therefore two levels up from this file's parent directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
