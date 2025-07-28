"""dreaming_hawk package shim.

Allows code to `import dreaming_hawk.wordGraph` or `from dreaming_hawk import wordGraph`
without moving the original standalone modules.  It simply re-exports the existing
`wordGraph` and `textUtils` modules that live at the project root.
"""
from __future__ import annotations
import importlib
import sys
from pathlib import Path

# Ensure the project root is importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Lazily import existing standalone modules
wordGraph = importlib.import_module("wordGraph")
textUtils = importlib.import_module("textUtils")

# Expose submodules so they can be imported as `dreaming_hawk.wordGraph`
module_name = __name__
sys.modules[f"{module_name}.wordGraph"] = wordGraph
sys.modules[f"{module_name}.textUtils"] = textUtils

__all__ = ["wordGraph", "textUtils"]
