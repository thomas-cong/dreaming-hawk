"""dreaming_hawk package shim.

Allows code to `import dreaming_hawk.wordGraph` or `from dreaming_hawk import wordGraph`
without moving the original standalone modules.  It simply re-exports the existing
`wordGraph` and `textUtils` modules that live at the project root.
"""

from __future__ import annotations
import importlib
import sys
from pathlib import Path

# Ensure the project root and key sub-directories are importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Also add commonly referenced nested source directories for both legacy
# and current layouts, so modules like `wordGraph` resolve without local tweaks.
for p in [
    PROJECT_ROOT / "Graphs",
    PROJECT_ROOT / "backend" / "Graphs",
    PROJECT_ROOT / "backend" / "Graphs" / "WordGraph",
    PROJECT_ROOT / "Graphs" / "WordGraph",
]:
    sp = str(p)
    if p.exists() and sp not in sys.path:
        sys.path.insert(0, sp)

# Lazily import existing standalone modules. Support either
# project-root `wordGraph.py` or nested `Graphs/WordGraph/wordGraph.py`
_wordgraph_import_err = None
try:
    wordGraph = importlib.import_module("wordGraph")
except ModuleNotFoundError as e:
    _wordgraph_import_err = e
    # Try known nested locations explicitly
    for candidate in [
        PROJECT_ROOT / "Graphs" / "WordGraph",
        PROJECT_ROOT / "backend" / "Graphs" / "WordGraph",
    ]:
        if candidate.exists():
            sp = str(candidate)
            if sp not in sys.path:
                sys.path.insert(0, sp)
            try:
                wordGraph = importlib.import_module("wordGraph")
                _wordgraph_import_err = None
                break
            except ModuleNotFoundError as e2:
                _wordgraph_import_err = e2
                continue
    if _wordgraph_import_err is not None:
        raise _wordgraph_import_err

textUtils = importlib.import_module("textUtils")

# Expose submodules so they can be imported as `dreaming_hawk.wordGraph`
module_name = __name__
sys.modules[f"{module_name}.wordGraph"] = wordGraph
sys.modules[f"{module_name}.textUtils"] = textUtils

__all__ = ["wordGraph", "textUtils"]
