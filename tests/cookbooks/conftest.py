"""Make the repo-root ``cookbooks/`` namespace package importable in tests.

``cookbooks/`` has no ``__init__.py``, so it's a PEP-420 namespace package that
gets picked up when the repo root is on ``sys.path``. ``pytest`` doesn't add
the repo root by default, so we do it here for cookbook-scoped tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
