"""Put `app` and `scripts` on the import path.

In the container both live under /app (scripts is bind-mounted to /app/scripts);
on the host they are backend/ and the repo root. Adding both keeps `make test`
and a bare `pytest` from the repo root equivalent.
"""

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND.parent

for path in (_BACKEND, _REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
