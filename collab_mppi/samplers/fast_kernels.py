from __future__ import annotations

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - optional acceleration dependency
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorate(fn):
            return fn
        return decorate
