# Copyright (c) 2026, AMD.

"""Pytest fixtures / setup for the AMD port tests.

Two quirks of FlyDSL's current JIT compilation that this module works around:

1. **Disk-cache invalidation on source change.** When iterating on the AMD
   kernel builders, the FlyDSL cache key doesn't always capture every
   closure change, so stale kernels can leak across sessions. We wipe
   ``~/.flydsl/cache`` at ``pytest_configure`` — cheap relative to kernel
   recompile time, and makes test results deterministic.

2. **Grid-dimension bake-in on first-call compile.** FlyDSL's ``@flyc.jit``
   caches the compiled launcher keyed on argument *types* rather than values.
   For kernels where the first call passes a small grid (``M=1``), the
   cached binary does not correctly handle larger grids on subsequent calls
   with the same type signature — the rmsnorm-bwd and cross-entropy kernels
   exhibit this. The tests parametrise ``M`` with the largest value first
   (``[128, 4, 1]``) so the cached kernel is compiled against the full grid.
   A proper fix is tracked upstream in FlyDSL; this ordering is a pragmatic
   workaround for the tests only.
"""

import os
import shutil
from pathlib import Path


def pytest_configure(config):
    cache_dir = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR") or (Path.home() / ".flydsl" / "cache")
    cache_dir = Path(cache_dir)
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
