"""QuACK AMDGPU kernels — FlyDSL port of the CuTe-DSL kernels in `quack/`.

This subpackage is the AMDGPU counterpart to the top-level `quack` modules,
implemented in FlyDSL (https://github.com/ROCm/FlyDSL). The public entry
points mirror QuACK's NVIDIA API so call sites can switch backends without
renaming functions.

Supported architectures (whatever FlyDSL itself supports):
    - gfx942 (MI300X, CDNA3, wave64, 64 KiB LDS)
    - gfx950 (MI350/MI355X, CDNA4, wave64, 160 KiB LDS)
    - gfx1201 (RDNA4, wave32, 64 KiB LDS)
    - gfx1250 (MI450, wave32, 320 KiB LDS, experimental cluster ops)

Importing this package requires `flydsl` to be installed. Install with
``pip install -e '.[amd]'``.
"""

try:
    import flydsl  # noqa: F401
except ImportError as e:
    raise ImportError(
        "quack.amd requires FlyDSL. Install with `pip install -e '.[amd]'` "
        "or see https://github.com/ROCm/FlyDSL for build instructions."
    ) from e

from quack.amd.nn import (  # noqa: F401
    RMSNorm,
    LayerNorm,
    rmsnorm,
    layernorm,
    softmax,
    cross_entropy,
)
from quack.amd.topk import topk  # noqa: F401
