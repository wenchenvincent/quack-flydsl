__version__ = "0.3.11"

import os

try:
    from quack.rmsnorm import rmsnorm
    from quack.softmax import softmax
    from quack.cross_entropy import cross_entropy
    from quack.rounding import RoundingMode
except ImportError:
    # CuTe-DSL (nvidia-cutlass-dsl) is not installed. The NVIDIA kernels are
    # unavailable in this environment, but `quack.amd` (FlyDSL-based AMDGPU port)
    # may still be usable. Eager NVIDIA imports are skipped so `import quack.amd`
    # works on AMD-only installs.
    pass


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    import quack.cute_dsl_ptxas  # noqa: F401

    # Patch to dump ptx and then use system ptxas to compile to cubin
    quack.cute_dsl_ptxas.patch()


__all__ = [
    "rmsnorm",
    "softmax",
    "cross_entropy",
    "RoundingMode",
]
