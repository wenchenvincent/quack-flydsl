# Copyright (c) 2026, AMD.

"""Shared layout helpers for QuACK's AMD kernels.

AMDGPU counterpart to `quack/layout_utils.py`. QuACK's NVIDIA-side layout
helpers wrap CuTe layout algebra; on AMD, FlyDSL exposes a very similar API
directly via ``flydsl.expr.primitive.{make_shape, make_stride, make_layout}``
and ``fx.logical_divide`` / ``fx.slice``. Most kernels use these directly.

This module exists to host any cross-kernel helpers that emerge as the port
grows (e.g. ``expand``, ``transpose`` wrappers, layout-algebra shortcuts).
It's intentionally minimal today — add here only when two or more kernels
share the exact same layout manipulation.
"""

__all__: list = []
