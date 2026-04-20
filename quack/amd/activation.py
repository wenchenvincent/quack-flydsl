# Copyright (c) 2026, AMD.

"""Activation functions used by QuACK's AMD kernels.

Scalar-only port of `quack/activation.py`: the packed f32x2 fast paths that
NVIDIA SM90+ exposes via `cute.arch.*_packed_f32x2` don't have a direct AMD
analogue, so each function operates on a single Float32 value. Call sites
that want vectorization should rely on MLIR vector ops or explicit unrolling.

Functions are designed to be invoked inside a `@flyc.kernel` body: the
`Float32` argument is a FlyDSL Numeric value whose operator overloads lower
to MLIR arith/math ops.
"""

import math as _py_math
from typing import Tuple

from flydsl.expr import Float32, math, arith


# --- Elementwise -----------------------------------------------------------


def tanh(x: Float32) -> Float32:
    """Hardware tanh (maps to ROCDL v_tanh on CDNA / the MLIR math.tanh path)."""
    return math.tanh(x, fastmath="fast")


def sigmoid(x: Float32) -> Float32:
    # 0.5 + 0.5 * tanh(0.5 * x) — fewer instructions than 1/(1+exp(-x))
    return 0.5 + 0.5 * tanh(0.5 * x)


def dsigmoid_from_output(out: Float32, dout: Float32) -> Float32:
    return dout * (out - out * out)


def relu(x: Float32) -> Float32:
    return arith.maximumf(x, Float32(0.0))


def drelu(x: Float32, dout: Float32) -> Tuple[Float32, Float32]:
    dx = arith.select(x > Float32(0.0), dout, Float32(0.0))
    return dx, relu(x)


def relu_sq(x: Float32) -> Float32:
    return relu(x) * x


def drelu_sq(x: Float32, dout: Float32) -> Tuple[Float32, Float32]:
    relu_x = relu(x)
    relu_sq_out = relu_x * x
    # d/dx[max(x,0) * x] = 2*x if x > 0, else 0
    dx = Float32(2.0) * (dout * relu_x)
    return dx, relu_sq_out


_SQRT_2_OVER_PI = _py_math.sqrt(2.0 / _py_math.pi)  # ~0.797885
_GELU_COEFF_2 = 0.044715 * _SQRT_2_OVER_PI  # ~0.0356774
_GELU_COEFF_3 = 3.0 * _GELU_COEFF_2  # ~0.01070322


def gelu_tanh_approx(x: Float32) -> Float32:
    """gelu(x) = 0.5 * x * (1 + tanh(x * (c1 + c2 * x^2)))."""
    return Float32(0.5) * (x * (Float32(1.0) + tanh(x * (_SQRT_2_OVER_PI + _GELU_COEFF_2 * (x * x)))))


def dgelu_tanh_approx(x: Float32, dout: Float32) -> Tuple[Float32, Float32]:
    """Returns (dx, gelu_out). Chain rule over the tanh approximation."""
    x_sq = x * x
    tanh_z = tanh(x * (_SQRT_2_OVER_PI + _GELU_COEFF_2 * x_sq))
    half_tanh_plus_1 = Float32(0.5) + Float32(0.5) * tanh_z
    gelu_out = x * half_tanh_plus_1
    sech2_z = Float32(1.0) - tanh_z * tanh_z
    dz_dx = _SQRT_2_OVER_PI + _GELU_COEFF_3 * x_sq
    dgelu = half_tanh_plus_1 + x * (Float32(0.5) * (sech2_z * dz_dx))
    return dout * dgelu, gelu_out


def softplus(x: Float32) -> Float32:
    """softplus(x) = log(1 + exp(x)), with a linear bypass for large x to avoid overflow."""
    linear = x > Float32(20.0)
    computed = math.log(Float32(math.exp(x, fastmath="fast")) + Float32(1.0), fastmath="fast")
    return arith.select(linear, x, computed)


def dsoftplus_from_output(out: Float32, dout: Float32) -> Float32:
    linear = out > Float32(20.0)
    dx = dout - dout * math.exp(-out, fastmath="fast")
    return arith.select(linear, dout, dx)


def silu(x: Float32, already_halved: bool = False) -> Float32:
    """silu(x) = x * sigmoid(x); rewritten as (0.5 * x) * tanh(0.5 * x) + (0.5 * x)."""
    x_half = x if already_halved else Float32(0.5) * x
    return x_half * tanh(x_half) + x_half


# --- Gated variants (two-input) -------------------------------------------


def swiglu(x: Float32, y: Float32) -> Float32:
    return silu(x) * y


def dswiglu(x: Float32, y: Float32, dout: Float32) -> Tuple[Float32, Float32, Float32]:
    """d/dx[silu(x) * y], d/dy[silu(x) * y], silu(x) * y."""
    sigmoid_x = sigmoid(x)
    silu_x = x * sigmoid_x
    silu_x_dout = silu_x * dout
    # d_silu(x) * dout = (sigmoid_x - silu_x * sigmoid_x) * dout + silu_x * dout
    d_silu_x_dout = (sigmoid_x - silu_x * sigmoid_x) * dout + silu_x_dout
    dx = d_silu_x_dout * y
    dy = silu_x_dout
    swiglu_out = silu_x * y
    return dx, dy, swiglu_out


def swiglu_oai(x: Float32, y: Float32, alpha: float = 1.702) -> Float32:
    """x * sigmoid(alpha * x) * (y + 1) — gpt-oss variant."""
    x_half = Float32(0.5) * x
    silu_x = x_half * tanh(Float32(alpha) * x_half) + x_half
    return silu_x * y + silu_x


def dswiglu_oai(
    x: Float32, y: Float32, dout: Float32, alpha: float = 1.702
) -> Tuple[Float32, Float32, Float32]:
    alpha_x_half = Float32(0.5 * alpha) * x
    sigmoid_alpha_x = Float32(0.5) + Float32(0.5) * tanh(alpha_x_half)
    silu_x = x * sigmoid_alpha_x
    silu_x_dout = silu_x * dout
    d_silu_x_dout = (sigmoid_alpha_x + Float32(alpha) * (silu_x - silu_x * sigmoid_alpha_x)) * dout
    dx = d_silu_x_dout * y + d_silu_x_dout
    dy = silu_x_dout
    swiglu_out = silu_x * y + silu_x
    return dx, dy, swiglu_out


def glu(x: Float32, y: Float32) -> Float32:
    return sigmoid(x) * y


def dglu(x: Float32, y: Float32, dout: Float32) -> Tuple[Float32, Float32, Float32]:
    sigmoid_x = sigmoid(x)
    sigmoid_x_dout = sigmoid_x * dout
    glu_out = sigmoid_x * y
    # dx = (y - glu_out) * sigmoid_x_dout
    dx = (y - glu_out) * sigmoid_x_dout
    dy = sigmoid_x_dout
    return dx, dy, glu_out


def reglu(x: Float32, y: Float32) -> Float32:
    return relu(x) * y


def dreglu(x: Float32, y: Float32, dout: Float32) -> Tuple[Float32, Float32, Float32]:
    x_pos = x > Float32(0.0)
    relu_x = relu(x)
    dx = arith.select(x_pos, dout * y, Float32(0.0))
    dy = dout * relu_x
    reglu_out = relu_x * y
    return dx, dy, reglu_out


def geglu(x: Float32, y: Float32) -> Float32:
    return gelu_tanh_approx(x) * y


def dgeglu(x: Float32, y: Float32, dout: Float32) -> Tuple[Float32, Float32, Float32]:
    dgelu_x_dout, gelu_x = dgelu_tanh_approx(x, dout)
    dx = dgelu_x_dout * y
    dy = gelu_x * dout
    geglu_out = gelu_x * y
    return dx, dy, geglu_out


# --- Name maps (match QuACK's activation.py contract) ---------------------


act_fn_map = {
    None: None,
    "silu": silu,
    "relu": relu,
    "relu_sq": relu_sq,
    "gelu_tanh_approx": gelu_tanh_approx,
}

dact_fn_map = {
    None: None,
    "relu": drelu,
    "relu_sq": drelu_sq,
    "gelu_tanh_approx": dgelu_tanh_approx,
}

gate_fn_map = {
    "swiglu": swiglu,
    "swiglu_oai": swiglu_oai,
    "reglu": reglu,
    "geglu": geglu,
    "glu": glu,
}

dgate_fn_map = {
    "swiglu": dswiglu,
    "swiglu_oai": dswiglu_oai,
    "reglu": dreglu,
    "geglu": dgeglu,
    "glu": dglu,
}
