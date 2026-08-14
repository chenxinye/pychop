"""Low-precision Newton iteration with pychop CPFloat."""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pychop
from pychop import Chop
from pychop.builtin import CPFloat, cast_precision


def low_precision_newton(
    f,
    df,
    x0,
    *,
    function_chopper=None,
    update_chopper=None,
    solution_chopper=None,
    maxiter: int = 20,
    tol: float = 1e-10,
):
    """
    Find a scalar root using Newton's method with per-step precision controls.

    ``function_chopper`` controls function/derivative evaluation values,
    ``update_chopper`` controls Newton updates, and ``solution_chopper`` controls
    the returned solution. The function returns ``(x, history)``.
    """
    pychop.backend("numpy")
    function_chopper = function_chopper or Chop(exp_bits=5, sig_bits=10)
    update_chopper = update_chopper or function_chopper
    solution_chopper = solution_chopper or update_chopper

    x = CPFloat(x0, solution_chopper)
    history = []
    for _ in range(maxiter):
        x_eval = cast_precision(x, function_chopper)
        fx = CPFloat(f(float(x_eval)), function_chopper)
        dfx = CPFloat(df(float(x_eval)), function_chopper)
        history.append(abs(float(fx)))
        if abs(float(fx)) <= tol:
            break
        step = CPFloat(float(fx) / float(dfx), update_chopper)
        x = cast_precision(float(x) - float(step), solution_chopper)

    return x, history


if __name__ == "__main__":
    fp16 = Chop(exp_bits=5, sig_bits=10)
    fp32_like = Chop(exp_bits=8, sig_bits=23)
    root, hist = low_precision_newton(
        lambda x: x * x - 2.0,
        lambda x: 2.0 * x,
        1.0,
        function_chopper=fp16,
        update_chopper=fp16,
        solution_chopper=fp32_like,
    )
    print("root:", root)
    print("sqrt(2):", math.sqrt(2.0))
    print("residual history:", hist)
