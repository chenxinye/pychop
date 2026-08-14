"""Mixed-precision iterative refinement with pychop built-in types."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pychop
from pychop import Chop
from pychop.builtin import CPArray, cast_precision
from pychop.builtin.linalg import norm, solve


def _wrap_backend(x, chopper, backend):
    if backend == "numpy":
        return CPArray(np.asarray(x), chopper)
    if backend == "jax":
        import jax.numpy as jnp
        from pychop.builtin import CPJaxArray
        return CPJaxArray(jnp.asarray(x), chopper)
    if backend == "torch":
        import torch
        from pychop.builtin import CPTensor
        return CPTensor(torch.as_tensor(x, dtype=torch.float32), chopper)
    raise ValueError("backend must be one of {'numpy', 'jax', 'torch'}")


def _to_numpy(x):
    if hasattr(x, "to_regular"):
        x = x.to_regular()
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def iterative_refinement(
    A,
    b,
    *,
    factor_chopper=None,
    residual_chopper=None,
    solution_chopper=None,
    backend: str = "numpy",
    maxiter: int = 5,
    tol: float = 1e-10,
):
    """
    Solve ``A x = b`` by mixed-precision iterative refinement.

    ``factor_chopper`` controls the low-precision solve, ``residual_chopper``
    controls residual/correction casting, and ``solution_chopper`` controls the
    precision of the returned solution. The function returns ``(x, history)``.
    """
    pychop.backend(backend)
    factor_chopper = factor_chopper or Chop(exp_bits=5, sig_bits=10)
    residual_chopper = residual_chopper or factor_chopper
    solution_chopper = solution_chopper or residual_chopper

    A_np = np.asarray(A, dtype=float)
    b_np = np.asarray(b, dtype=float)
    A_low = _wrap_backend(A_np, factor_chopper, backend)
    b_low = _wrap_backend(b_np, factor_chopper, backend)
    x = cast_precision(solve(A_low, b_low), solution_chopper)

    history = []
    for _ in range(maxiter):
        x_np = _to_numpy(x)
        r_np = b_np - A_np @ x_np
        rel_res = np.linalg.norm(r_np) / max(np.linalg.norm(b_np), 1.0)
        history.append(float(rel_res))
        if rel_res <= tol:
            break
        r_residual = _wrap_backend(r_np, residual_chopper, backend)
        r_low = cast_precision(r_residual, factor_chopper)
        d = solve(A_low, r_low)
        x = cast_precision(_to_numpy(x) + _to_numpy(d), solution_chopper)

    return x, history


if __name__ == "__main__":
    A_demo = np.array([[4.0, 1.0], [1.0, 3.0]])
    b_demo = np.array([1.0, 2.0])
    fp16 = Chop(exp_bits=5, sig_bits=10)
    fp32_like = Chop(exp_bits=8, sig_bits=23)
    sol, hist = iterative_refinement(
        A_demo,
        b_demo,
        factor_chopper=fp16,
        residual_chopper=fp32_like,
        solution_chopper=fp32_like,
    )
    print("solution:", sol)
    print("relative residual history:", hist)
