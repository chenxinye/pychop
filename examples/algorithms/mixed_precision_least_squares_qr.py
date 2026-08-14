"""Mixed-precision QR least-squares example with pychop built-in types."""

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
from pychop.builtin.linalg import qr


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


def mixed_precision_least_squares_qr(
    A,
    b,
    *,
    factor_chopper=None,
    projection_chopper=None,
    solution_chopper=None,
    backend: str = "numpy",
):
    """
    Solve ``min_x ||A x - b||_2`` using QR with separate step precisions.

    ``factor_chopper`` controls the QR factorization inputs/outputs,
    ``projection_chopper`` controls ``Q.T @ b``, and ``solution_chopper``
    controls the returned solution. The function returns ``(x, diagnostics)``.
    """
    pychop.backend(backend)
    factor_chopper = factor_chopper or Chop(exp_bits=5, sig_bits=10)
    projection_chopper = projection_chopper or factor_chopper
    solution_chopper = solution_chopper or projection_chopper

    A_np = np.asarray(A, dtype=float)
    b_np = np.asarray(b, dtype=float)
    A_low = _wrap_backend(A_np, factor_chopper, backend)
    b_proj = _wrap_backend(b_np, projection_chopper, backend)

    Q, R = qr(A_low, mode="reduced")
    c = cast_precision(_to_numpy(Q).T @ _to_numpy(b_proj), projection_chopper)
    x_np = np.linalg.solve(_to_numpy(R), _to_numpy(c))
    x = _wrap_backend(x_np, solution_chopper, backend)
    residual = np.linalg.norm(A_np @ _to_numpy(x) - b_np)
    return x, {"residual_norm": float(residual)}


if __name__ == "__main__":
    A_demo = np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]])
    b_demo = np.array([6.0, 5.0, 7.0, 10.0])
    fp16 = Chop(exp_bits=5, sig_bits=10)
    fp32_like = Chop(exp_bits=8, sig_bits=23)
    sol, info = mixed_precision_least_squares_qr(
        A_demo,
        b_demo,
        factor_chopper=fp16,
        projection_chopper=fp32_like,
        solution_chopper=fp32_like,
    )
    print("solution:", sol)
    print("diagnostics:", info)
