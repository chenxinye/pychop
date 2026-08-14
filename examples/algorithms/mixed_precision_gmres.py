"""Restarted mixed-precision GMRES with pychop built-in types."""

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
from pychop.builtin.linalg import norm


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


def mixed_precision_gmres(
    A,
    b,
    *,
    matvec_chopper=None,
    orthogonalization_chopper=None,
    solution_chopper=None,
    backend: str = "numpy",
    restart: int = 10,
    maxiter: int = 20,
    tol: float = 1e-8,
):
    """
    Solve ``A x = b`` with restarted GMRES and user-selected step precisions.

    Matrix-vector products use ``matvec_chopper``; Arnoldi vector updates use
    ``orthogonalization_chopper``; the returned solution uses
    ``solution_chopper``. The function returns ``(x, history)``.
    """
    pychop.backend(backend)
    matvec_chopper = matvec_chopper or Chop(exp_bits=5, sig_bits=10)
    orthogonalization_chopper = orthogonalization_chopper or matvec_chopper
    solution_chopper = solution_chopper or orthogonalization_chopper

    A_np = np.asarray(A, dtype=float)
    b_np = np.asarray(b, dtype=float)
    x = _wrap_backend(np.zeros_like(b_np), solution_chopper, backend)
    history = []

    for _ in range(maxiter):
        r_np = b_np - A_np @ _to_numpy(x)
        beta = np.linalg.norm(r_np)
        rel_res = beta / max(np.linalg.norm(b_np), 1.0)
        history.append(float(rel_res))
        if rel_res <= tol:
            break

        v0 = _wrap_backend(r_np / beta, orthogonalization_chopper, backend)
        V = [v0]
        H = np.zeros((restart + 1, restart), dtype=float)

        used = 0
        for j in range(restart):
            Av = _wrap_backend(A_np @ _to_numpy(V[j]), matvec_chopper, backend)
            w = cast_precision(Av, orthogonalization_chopper)

            for i in range(j + 1):
                hij = float(np.vdot(_to_numpy(V[i]), _to_numpy(w)))
                H[i, j] = hij
                w = cast_precision(_to_numpy(w) - hij * _to_numpy(V[i]), orthogonalization_chopper)

            h_next = float(norm(w))
            H[j + 1, j] = h_next
            used = j + 1
            if h_next == 0.0:
                break
            V.append(_wrap_backend(_to_numpy(w) / h_next, orthogonalization_chopper, backend))

        if used == 0:
            break

        e1 = np.zeros(used + 1)
        e1[0] = beta
        y, *_ = np.linalg.lstsq(H[: used + 1, :used], e1, rcond=None)
        dx_np = sum(y[i] * _to_numpy(V[i]) for i in range(used))
        x = cast_precision(_to_numpy(x) + dx_np, solution_chopper)

    return x, history


if __name__ == "__main__":
    A_demo = np.array([[4.0, 1.0], [1.0, 3.0]])
    b_demo = np.array([1.0, 2.0])
    fp16 = Chop(exp_bits=5, sig_bits=10)
    fp32_like = Chop(exp_bits=8, sig_bits=23)
    sol, hist = mixed_precision_gmres(
        A_demo,
        b_demo,
        matvec_chopper=fp16,
        orthogonalization_chopper=fp32_like,
        solution_chopper=fp32_like,
    )
    print("solution:", sol)
    print("relative residual history:", hist)
