"""LU/QR-based solve with precision switching.

Run directly:
    python examples/mixed_precision/lu_qr_precision_switching_solve.py
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import pychop
from pychop import Chop


_PRECISIONS = {
    "fp16": {"exp_bits": 5, "sig_bits": 10},
    "half": {"exp_bits": 5, "sig_bits": 10},
    "bf16": {"exp_bits": 8, "sig_bits": 7},
    "bfloat16": {"exp_bits": 8, "sig_bits": 7},
    "fp32": {"exp_bits": 8, "sig_bits": 23},
    "fp8_e4m3": {"exp_bits": 4, "sig_bits": 3},
    "fp8_e5m2": {"exp_bits": 5, "sig_bits": 2},
}


def make_chopper(precision: str | dict[str, Any] | None) -> Chop:
    spec = dict(_PRECISIONS["fp16"] if precision is None else _PRECISIONS[precision.lower()] if isinstance(precision, str) else precision)
    spec.setdefault("rmode", 1)
    spec.setdefault("subnormal", True)
    return Chop(**spec)


def default_problem() -> tuple[np.ndarray, np.ndarray]:
    A = np.array(
        [
            [3.0, 1.0, 0.25, 0.0],
            [0.5, 2.5, 1.0, 0.0],
            [0.0, 0.5, 3.0, 1.0],
            [0.25, 0.0, 1.0, 2.5],
        ],
        dtype=float,
    )
    b = np.array([1.0, -1.0, 2.0, 0.5], dtype=float)
    return A, b


def lu_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Small partial-pivot LU solver kept local so this file is self-contained."""
    U = A.copy()
    y = b.copy()
    n = A.shape[0]
    L = np.eye(n)
    piv = np.arange(n)
    for k in range(n - 1):
        pivot = k + int(np.argmax(np.abs(U[k:, k])))
        if pivot != k:
            U[[k, pivot], :] = U[[pivot, k], :]
            L[[k, pivot], :k] = L[[pivot, k], :k]
            piv[[k, pivot]] = piv[[pivot, k]]
        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]
    y = y[piv]
    z = np.linalg.solve(L, y)
    return np.linalg.solve(U, z)


def factor_solve(A: np.ndarray, b: np.ndarray, method: str) -> np.ndarray:
    if method == "lu":
        return lu_solve(A, b)
    if method == "qr":
        Q, R = np.linalg.qr(A)
        return np.linalg.solve(R, Q.T @ b)
    raise ValueError("method must be 'lu' or 'qr'")


def main(
    A: np.ndarray | None = None,
    b: np.ndarray | None = None,
    precision: str | dict[str, Any] | None = "fp16",
    method: str = "lu",
    correction_steps: int = 2,
    backend: str = "numpy",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Factor in low precision, then switch to high precision for residual correction."""
    pychop.backend(backend)
    A, b = default_problem() if A is None or b is None else (np.asarray(A, dtype=float), np.asarray(b, dtype=float))
    chop = make_chopper(precision)
    q = lambda x: np.asarray(chop(np.asarray(x, dtype=float)), dtype=float)

    A_low = q(A)
    x = q(factor_solve(A_low, q(b), method))
    b_norm = max(np.linalg.norm(b), np.finfo(float).tiny)
    residuals = [float(np.linalg.norm(b - A @ x) / b_norm)]
    for _ in range(correction_steps):
        r = b - A @ x
        x = q(x + factor_solve(A_low, q(r), method))
        residuals.append(float(np.linalg.norm(b - A @ x) / b_norm))

    return x, {"backend": backend, "precision": precision, "method": method, "relative_residuals": residuals}


if __name__ == "__main__":
    for solver in ("lu", "qr"):
        solution, info = main(method=solver)
        print(f"{solver.upper()} precision-switching solve")
        print("solution:", solution)
        print("relative residuals:", info["relative_residuals"])
