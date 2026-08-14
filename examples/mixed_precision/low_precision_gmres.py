"""Low-precision restarted GMRES example.

Run directly:
    python examples/mixed_precision/low_precision_gmres.py
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
            [4.0, 1.0, 0.0, 0.0],
            [1.0, 4.0, 1.0, 0.0],
            [0.0, 1.0, 4.0, 1.0],
            [0.0, 0.0, 1.0, 3.0],
        ],
        dtype=float,
    )
    b = np.array([1.0, 2.0, 2.0, 1.0], dtype=float)
    return A, b


def main(
    A: np.ndarray | None = None,
    b: np.ndarray | None = None,
    precision: str | dict[str, Any] | None = "fp16",
    restart: int = 4,
    max_steps: int = 20,
    tol: float = 1e-8,
    backend: str = "numpy",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve Ax=b with restarted GMRES while quantizing Krylov operations."""
    pychop.backend(backend)
    A, b = default_problem() if A is None or b is None else (np.asarray(A, dtype=float), np.asarray(b, dtype=float))
    chop = make_chopper(precision)
    q = lambda x: np.asarray(chop(np.asarray(x, dtype=float)), dtype=float)

    A_low = q(A)
    b_low = q(b)
    x = np.zeros_like(b_low)
    b_norm = max(np.linalg.norm(b), np.finfo(float).tiny)
    residuals: list[float] = []
    steps = 0

    while steps < max_steps:
        r = q(b_low - q(A_low @ x))
        beta = float(np.linalg.norm(r))
        rel = float(np.linalg.norm(b - A @ x) / b_norm)
        residuals.append(rel)
        if rel <= tol or beta == 0.0:
            break

        m = min(restart, max_steps - steps)
        V = np.zeros((A.shape[0], m + 1), dtype=float)
        H = np.zeros((m + 1, m), dtype=float)
        V[:, 0] = q(r / beta)
        used = 0

        for j in range(m):
            used = j + 1
            steps += 1
            w = q(A_low @ V[:, j])
            for i in range(j + 1):
                H[i, j] = float(np.dot(V[:, i], w))
                w = q(w - H[i, j] * V[:, i])
            H[j + 1, j] = float(np.linalg.norm(w))
            if H[j + 1, j] == 0.0:
                break
            V[:, j + 1] = q(w / H[j + 1, j])

        e1 = np.zeros(used + 1)
        e1[0] = beta
        y, *_ = np.linalg.lstsq(H[: used + 1, :used], e1, rcond=None)
        x = q(x + q(V[:, :used] @ y))

    return x, {"backend": backend, "precision": precision, "relative_residuals": residuals, "steps": steps}


if __name__ == "__main__":
    solution, info = main()
    print("low-precision GMRES")
    print("solution:", solution)
    print("relative residuals:", info["relative_residuals"])
