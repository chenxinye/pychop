"""Low-precision conjugate gradient example.

Run directly:
    python examples/mixed_precision/low_precision_cg.py
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
            [5.0, 1.0, 0.5, 0.0],
            [1.0, 4.0, 0.0, 0.5],
            [0.5, 0.0, 3.5, 1.0],
            [0.0, 0.5, 1.0, 3.0],
        ],
        dtype=float,
    )
    b = np.array([2.0, 1.0, 0.0, 1.0], dtype=float)
    return A, b


def main(
    A: np.ndarray | None = None,
    b: np.ndarray | None = None,
    precision: str | dict[str, Any] | None = "fp16",
    max_steps: int = 25,
    tol: float = 1e-8,
    backend: str = "numpy",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve symmetric positive definite Ax=b with quantized CG recurrences."""
    pychop.backend(backend)
    A, b = default_problem() if A is None or b is None else (np.asarray(A, dtype=float), np.asarray(b, dtype=float))
    chop = make_chopper(precision)
    q = lambda x: np.asarray(chop(np.asarray(x, dtype=float)), dtype=float)

    A_low = q(A)
    b_low = q(b)
    x = np.zeros_like(b_low)
    r = q(b_low - q(A_low @ x))
    p = r.copy()
    rsold = float(np.dot(r, r))
    b_norm = max(np.linalg.norm(b), np.finfo(float).tiny)
    residuals: list[float] = []

    for step in range(max_steps):
        rel = float(np.linalg.norm(b - A @ x) / b_norm)
        residuals.append(rel)
        if rel <= tol or rsold == 0.0:
            return x, {"backend": backend, "precision": precision, "relative_residuals": residuals, "steps": step}
        Ap = q(A_low @ p)
        denom = float(np.dot(p, Ap))
        if denom == 0.0:
            break
        alpha = rsold / denom
        x = q(x + alpha * p)
        r = q(r - alpha * Ap)
        rsnew = float(np.dot(r, r))
        beta = 0.0 if rsold == 0.0 else rsnew / rsold
        p = q(r + beta * p)
        rsold = rsnew

    residuals.append(float(np.linalg.norm(b - A @ x) / b_norm))
    return x, {"backend": backend, "precision": precision, "relative_residuals": residuals, "steps": max_steps}


if __name__ == "__main__":
    solution, info = main()
    print("low-precision CG")
    print("solution:", solution)
    print("relative residuals:", info["relative_residuals"])
