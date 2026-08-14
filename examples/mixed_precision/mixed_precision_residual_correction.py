"""Mixed-precision residual correction example.

Run directly:
    python examples/mixed_precision/mixed_precision_residual_correction.py
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
            [6.0, -1.0, 0.5, 0.0],
            [-1.0, 5.0, -1.0, 0.5],
            [0.5, -1.0, 4.5, -1.0],
            [0.0, 0.5, -1.0, 4.0],
        ],
        dtype=float,
    )
    b = np.array([1.0, 0.0, 2.0, -1.0], dtype=float)
    return A, b


def main(
    A: np.ndarray | None = None,
    b: np.ndarray | None = None,
    precision: str | dict[str, Any] | None = "fp16",
    max_steps: int = 12,
    tol: float = 1e-8,
    damping: float = 1.0,
    backend: str = "numpy",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply low-precision diagonal residual corrections with high-precision checks."""
    pychop.backend(backend)
    A, b = default_problem() if A is None or b is None else (np.asarray(A, dtype=float), np.asarray(b, dtype=float))
    chop = make_chopper(precision)
    q = lambda x: np.asarray(chop(np.asarray(x, dtype=float)), dtype=float)

    A_low = q(A)
    diagonal = q(np.diag(A_low))
    x = np.zeros_like(b)
    b_norm = max(np.linalg.norm(b), np.finfo(float).tiny)
    residuals: list[float] = []

    for step in range(max_steps):
        r = b - A @ x
        rel = float(np.linalg.norm(r) / b_norm)
        residuals.append(rel)
        if rel <= tol:
            return x, {"backend": backend, "precision": precision, "relative_residuals": residuals, "steps": step}
        correction = q(q(r) / diagonal)
        x = q(x + damping * correction)

    residuals.append(float(np.linalg.norm(b - A @ x) / b_norm))
    return x, {"backend": backend, "precision": precision, "relative_residuals": residuals, "steps": max_steps}


if __name__ == "__main__":
    solution, info = main()
    print("mixed-precision residual correction")
    print("solution:", solution)
    print("relative residuals:", info["relative_residuals"])
