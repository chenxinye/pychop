"""Mixed-precision iterative refinement example.

Run directly:
    python examples/mixed_precision/mixed_precision_iterative_refinement.py
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
    """Create a pychop chopper from a named or explicit precision."""
    spec: dict[str, Any]
    if precision is None:
        spec = dict(_PRECISIONS["fp16"])
    elif isinstance(precision, str):
        spec = dict(_PRECISIONS[precision.lower()])
    else:
        spec = dict(precision)
    spec.setdefault("rmode", 1)
    spec.setdefault("subnormal", True)
    return Chop(**spec)


def default_problem() -> tuple[np.ndarray, np.ndarray]:
    """Small diagonally dominant system for repeatable direct runs."""
    A = np.array(
        [
            [4.0, 1.0, 0.5, 0.0],
            [1.0, 3.5, 0.0, 0.25],
            [0.5, 0.0, 3.0, 1.0],
            [0.0, 0.25, 1.0, 2.75],
        ],
        dtype=float,
    )
    b = np.array([1.0, 2.0, 0.5, -1.0], dtype=float)
    return A, b


def main(
    A: np.ndarray | None = None,
    b: np.ndarray | None = None,
    precision: str | dict[str, Any] | None = "fp16",
    max_steps: int = 6,
    tol: float = 1e-10,
    backend: str = "numpy",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve Ax=b using a low-precision factor solve and high-precision residuals."""
    pychop.backend(backend)
    A, b = default_problem() if A is None or b is None else (np.asarray(A, dtype=float), np.asarray(b, dtype=float))
    chop = make_chopper(precision)
    q = lambda x: np.asarray(chop(np.asarray(x, dtype=float)), dtype=float)

    A_low = q(A)
    b_low = q(b)
    x = q(np.linalg.solve(A_low, b_low))
    b_norm = max(np.linalg.norm(b), np.finfo(float).tiny)
    residuals: list[float] = []

    for _ in range(max_steps + 1):
        r = b - A @ x
        rel = float(np.linalg.norm(r) / b_norm)
        residuals.append(rel)
        if rel <= tol:
            break
        correction = np.linalg.solve(A_low, q(r))
        x = q(x + correction)

    return x, {"backend": backend, "precision": precision, "relative_residuals": residuals}


if __name__ == "__main__":
    solution, info = main()
    print("mixed-precision iterative refinement")
    print("solution:", solution)
    print("relative residuals:", info["relative_residuals"])
