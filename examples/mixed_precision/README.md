# Mixed-precision algorithm examples

Each file in this directory is self-contained, can be run directly, and exposes a
`main(A, b, precision, ...)` entry point for compatibility with experiment
drivers.  The examples use the NumPy backend by default and avoid notebook-only
algorithm code.

Classic examples currently included:

- `mixed_precision_iterative_refinement.py`
- `low_precision_gmres.py`
- `low_precision_cg.py`
- `lu_qr_precision_switching_solve.py`
- `mixed_precision_residual_correction.py`

Research-oriented examples such as stochastic-rounding iterative methods,
adaptive-precision Krylov methods, low-precision preconditioned solvers, and
mixed-precision optimization examples can be added later using the same file
layout and `main` signature.
