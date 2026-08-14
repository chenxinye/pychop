from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "mixed_precision"


def load_example(name: str):
    path = EXAMPLES / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def assert_example_result(module, **kwargs):
    x, info = module.main(**kwargs)
    assert isinstance(x, np.ndarray)
    assert x.shape == (4,)
    assert np.all(np.isfinite(x))
    assert "relative_residuals" in info
    assert np.isfinite(info["relative_residuals"][-1])


def test_mixed_precision_iterative_refinement_main():
    assert_example_result(load_example("mixed_precision_iterative_refinement.py"), max_steps=3)


def test_low_precision_gmres_main():
    assert_example_result(load_example("low_precision_gmres.py"), max_steps=8)


def test_low_precision_cg_main():
    assert_example_result(load_example("low_precision_cg.py"), max_steps=8)


def test_lu_qr_precision_switching_solve_main():
    module = load_example("lu_qr_precision_switching_solve.py")
    assert_example_result(module, method="lu")
    assert_example_result(module, method="qr")


def test_mixed_precision_residual_correction_main():
    assert_example_result(load_example("mixed_precision_residual_correction.py"), max_steps=4)
