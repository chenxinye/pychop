import numpy as np
import pytest

import pychop
from pychop import Chop
from pychop.builtin import CPArray, CPFloat, cast_precision
from pychop.builtin.linalg import _ensure_backend_for_obj


def test_cpfloat_supports_auto_backend():
    pychop.backend("auto")
    half = Chop(exp_bits=5, sig_bits=10)

    x = CPFloat(1.234567, half)

    assert isinstance(x, CPFloat)
    assert isinstance(float(x), float)


def test_linalg_auto_backend_inference_for_cparray():
    pychop.backend("auto")
    half = Chop(exp_bits=5, sig_bits=10)
    x = CPArray(np.array([1.0, 2.0]), half)

    assert _ensure_backend_for_obj(x) == "numpy"
    assert pychop.get_backend() == "numpy"


def test_chop_constructed_under_auto_initializes_after_backend_selection():
    pychop.backend("auto")
    half = Chop(exp_bits=5, sig_bits=10)
    pychop.backend("numpy")

    x = CPArray(np.array([1.0, 2.0]), half)

    assert isinstance(x, CPArray)


def test_cast_precision_preserves_cparray_family():
    pychop.backend("numpy")
    half = Chop(exp_bits=5, sig_bits=10)
    fp8 = Chop(exp_bits=4, sig_bits=3)
    x = CPArray(np.array([1.25, 2.5]), half)

    y = cast_precision(x, fp8)

    assert isinstance(y, CPArray)
    assert y.chopper is fp8


def test_cast_precision_wraps_native_scalar_as_cpfloat():
    pychop.backend("numpy")
    fp8 = Chop(exp_bits=4, sig_bits=3)

    y = cast_precision(1.25, fp8)

    assert isinstance(y, CPFloat)
    assert y.chopper is fp8

    z = cast_precision(np.float32(1.25), fp8)
    assert isinstance(z, CPFloat)
    assert z.chopper is fp8


def test_cptensor_preserves_nonfloating_torch_outputs():
    torch = pytest.importorskip("torch")
    pychop.backend("torch")
    half = Chop(exp_bits=5, sig_bits=10)
    from pychop.builtin import CPTensor

    x = CPTensor(torch.tensor([1.0, 2.0]), half)
    mask = x > 1.5

    assert isinstance(mask, torch.Tensor)
    assert not isinstance(mask, CPTensor)
    assert mask.dtype == torch.bool
