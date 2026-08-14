"""Precision-casting helpers for chopped-precision containers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pychop


def cast_precision(x: Any, chopper: Any) -> Any:
    """
    Cast a value to a chopped-precision container using ``chopper``.

    Existing ``CPFloat``, ``CPArray``, ``CPJaxArray``, and ``CPTensor`` inputs
    keep their container family. Native arrays/tensors are wrapped according to
    the active backend, or by auto-detecting the input type when the backend is
    ``"auto"``. The cast chops immediately.
    """
    from .cpfloat import CPFloat
    from .cparray import CPArray
    try:
        from .cparray_jax import CPJaxArray
    except ImportError:
        CPJaxArray = None  # type: ignore
    try:
        from .cptensor import CPTensor
    except ImportError:
        CPTensor = None  # type: ignore

    if isinstance(x, CPFloat):
        return CPFloat(x.value, chopper)
    if isinstance(x, CPArray):
        return CPArray(x.to_regular(), chopper)
    if CPJaxArray is not None and isinstance(x, CPJaxArray):
        return CPJaxArray(x.to_regular(), chopper)
    if CPTensor is not None and isinstance(x, CPTensor):
        return CPTensor(x.to_regular(), chopper)
    if isinstance(x, (int, float, complex, np.number)):
        return CPFloat(x, chopper)

    backend = pychop.get_backend()
    if backend == "auto":
        from pychop.utils import detect_array_type
        backend = detect_array_type(x)

    if backend == "numpy":
        return CPArray(x, chopper)
    if backend == "jax":
        from .cparray_jax import CPJaxArray
        return CPJaxArray(x, chopper)
    if backend == "torch":
        from .cptensor import CPTensor
        return CPTensor(x, chopper)

    raise ValueError(f"Unsupported backend for precision cast: {backend!r}")


def cast(x: Any, chopper: Any) -> Any:
    """Alias for :func:`cast_precision`."""
    return cast_precision(x, chopper)
