"""Smoke tests: the toolchain itself works before any quantum code exists."""

import sys

import numpy as np

import qsim
from qsim.errors import DeadQubitError, DirtyAncillaError, NoCloningError


def test_python_version() -> None:
    assert sys.version_info >= (3, 14)


def test_numpy_present() -> None:
    # Quantum amplitudes are complex numbers, so every state array in qsim is
    # complex128 (two 64-bit floats: real and imaginary part).
    assert np.zeros((2, 2), dtype=np.complex128).dtype == np.complex128


def test_errors_importable() -> None:
    """Every qsim error is a QsimError, so one `except` clause catches them all."""
    for exc in (NoCloningError, DeadQubitError, DirtyAncillaError):
        assert issubclass(exc, qsim.errors.QsimError)
