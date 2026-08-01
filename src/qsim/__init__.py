"""qsim — a NumPy state-vector quantum simulator for learning quantum mechanics.

The whole simulator rests on one identification: the state of n qubits is a single
NumPy array of shape (2,) * n, and **the axes of that array are the tensor factors
of the Hilbert space** — axis k is qubit k. Entangling two qubits is a contraction
that ties two axes together; a gate on one qubit is a 2x2 matrix applied along one
axis. Everything else in this package is built from that picture.

Conceptual transparency beats speed here, always: this code is meant to be read.

Phase 0 ships only the error types; the quantum machinery arrives in Phase 1.
"""

from qsim import errors
from qsim.errors import (
    DeadQubitError,
    DirtyAncillaError,
    NoCloningError,
    QsimError,
)

__all__ = [
    "DeadQubitError",
    "DirtyAncillaError",
    "NoCloningError",
    "QsimError",
    "errors",
]
