"""qsim — a NumPy state-vector quantum simulator for learning quantum mechanics.

The whole simulator rests on one identification: the state of n qubits is a single
NumPy array of shape (2,) * n, and **the axes of that array are the tensor factors
of the Hilbert space** — axis k is qubit k. Entangling two qubits is a contraction
that ties two axes together; a gate on one qubit is a 2x2 matrix applied along one
axis. Everything else in this package is built from that picture.

Conceptual transparency beats speed here, always: this code is meant to be read.

A first program::

    import numpy as np
    import qsim
    from qsim import Circuit
    from qsim.gates import H, CNOT

    qc = Circuit(name="bell", seed=1234)
    a, b = qc.alloc_many(2)
    H(a)                       # a is now (|0> + |1>)/sqrt(2)
    CNOT(a, b)                 # ... and now the pair is entangled
    print(qc.inspect.ket())    # 0.707|00⟩ + 0.707|11⟩
    print(qc.measure(a) == qc.measure(b))   # always True

Where to look next: ``state.py`` for what a state *is*, ``circuit.py`` for who owns
it, ``measure.py`` for the one irreversible operation, and ``inspector.py`` for
everything a real quantum computer would never let you see. The notebooks in
``notebooks/`` are the guided path through all of it.
"""

from qsim import errors, gates, viz
from qsim.circuit import Circuit, Op, Qubit, Register
from qsim.combinators import Block, gate
from qsim.errors import (
    DeadQubitError,
    DirtyAncillaError,
    NoCloningError,
    QsimError,
)
from qsim.gates import (
    CNOT,
    CZ,
    SWAP,
    SX,
    ControlledControlledNot,
    ControlledNot,
    ControlledPhase,
    ControlledSwap,
    ControlledZ,
    CPhase,
    FourthRootZ,
    Fredkin,
    Gate,
    H,
    Hadamard,
    ParametrizedGate,
    PauliX,
    PauliY,
    PauliZ,
    Phase,
    RotationX,
    RotationY,
    RotationZ,
    Rx,
    Ry,
    Rz,
    S,
    SqrtX,
    SqrtZ,
    Swap,
    T,
    Toffoli,
    X,
    Y,
    Z,
)
from qsim.inspector import Bra, Inspector, Ket
from qsim.state import get_dtype, set_dtype

__all__ = [
    # core objects
    "Circuit",
    "Op",
    "Qubit",
    "Register",
    "Inspector",
    "Ket",
    "Bra",
    # combinators
    "Block",
    "gate",
    # gates
    "Gate",
    "ParametrizedGate",
    "H",
    "X",
    "Y",
    "Z",
    "S",
    "T",
    "SX",
    "CNOT",
    "CZ",
    "SWAP",
    "Toffoli",
    "Fredkin",
    "Rx",
    "Ry",
    "Rz",
    "Phase",
    "CPhase",
    # gates, spelled out — the same objects under longer names
    "Hadamard",
    "PauliX",
    "PauliY",
    "PauliZ",
    "SqrtZ",
    "FourthRootZ",
    "SqrtX",
    "Swap",
    "ControlledNot",
    "ControlledZ",
    "ControlledControlledNot",
    "ControlledSwap",
    "RotationX",
    "RotationY",
    "RotationZ",
    "ControlledPhase",
    # errors
    "QsimError",
    "NoCloningError",
    "DeadQubitError",
    "DirtyAncillaError",
    # precision
    "set_dtype",
    "get_dtype",
    # submodules
    "errors",
    "gates",
    "viz",
]
