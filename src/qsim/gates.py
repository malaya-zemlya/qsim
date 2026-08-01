"""The gate set: the verbs of a quantum program.

**The physical fact this module makes concrete:** every gate is a *unitary* matrix —
one that preserves lengths. Since the length of the state vector is the total
probability, and total probability is always 1, "unitary" is just the mathematical
spelling of "this is a thing that can physically happen". Unitary maps are also
always invertible, which is why every quantum operation except measurement can be
run backwards.

Gates here are **callables that mutate the circuit and return ``None``**:

    H(a)
    CNOT(a, b)
    Rz(a, theta=np.pi/4)

A gate finds its circuit from the handles you give it, so there is no ``qc.h(a)``
form to keep in sync. Angles are keyword-only: every positional argument to a gate is
a qubit, so a bare number in that position would be the one thing a reader has to
stop and decode.

Every gate has two names
------------------------
The one-letter symbols are what the literature, the circuit diagrams and the recorded
history all use, but they are opaque until you have memorized them. So each gate is
also importable under its spelled-out name, and the two are the *same object*::

    Hadamard is H          # True
    PauliX is X            # True
    ControlledNot is CNOT  # True

Use whichever reads better on a given line; mixing them is fine. ``gate.name`` is the
short symbol and ``gate.full_name`` the long one. S and T are the two gates with no
settled name in the literature, so qsim names them for what they do: ``S = SqrtZ``
(S squared is Z) and ``T = FourthRootZ`` (T to the fourth is Z).

Two families, and why the distinction is structural
----------------------------------------------------
Some gates are **diagonal** in the computational basis (Z, S, T, Rz, Phase, CZ,
CPhase): they leave every |amplitude| alone and only rotate complex phases. Those
route through ``state.apply_diag``, which literally cannot change a magnitude. The
rest route through ``state.apply_1q``/``apply_2q``. Controlled gates of either kind
are applied by *slicing* the control axes — never by building a bigger matrix. A
Toffoli in qsim is the same 2x2 X matrix as an ordinary X, applied to a quarter of
the amplitudes.
"""

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from qsim.circuit import Op
from qsim.errors import QsimError

if TYPE_CHECKING:
    from qsim.circuit import Qubit

# ---------------------------------------------------------------------------
# Gate matrices. Each is written in the conventional form you would find in a
# textbook, with the basis order |0>, |1>.
# ---------------------------------------------------------------------------

_INV_SQRT2 = 1.0 / np.sqrt(2.0)

# H = (1/sqrt2) [[1,  1],
#                [1, -1]]
_H_MATRIX = _INV_SQRT2 * np.array([[1, 1], [1, -1]], dtype=np.complex128)

# X = [[0, 1],
#      [1, 0]]
_X_MATRIX = np.array([[0, 1], [1, 0]], dtype=np.complex128)

# Y = [[0, -i],
#      [i,  0]]
_Y_MATRIX = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)

# sqrt(X) = (1/2) [[1+i, 1-i],
#                  [1-i, 1+i]]      -- applied twice, it is X.
_SX_MATRIX = 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]], dtype=np.complex128)
_SX_DAGGER_MATRIX = _SX_MATRIX.conj().T

# Diagonal gates are stored as the diagonal only: the two numbers that multiply the
# |0> and |1> amplitudes. Z = diag(1, -1), S = diag(1, i), T = diag(1, e^{i pi/4}).
_Z_PHASES = np.array([1, -1], dtype=np.complex128)
_S_PHASES = np.array([1, 1j], dtype=np.complex128)
_S_DAGGER_PHASES = _S_PHASES.conj()
_T_PHASES = np.array([1, np.exp(1j * np.pi / 4)], dtype=np.complex128)
_T_DAGGER_PHASES = _T_PHASES.conj()


def _swap_tensor() -> np.ndarray:
    """SWAP as a (2,2,2,2) tensor indexed [out_j, out_k, in_j, in_k]."""
    # The entry is 1 exactly when the outputs are the inputs exchanged. Writing it
    # this way, rather than as a 4x4 matrix, keeps the two qubits' indices separate —
    # which is what lets it be contracted against two axes of the state tensor.
    tensor = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    for in_j in (0, 1):
        for in_k in (0, 1):
            tensor[in_k, in_j, in_j, in_k] = 1.0
    return tensor


_SWAP_TENSOR = _swap_tensor()


def _rx_matrix(theta: float) -> np.ndarray:
    """Rotation by ``theta`` about the x-axis of the Bloch sphere."""
    # The half-angles are not a typo. A qubit's state lives on a sphere, but the
    # underlying vector picks up only half the rotation angle: turning the Bloch
    # vector by a full 2*pi returns it to where it started while multiplying the
    # state by -1. Spin-1/2 systems really do behave this way.
    half = theta / 2.0
    return np.array(
        [[np.cos(half), -1j * np.sin(half)], [-1j * np.sin(half), np.cos(half)]],
        dtype=np.complex128,
    )


def _ry_matrix(theta: float) -> np.ndarray:
    """Rotation by ``theta`` about the y-axis of the Bloch sphere."""
    half = theta / 2.0
    return np.array(
        [[np.cos(half), -np.sin(half)], [np.sin(half), np.cos(half)]],
        dtype=np.complex128,
    )


def _rz_phases(theta: float) -> np.ndarray:
    """Rotation by ``theta`` about the z-axis, as a diagonal."""
    # Symmetric about 0: |0> picks up e^{-i theta/2} and |1> picks up e^{+i theta/2}.
    # Only the *difference* between the two is observable, so the symmetric form is a
    # convention -- it just keeps the gate's determinant equal to 1.
    return np.array([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)], dtype=np.complex128)


def _phase_phases(theta: float) -> np.ndarray:
    """Multiply the |1> amplitude by e^{i theta}, leaving |0> alone."""
    return np.array([1.0, np.exp(1j * theta)], dtype=np.complex128)


# ---------------------------------------------------------------------------
# The gate machinery
# ---------------------------------------------------------------------------

# How a gate reaches the state. "diag" gates carry a (2,) diagonal; "unitary1" a
# (2,2) matrix; "unitary2" a (2,2,2,2) tensor.
_DIAG = "diag"
_UNITARY1 = "unitary1"
_UNITARY2 = "unitary2"


class _GateBase:
    """Shared machinery: validation, recording, and the two universal combinators.

    Every gate can be **inverted** (``gate.adjoint()``) and **controlled**
    (``gate.controlled()``), and both return another gate — so they compose:
    ``T.adjoint().controlled()`` is a perfectly good controlled-T†. Blocks
    (``@qsim.gate``) offer the same two methods, so one vocabulary covers both.
    """

    def __init__(
        self, name: str, kind: str, n_controls: int, n_targets: int, full_name: str = ""
    ) -> None:
        #: The short symbol. This is what appears in the history, in ``gate_counts()``
        #: and on circuit diagrams, where a long name would not fit.
        self.name = name
        #: The spelled-out name. Every gate is also importable under this name, so
        #: ``Hadamard`` and ``H`` are the same object.
        self.full_name = full_name or name
        self._kind = kind
        self.n_controls = n_controls
        self.n_targets = n_targets
        self._adjoint_cache: _GateBase | None = None

    @property
    def label(self) -> str:
        """Both names together, for error messages: ``"H (Hadamard)"``."""
        return self.name if self.full_name == self.name else f"{self.name} ({self.full_name})"

    @property
    def n_qubits(self) -> int:
        """How many qubit handles this gate must be given."""
        return self.n_controls + self.n_targets

    def _emit(self, qubits: tuple[Qubit, ...], params: tuple[float, ...]) -> None:
        """Validate the handles and hand the circuit an ``Op`` describing this call.

        Note what this does *not* do: touch the state. Since Phase 2 the circuit
        decides whether an op runs now or is recorded for later transformation by a
        combinator scope, and axis numbers are resolved at that point rather than here
        — which is why an op stores qubit *ids*.
        """
        if len(qubits) != self.n_qubits:
            raise QsimError(
                f"{self.label} acts on {self.n_qubits} qubit(s) "
                f"({self.n_controls} control(s) and {self.n_targets} target(s)), "
                f"but got {len(qubits)}."
            )
        circuit = qubits[0]._circuit
        # Rejects a qubit passed twice (no-cloning), a released qubit, and handles from
        # two different circuits. Raises here, at the call site, even in record mode —
        # an error is far more useful pointing at the line that caused it.
        circuit._validate(qubits)

        circuit._emit(
            Op(
                name=self.name,
                qubit_ids=tuple(q._id for q in qubits[self.n_controls :]),
                params=params,
                controls=tuple(q._id for q in qubits[: self.n_controls]),
                block=circuit._current_block,
                gate=self,
            )
        )

    def _data_for(self, params: tuple[float, ...]) -> np.ndarray:
        """The matrix, diagonal or tensor this gate applies for the given parameters."""
        raise NotImplementedError  # pragma: no cover - both subclasses override it

    def adjoint(self) -> _GateBase:
        """The gate that undoes this one. ``H.adjoint() is H``; ``S.adjoint()`` is S†.

        Every gate has one, because every gate is a unitary and unitaries are always
        invertible — which is the same fact as "no information is lost". Contrast a
        classical AND gate: given only its output you cannot recover its inputs, so
        there is no gate that undoes it.
        """
        # Built on demand and cached, then wired up mutually. Doing it lazily is what
        # stops a controlled-S from recursing forever while trying to build the
        # controlled-S† that would in turn need a controlled-S.
        if self._adjoint_cache is None:
            partner = self._build_adjoint()
            self._adjoint_cache = partner
            partner._adjoint_cache = self
        return self._adjoint_cache

    def _build_adjoint(self) -> _GateBase:
        raise NotImplementedError  # pragma: no cover - both subclasses override it

    def adjoint_op(self, op: Op) -> Op:
        """The recorded operation that undoes ``op`` — used by ``with qc.adjoint()``."""
        raise NotImplementedError  # pragma: no cover - both subclasses override it

    def __repr__(self) -> str:
        return f"<Gate {self.label} on {self.n_qubits} qubit(s)>"


class Gate(_GateBase):
    """A gate with no parameters: ``H(a)``, ``CNOT(a, b)``."""

    def __init__(
        self,
        name: str,
        kind: str,
        data: np.ndarray,
        *,
        n_controls: int = 0,
        n_targets: int = 1,
        full_name: str = "",
    ) -> None:
        super().__init__(name, kind, n_controls, n_targets, full_name)
        self._data = data
        # Most fixed gates are their own inverse (applying X twice does nothing). S, T
        # and SX are the exceptions; their partners are wired up after both exist.
        self._explicit_inverse: Gate | None = None
        # Set by controlled(), so a derived gate can work out its own inverse.
        self._controlled_from: tuple[Gate, int] | None = None

    def __call__(self, *qubits: Qubit) -> None:
        self._emit(qubits, ())

    def _data_for(self, params: tuple[float, ...]) -> np.ndarray:
        return self._data

    def adjoint(self) -> Gate:
        """The gate that undoes this one. ``H.adjoint() is H``; ``S.adjoint()`` is S†."""
        return cast(Gate, super().adjoint())

    def controlled(self, n: int = 1, *, name: str = "", full_name: str = "") -> Gate:
        """A version of this gate that only fires when ``n`` further qubits are all 1.

        The controls come first in the argument list. Nothing about the underlying
        matrix changes — control is implemented by slicing the state, so this is the
        same gate applied to a smaller piece of it.
        """
        derived = Gate(
            name or "C" * n + self.name,
            self._kind,
            self._data,
            n_controls=self.n_controls + n,
            n_targets=self.n_targets,
            full_name=full_name or "Controlled" * n + self.full_name,
        )
        derived._controlled_from = (self, n)
        return derived

    def _build_adjoint(self) -> Gate:
        if self._explicit_inverse is not None:
            return self._explicit_inverse
        if self._controlled_from is not None:
            base, n = self._controlled_from
            base_inverse = base.adjoint()
            if base_inverse is base:
                return self  # controlling a self-inverse gate keeps it self-inverse
            return base_inverse.controlled(n)
        return self

    def adjoint_op(self, op: Op) -> Op:
        """The recorded operation that undoes ``op`` — used by ``with qc.adjoint()``."""
        inverse = self.adjoint()
        return Op(
            name=inverse.name,
            qubit_ids=op.qubit_ids,
            params=op.params,
            controls=op.controls,
            block=op.block,
            gate=inverse,
        )


class ParametrizedGate(_GateBase):
    """A gate that takes an angle: ``Rz(a, theta=np.pi/4)``.

    The angle is keyword-only on purpose — see this module's docstring.
    """

    def __init__(
        self,
        name: str,
        kind: str,
        data_fn: Any,
        *,
        n_controls: int = 0,
        n_targets: int = 1,
        full_name: str = "",
        sign: float = 1.0,
    ) -> None:
        super().__init__(name, kind, n_controls, n_targets, full_name)
        self._data_fn = data_fn
        # +1 for the gate itself, -1 for its adjoint. A rotation is undone by rotating
        # the other way, so the inverse of a parametrized gate is the same gate reading
        # its angle backwards — no daggered partner is needed.
        self._sign = sign

    def __call__(self, *qubits: Qubit, theta: float) -> None:
        # The recorded angle is the one actually applied, so an adjoint's history reads
        # Rz(theta=-0.3) rather than Rz(theta=0.3) with a hidden minus sign.
        self._emit(qubits, (self._sign * theta,))

    def _data_for(self, params: tuple[float, ...]) -> np.ndarray:
        return self._data_fn(params[0])

    def adjoint(self) -> ParametrizedGate:
        """The same rotation, read backwards: ``Rz.adjoint()(q, theta=t)`` applies −t."""
        return cast(ParametrizedGate, super().adjoint())

    def controlled(
        self, n: int = 1, *, name: str = "", full_name: str = ""
    ) -> ParametrizedGate:
        """A version of this gate that only fires when ``n`` further qubits are all 1."""
        return ParametrizedGate(
            name or "C" * n + self.name,
            self._kind,
            self._data_fn,
            n_controls=self.n_controls + n,
            n_targets=self.n_targets,
            full_name=full_name or "Controlled" * n + self.full_name,
            sign=self._sign,
        )

    def _build_adjoint(self) -> ParametrizedGate:
        return ParametrizedGate(
            self.name,
            self._kind,
            self._data_fn,
            n_controls=self.n_controls,
            n_targets=self.n_targets,
            full_name=self.full_name,
            sign=-self._sign,
        )

    def adjoint_op(self, op: Op) -> Op:
        """The recorded operation that undoes ``op``: the same rotation backwards.

        A rotation by theta is undone by a rotation by -theta. No lookup table
        needed — this is why parametrized gates are easier to invert than fixed ones.
        """
        return Op(
            name=self.name,
            qubit_ids=op.qubit_ids,
            params=tuple(-p for p in op.params),
            controls=op.controls,
            block=op.block,
            gate=self,
        )


# ---------------------------------------------------------------------------
# The public gate set
# ---------------------------------------------------------------------------

H = Gate("H", _UNITARY1, _H_MATRIX, full_name="Hadamard")
"""Hadamard: takes |0> to the equal superposition (|0> + |1>)/sqrt(2), and |1> to
(|0> - |1>)/sqrt(2). It is how "both at once" enters a computation — and, applied a
second time, how the two paths are brought back together to interfere."""

X = Gate("X", _UNITARY1, _X_MATRIX, full_name="PauliX")
"""Pauli-X, the bit flip: swaps |0> and |1>. The quantum NOT gate, and a 180-degree
rotation about the x-axis of the Bloch sphere."""

Y = Gate("Y", _UNITARY1, _Y_MATRIX, full_name="PauliY")
"""Pauli-Y: a bit flip and a phase flip at once (Y = iXZ); a 180-degree rotation
about the y-axis."""

Z = Gate("Z", _DIAG, _Z_PHASES, full_name="PauliZ")
"""Pauli-Z, the phase flip: leaves |0> alone and negates |1>. It does nothing
observable to a qubit in |0> or |1>, and turns |+> into |-> — invisible to
measurement until something interferes with it."""

S = Gate("S", _DIAG, _S_PHASES, full_name="SqrtZ")
"""A quarter turn about the z-axis: multiplies the |1> amplitude by i. S applied
twice is Z, which is why its full name is ``SqrtZ``. (It is often called "the phase
gate" in the literature; that name is taken here by the parametrized ``Phase``.)"""

_S_DAGGER = Gate("S†", _DIAG, _S_DAGGER_PHASES, full_name="SqrtZDagger")

T = Gate("T", _DIAG, _T_PHASES, full_name="FourthRootZ")
"""An eighth turn about z: multiplies the |1> amplitude by e^{i pi/4}. Applied four
times it is Z, hence ``FourthRootZ``; applied twice it is S.

T is the non-Clifford ingredient — the one that makes quantum circuits hard to
simulate classically. Circuits built only from H, S and CNOT can be simulated
efficiently on an ordinary computer; adding T destroys that."""

_T_DAGGER = Gate("T†", _DIAG, _T_DAGGER_PHASES, full_name="FourthRootZDagger")

SX = Gate("SX", _UNITARY1, _SX_MATRIX, full_name="SqrtX")
"""The square root of X: applied twice, it is a bit flip. There is no such thing as
"half of a classical NOT", which is a compact illustration that the space of quantum
operations is bigger than the classical one."""

_SX_DAGGER = Gate("SX†", _UNITARY1, _SX_DAGGER_MATRIX, full_name="SqrtXDagger")

SWAP = Gate("SWAP", _UNITARY2, _SWAP_TENSOR, n_targets=2, full_name="Swap")
"""Exchanges two qubits. Equal to three alternating CNOTs — CNOT(a,b), CNOT(b,a),
CNOT(a,b) — which notebook 01 checks by hand. Here it is applied as one two-qubit
tensor instead, so that ``apply_2q`` has a user."""

CNOT = X.controlled(name="CNOT", full_name="ControlledNot")
"""Flips the target if the control is 1. The workhorse entangling gate: applied to
(|0> + |1>)/sqrt(2) on the control it produces a Bell pair, a state that cannot be
written as one qubit's state times another's."""

CZ = Z.controlled(name="CZ", full_name="ControlledZ")
"""Negates only the |11> amplitude. Note it is symmetric: "apply Z to b if a is 1" and
"apply Z to a if b is 1" are the same operation, so for this gate the labels
"control" and "target" are ours, not nature's."""

Toffoli = X.controlled(2, name="Toffoli", full_name="ControlledControlledNot")
"""Flips the target if *both* controls are 1 — the reversible AND. Classical
computation embeds into quantum computation through this gate, which is why Phase 4
can build adders out of it."""

Fredkin = SWAP.controlled(name="Fredkin", full_name="ControlledSwap")
"""Swaps two targets if the control is 1 — the controlled-SWAP, and the other
classical-universal reversible gate."""

Rx = ParametrizedGate("Rx", _UNITARY1, _rx_matrix, full_name="RotationX")
"""Rotate the Bloch vector by ``theta`` about the x-axis: ``Rx(q, theta=np.pi)`` is X
up to an overall phase."""

Ry = ParametrizedGate("Ry", _UNITARY1, _ry_matrix, full_name="RotationY")
"""Rotate by ``theta`` about the y-axis. The rotation with real matrix entries, so it
moves amplitude between |0> and |1> without introducing complex phases."""

Rz = ParametrizedGate("Rz", _DIAG, _rz_phases, full_name="RotationZ")
"""Rotate by ``theta`` about the z-axis. Being diagonal, it cannot change any
measurement probability in the computational basis — only phases."""

Phase = ParametrizedGate("Phase", _DIAG, _phase_phases)
"""Multiply the |1> amplitude by e^{i theta}, leaving |0> untouched. Same physical
effect as Rz up to an overall phase, but with the convention that |0> is the one
left alone."""

CPhase = Phase.controlled(name="CPhase", full_name="ControlledPhase")
"""Multiply only the |11> amplitude by e^{i theta}. The engine of the QFT (Phase 3),
where the angles are the binary place values of a number."""


# ---------------------------------------------------------------------------
# Spelled-out aliases
#
# Each of these is the *same object* as its short form: ``Hadamard is H``. The
# short symbols are what the literature and circuit diagrams use, and what the
# recorded history stores; the long names are for code that would rather be read
# than decoded. Use whichever makes a given line clearer — mixing them is fine.
# ---------------------------------------------------------------------------

Hadamard = H
PauliX = X
PauliY = Y
PauliZ = Z
SqrtZ = S
FourthRootZ = T
SqrtX = SX
Swap = SWAP
ControlledNot = CNOT
ControlledZ = CZ
ControlledControlledNot = Toffoli
ControlledSwap = Fredkin
RotationX = Rx
RotationY = Ry
RotationZ = Rz
ControlledPhase = CPhase


# The three gates that are not their own inverse, wired to their partners. Done here
# rather than in the constructors because each pair refers to the other, and the two
# objects cannot both exist before either is built.
S._explicit_inverse = _S_DAGGER
_S_DAGGER._explicit_inverse = S
T._explicit_inverse = _T_DAGGER
_T_DAGGER._explicit_inverse = T
SX._explicit_inverse = _SX_DAGGER
_SX_DAGGER._explicit_inverse = SX


#: Every gate by name, so recorded history can be replayed (Phase 2's combinators).
#: Keyed by the short symbol — the same string that appears in ``Op.name``.
GATES: dict[str, _GateBase] = {
    g.name: g
    for g in (
        H, X, Y, Z, S, _S_DAGGER, T, _T_DAGGER, SX, _SX_DAGGER, SWAP,
        CNOT, CZ, Toffoli, Fredkin, Rx, Ry, Rz, Phase, CPhase,
    )
}
