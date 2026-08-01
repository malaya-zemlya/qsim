"""Circuits, qubit handles, and registers — who owns what.

**The physical fact this module makes concrete:** there is only ever *one* state,
and it belongs to the circuit. Individual qubits do not carry states around with
them; they are names for axes of the one shared tensor. Once two qubits are
entangled, neither has a state of its own to report — not because we failed to
track it, but because there isn't one (``02-entanglement.ipynb`` shows this).

The object model follows from that:

- ``Circuit`` owns the state tensor, the id -> axis table, and the recorded history.
- ``Qubit`` is a handle. It stores a stable id and asks the circuit where that id
  currently lives whenever it is used.
- ``Register`` is an ordered group of handles, so that algorithms written over many
  qubits read like arithmetic instead of index juggling.

Why handles store ids and not axis numbers
-------------------------------------------
Allocating a qubit mid-circuit adds an axis; releasing an ancilla (Phase 2) *removes*
one, which shifts the position of every axis after it. If a handle remembered "I am
axis 3", every handle after the released one would silently start pointing at the
wrong qubit — the worst kind of bug, because the program keeps running and just
computes something else. So a handle remembers "I am qubit id 3", the circuit owns
the single table from ids to axes, and every operation looks the axis up at the
moment it acts. The story "axis k is qubit k" survives, one indirection away.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

import numpy as np

from qsim import state
from qsim.errors import DeadQubitError, NoCloningError, QsimError

if TYPE_CHECKING:
    from qsim.inspector import Inspector


@dataclass(frozen=True)
class Op:
    """One recorded operation: what was done, to which qubits, with what parameters.

    The history is a record of the *program*, not of the state. It is what
    ``gate_counts`` and ``depth`` summarize, what Phase 2's combinators transform to
    build controlled and inverted blocks, and what Phase 6 draws as a diagram.
    """

    name: str
    qubit_ids: tuple[int, ...]
    params: tuple[float, ...] = ()
    controls: tuple[int, ...] = ()
    # Measurement is the one operation whose outcome is not determined by the program,
    # so it is the one operation that records a result.
    result: int | None = None

    @property
    def all_qubit_ids(self) -> tuple[int, ...]:
        """Every qubit this op touches, controls included."""
        return self.controls + self.qubit_ids


class Qubit:
    """A handle to one axis of a circuit's state — not a value.

    A ``Qubit`` does not have a state. There is one joint state tensor owned by the
    ``Circuit``; a ``Qubit`` names one of its axes. Once entangled, an individual
    qubit has no pure state to report — asking for one is a category error, not a
    missing feature. Use ``circuit.inspect.reduced_density_matrix()`` if you want the
    mixed state of a subsystem.

    That is why this class has no ``.state``, no ``.value``, no ``.amplitude``, and no
    ``__bool__``: every one of them would be a lie that reads like a convenience.
    """

    __slots__ = ("_circuit", "_id", "_name", "_live")

    def __init__(self, circuit: Circuit, qubit_id: int, name: str) -> None:
        self._circuit = circuit
        self._id = qubit_id
        self._name = name
        self._live = True

    @property
    def name(self) -> str:
        """The qubit's label, used in diagrams and error messages."""
        return self._name

    def __copy__(self) -> Qubit:
        raise NoCloningError(
            f"cannot copy {self._name}. This is the no-cloning theorem: there is no "
            "physical operation that copies an unknown quantum state, because copying "
            "is not a linear map and quantum evolution is linear. Copying the handle "
            "would suggest otherwise. If two variables should refer to the same qubit, "
            "plain assignment does that; if you want a second qubit prepared the same "
            "way, allocate one and repeat the gates that prepared this one."
        )

    def __deepcopy__(self, memo: dict[int, Any]) -> Qubit:
        return self.__copy__()

    def __eq__(self, other: object) -> bool:
        # Identity, not value. Two handles are the same qubit only if they *are* the
        # same handle — there is no state to compare, and equal states would not mean
        # equal qubits anyway.
        return self is other

    def __hash__(self) -> int:
        return id(self)

    def __repr__(self) -> str:
        label = f" of Circuit {self._circuit.name!r}" if self._circuit.name else ""
        dead = "" if self._live else ", released"
        return f"<Qubit {self._name}{label}{dead}>"


class Register(Sequence[Qubit]):
    """An ordered sequence of qubits, addressed as a group.

    Registers exist because algorithms think in numbers, not in individual qubits:
    Shor's algorithm is arithmetic on registers, and writing it with bare indices
    would bury the physics in bookkeeping.

    Throughout qsim, **``reg[0]`` is the most significant bit** — the same convention
    the state tensor uses (see ``state.py``).
    """

    __slots__ = ("_qubits", "_name")

    def __init__(self, qubits: tuple[Qubit, ...], name: str = "") -> None:
        self._qubits = qubits
        self._name = name

    @property
    def name(self) -> str:
        """The register's label."""
        return self._name

    @overload
    def __getitem__(self, index: int) -> Qubit: ...

    @overload
    def __getitem__(self, index: slice) -> Register: ...

    def __getitem__(self, index: int | slice) -> Qubit | Register:
        # A slice returns a Register *view* over the same handles — no qubits are
        # copied or reallocated, because there is nothing to copy: handles are names.
        if isinstance(index, slice):
            return Register(self._qubits[index], name=self._name)
        return self._qubits[index]

    def __len__(self) -> int:
        return len(self._qubits)

    def __iter__(self) -> Iterator[Qubit]:
        return iter(self._qubits)

    def reversed(self) -> Register:
        """The same qubits in the opposite order.

        Needed constantly by the QFT (Phase 3), whose output comes out bit-reversed.
        """
        return Register(tuple(reversed(self._qubits)), name=self._name)

    def concat(self, other: Register) -> Register:
        """The qubits of this register followed by the qubits of ``other``."""
        return Register(self._qubits + tuple(other), name=self._name)

    def encode(self, value: int) -> None:
        """Set this register to the basis state |value>, MSB first.

        A simulator-only convenience for *preparing inputs* — real hardware would do
        the same thing (it is just X gates), but only because the register is known
        to start in |0...0>. That precondition is checked here rather than assumed:
        applying X gates to a register already in superposition would not "set" it to
        anything, it would scramble it.
        """
        size = len(self._qubits)
        if not 0 <= value < 2**size:
            raise ValueError(
                f"cannot encode {value} into a {size}-qubit register: it holds values "
                f"0 to {2**size - 1}."
            )
        circuit = self._qubits[0]._circuit
        # Refuse unless every qubit here really is |0>. assert_zero raises a message
        # explaining why, so we let it speak.
        circuit.inspect.assert_zero(self)

        from qsim.gates import X

        for i, q in enumerate(self._qubits):
            # reg[0] is the most significant bit, so qubit i carries bit
            # (size - 1 - i) of the value.
            if (value >> (size - 1 - i)) & 1:
                X(q)

    def __repr__(self) -> str:
        label = f" {self._name!r}" if self._name else ""
        return f"<Register{label} of {len(self._qubits)} qubits>"


class Circuit:
    """A quantum circuit: the owner of one state tensor and the qubits in it.

    The circuit *is* the qubit pool. It hands out handles (``alloc``), records what
    was done to them (``history``), and from Phase 2 on reclaims them (``ancilla``).

    Pass ``seed`` to make measurement reproducible — the randomness of quantum
    measurement is real, but a test that cannot be repeated is not a test.
    """

    def __init__(
        self,
        n: int = 0,
        *,
        name: str = "",
        dtype: Any = None,
        seed: int | None = None,
    ) -> None:
        self._name = name
        self._dtype = state.get_dtype() if dtype is None else np.dtype(dtype)
        # Zero qubits: a 1-dimensional space holding the single amplitude 1. Every
        # allocation below tensors one more |0> onto it, so there is no special case.
        self._psi = state.zero_state(0, self._dtype)
        self._axis_of: dict[int, int] = {}
        self._qubits: list[Qubit] = []
        self._next_id = 0
        self._history: list[Op] = []
        self._rng = np.random.default_rng(seed)
        # A second, independent stream used only by inspect.sample(). Sampling is a
        # simulator cheat, not a physical measurement, so it must not consume draws
        # from the stream that real measurements use — otherwise adding a sample()
        # call to a notebook would silently change every measurement after it.
        self._sample_rng = np.random.default_rng(seed)
        for _ in range(n):
            self.alloc()

    # ---- properties ------------------------------------------------------------

    @property
    def name(self) -> str:
        """The circuit's label."""
        return self._name

    @property
    def n_qubits(self) -> int:
        """How many qubits are currently allocated."""
        return len(self._axis_of)

    @property
    def qubits(self) -> Register:
        """Every live qubit, in allocation order.

        Mostly for the ``Circuit(n)`` form, which allocates qubits before you have
        anything to name them with: ``a, b = Circuit(2).qubits``.
        """
        return Register(tuple(self._qubits), name=self._name)

    @property
    def history(self) -> list[Op]:
        """Every operation applied so far, in order."""
        return list(self._history)

    @property
    def inspect(self) -> Inspector:
        """Tomography-style introspection — everything impossible on real hardware.

        See ``inspector.py``: the boundary of this namespace is the boundary between
        what the math knows and what an experiment could actually extract.
        """
        # Imported here rather than at module scope: inspector.py needs the types
        # defined in this module, so a top-level import would be circular.
        from qsim.inspector import Inspector

        return Inspector(self)

    # ---- allocation ------------------------------------------------------------

    def alloc(self, name: str = "") -> Qubit:
        """Allocate one fresh qubit in state |0> and return a handle to it."""
        # Tensor product with a new qubit in |0>. np.multiply.outer appends one axis
        # of length 2 whose index-0 slice is a copy of the old state and whose index-1
        # slice is all zeros — i.e. the new qubit is |0>, and it is unentangled with
        # everything else, which is exactly what "a fresh qubit" means.
        self._psi = np.multiply.outer(self._psi, np.array([1, 0], dtype=self._dtype))
        qubit_id = self._next_id
        self._next_id += 1
        # The new axis is the last one, because outer() appended it.
        self._axis_of[qubit_id] = self._psi.ndim - 1
        qubit = Qubit(self, qubit_id, name or f"q{qubit_id}")
        self._qubits.append(qubit)
        return qubit

    def alloc_many(self, count: int) -> tuple[Qubit, ...]:
        """Allocate ``count`` fresh qubits and return their handles as a tuple.

        Separate from ``alloc`` rather than returning either one qubit or several:
        a function that returns different types depending on its argument forces
        every caller to re-check what it got.
        """
        if count < 1:
            raise ValueError(f"cannot allocate {count} qubits; ask for at least 1")
        return tuple(self.alloc() for _ in range(count))

    def register(self, size: int, *, name: str = "") -> Register:
        """Allocate ``size`` fresh qubits as a named ``Register``."""
        if size < 1:
            raise ValueError(f"cannot create a register of {size} qubits; ask for at least 1")
        return Register(tuple(self.alloc(f"{name}{i}" if name else "") for i in range(size)), name)

    # ---- physical operations ---------------------------------------------------

    # measure.py, gates.py and inspector.py all build on the types defined here, so
    # this module imports them where they are used rather than at the top. The
    # dependency really does run that way round: the circuit is the thing they act on.

    def measure(self, q: Qubit) -> int:
        """Measure ``q`` in the computational basis; returns 0 or 1. See ``measure.py``."""
        from qsim import measure as measure_mod

        return measure_mod.measure(self, q)

    def measure_all(self, reg: Register) -> int:
        """Measure every qubit of ``reg``, returning the result as an integer, MSB first."""
        from qsim import measure as measure_mod

        return measure_mod.measure_all(self, reg)

    def reset(self, q: Qubit) -> None:
        """Return ``q`` to |0> by measuring it and flipping it if it read 1."""
        from qsim import measure as measure_mod

        measure_mod.reset(self, q)

    # ---- history summaries -----------------------------------------------------

    def gate_counts(self) -> dict[str, int]:
        """How many times each operation appears in the history."""
        counts: dict[str, int] = {}
        for op in self._history:
            counts[op.name] = counts.get(op.name, 0) + 1
        return counts

    def depth(self) -> int:
        """Circuit depth: how many layers of simultaneous operations the program needs.

        Operations on disjoint qubits happen at the same time on real hardware, so
        depth — not gate count — is what sets how long a circuit takes to run, and
        how much time it has to decohere. Computed greedily: walk the history and
        start a new layer whenever an operation touches a qubit already used in the
        current layer.
        """
        layers = 0
        current: set[int] = set()
        for op in self._history:
            touched = set(op.all_qubit_ids)
            if current & touched:
                layers += 1
                current = touched
            else:
                current |= touched
        # The final in-progress layer counts too, unless there were no ops at all.
        return layers + 1 if self._history else 0

    # ---- internals used by gates.py, measure.py and inspector.py ----------------

    def _axis(self, q: Qubit) -> int:
        """Resolve one handle to its current axis, checking that it may be used."""
        if q._circuit is not self:
            raise QsimError(
                f"{q._name} belongs to a different circuit. Two circuits are two "
                "separate physical systems with separate states; a gate spanning both "
                "is not an operation that exists. Allocate all the qubits an operation "
                "touches from the same Circuit."
            )
        if not q._live:
            raise DeadQubitError(
                f"{q._name} has been released and no longer refers to a qubit. Its axis "
                "is gone from the state, so this handle names nothing physical — qsim "
                "refuses rather than silently acting on whichever qubit shifted into "
                "its place."
            )
        return self._axis_of[q._id]

    def _axes(self, qubits: Sequence[Qubit]) -> list[int]:
        """Resolve several handles at once, rejecting repeats."""
        for i, q in enumerate(qubits):
            for other in qubits[i + 1 :]:
                if q is other:
                    raise NoCloningError(
                        f"{q._name} was given twice to the same operation. A qubit "
                        "cannot control an operation on itself: that would require "
                        "reading its value to decide what to do to it, and then the "
                        "result would be a copy of that value. This is the no-cloning "
                        "theorem appearing at the API surface — the gate you want is "
                        "probably one acting on two different qubits."
                    )
        return [self._axis(q) for q in qubits]

    def _record(self, op: Op) -> None:
        self._history.append(op)

    def _repr_html_(self) -> str:
        # Jupyter calls this when a Circuit is the last expression in a cell.
        from qsim.viz import circuit_html

        return circuit_html(self)

    def __repr__(self) -> str:
        label = f" {self._name!r}" if self._name else ""
        return f"<Circuit{label} with {self.n_qubits} qubits, {len(self._history)} ops>"

