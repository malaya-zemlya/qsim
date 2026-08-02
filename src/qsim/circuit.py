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

The history is a tape
---------------------
Every operation that runs is appended to ``circuit.history``, and that record is not
just a log to print. Deep-learning libraries work the same way: they run each
operation immediately *and* record it, so that the recorded sequence can later be
walked backwards (that is what "backpropagation" walks). Ours can be walked backwards
too, and more cheaply, because every gate is invertible.

Three methods make the tape a thing you use rather than a thing you read:
:meth:`Circuit.checkpoint` marks a position, :meth:`Circuit.rewind` undoes everything
after a mark by running its inverses, and :meth:`Circuit.on_op` attaches an observer
that sees every operation as it happens. Notebook 04 uses all three.
"""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
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
    #: The block this op came from, if any — see ``@qsim.gate``. Grouping only: the op
    #: itself is an ordinary elementary gate.
    block: str = ""
    #: The gate object that will carry this op out, so execution needs no name lookup.
    #: ``None`` for measurement, which is not a gate. Excluded from equality and repr
    #: because it is machinery, not part of the record you read.
    gate: Any = field(default=None, compare=False, repr=False)

    @property
    def all_qubit_ids(self) -> tuple[int, ...]:
        """Every qubit this op touches, controls included."""
        return self.controls + self.qubit_ids


@dataclass(frozen=True)
class Checkpoint:
    """A position on a circuit's tape, plus the allocation fingerprint valid there.

    Made by :meth:`Circuit.checkpoint` and consumed by :meth:`Circuit.rewind`. It is
    deliberately *not* a copy of the state: nothing about the state is saved here, and
    nothing needs to be. Undoing a stretch of quantum program means running its gates
    backwards, and the tape already says which gates those were.

    (That is the one place where this library's tape is *simpler* than an autograd
    tape. PyTorch has to keep the intermediate tensors around to compute gradients on
    the way back; we keep nothing, because every gate is invertible. "No saved
    activations" is the software shadow of "unitaries destroy no information".)

    The three numbers are a fingerprint of which qubits existed at that moment, so
    that :meth:`Circuit.rewind` can refuse to replay ops naming qubits that have since
    been released or renumbered.
    """

    #: The circuit this position belongs to. A mark means nothing on another circuit.
    _circuit: Circuit = field(repr=False)
    #: How many ops the history held when the mark was taken.
    _history_len: int
    #: How many qubit ids had ever been handed out — catches allocation since the mark.
    _next_id: int
    #: How many qubits were live — catches an ancilla release since the mark.
    _n_qubits: int

    def __repr__(self) -> str:
        label = f" of Circuit {self._circuit.name!r}" if self._circuit.name else ""
        return f"<Checkpoint at op {self._history_len}{label}, {self._n_qubits} qubits>"


class HookHandle:
    """The receipt for a hook registered with :meth:`Circuit.on_op`.

    Keep it if you ever want the hook to stop firing; call :meth:`remove` and it
    detaches. Removing twice is harmless.
    """

    __slots__ = ("_circuit", "_fn")

    def __init__(self, circuit: Circuit, fn: Callable[[Op, Circuit], None]) -> None:
        self._circuit = circuit
        self._fn = fn

    def remove(self) -> None:
        """Detach the hook. Safe to call more than once."""
        if self._fn in self._circuit._hooks:
            self._circuit._hooks.remove(self._fn)

    def __repr__(self) -> str:
        attached = "attached" if self._fn in self._circuit._hooks else "removed"
        return f"<HookHandle {getattr(self._fn, '__name__', 'hook')}, {attached}>"


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
        # Combinator scopes push a buffer here. While the stack is non-empty the
        # circuit is in *record mode*: gates append to the innermost buffer instead of
        # running, so the scope can transform them before they execute.
        self._record_stack: list[list[Op]] = []
        # The name of the block currently being recorded, stamped onto each op.
        self._current_block = ""
        self._block_calls: list[str] = []
        # Which qubit ids count as "the environment" — see :meth:`environment`. This
        # set changes nothing about the physics; it only tells the Inspector which
        # qubits it should stop tracking when asked for the *system's* point of view.
        self._is_env: set[int] = set()
        # Observers attached with on_op(). They are called after every executed op,
        # gates and measurements alike, and are forbidden from emitting ops themselves
        # — see _refuse_while_hooked.
        self._hooks: list[Callable[[Op, Circuit], None]] = []
        self._in_hook = False
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

    def environment(self, count: int = 1, *, name: str = "E") -> Register:
        """Allocate ``count`` qubits and mark them as *the environment*.

            env = qc.environment(2)
            dephasing_coupling(q, env[0], theta=np.pi / 3)
            qc.inspect.system_entropy()      # q, seen without the environment

        **This traces nothing out.** The qubits stay in the state tensor, they stay
        entangled with everything they touch, and the global state stays perfectly
        pure — forever. Nothing here is discarded, approximated, or made stochastic.

        All the marking does is answer a question the Inspector would otherwise have
        to ask you every time: *which qubits are the thing we are studying, and which
        are the rest of the world?* With that answer on file,
        :meth:`~qsim.inspector.Inspector.system_density_matrix` and friends can report
        the system's point of view by default.

        That is the whole of decoherence, and the API is shaped to make it unmissable:
        decoherence is not something that happens *to* a state. It is what a subsystem
        looks like when you decline to track the rest of the world. The mixedness lives
        in the choice of view, not in the qubit — which is exactly why the eraser in
        ``decoherence.py`` can undo it.

        Marking is not a permission system: any qubit can play the part of an
        environment, and :func:`~qsim.decoherence.dephasing_coupling` will happily
        decohere a qubit against an ordinary ancilla. Being an environment is a
        decision about where you point your attention, so it would be dishonest to
        make it a property the simulator enforces.
        """
        register = self.register(count, name=name)
        self._is_env.update(q._id for q in register)
        return register

    @property
    def system_qubits(self) -> Register:
        """Every live qubit *not* marked as environment, in allocation order."""
        return Register(tuple(q for q in self._qubits if q._id not in self._is_env), name="system")

    @property
    def environment_qubits(self) -> Register:
        """Every live qubit marked as environment, in allocation order."""
        return Register(tuple(q for q in self._qubits if q._id in self._is_env), name="environment")

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
        """How many times each elementary operation appears in the history.

        Counts what actually ran, so a block contributes the gates it expands into —
        this is the answer to "what does this circuit cost". For the other question,
        "what is this circuit made of", see :meth:`block_counts`.
        """
        counts: dict[str, int] = {}
        for op in self._history:
            counts[op.name] = counts.get(op.name, 0) + 1
        return counts

    def block_counts(self) -> dict[str, int]:
        """How many times each ``@qsim.gate`` block was called."""
        counts: dict[str, int] = {}
        for name in self._block_calls:
            counts[name] = counts.get(name, 0) + 1
        return counts

    # ---- the tape: checkpoint, rewind, hooks ------------------------------------

    def checkpoint(self) -> Checkpoint:
        """Mark the current position on the tape, for :meth:`rewind` to return to.

            mark = qc.checkpoint()
            H(q); CNOT(q, e)          # entangle, and watch coherence die
            qc.rewind(mark)           # ... and bring it back

        Nothing is copied. A :class:`Checkpoint` is a position and an allocation
        fingerprint, and that is enough: undoing quantum work means running its gates
        backwards, and the tape already knows which gates those were.

        Cheap enough to take in a loop — that is how notebook 04 sweeps a parameter
        without rebuilding the circuit each time.
        """
        return Checkpoint(self, len(self._history), self._next_id, self.n_qubits)

    def rewind(self, mark: Checkpoint) -> None:
        """Undo everything done since ``mark``, by running it backwards.

        Each op recorded after the mark is inverted and executed, newest first —
        undoing "put on socks, then shoes" is "take off shoes, then socks". When the
        walk finishes the state is the state at the checkpoint, to floating-point
        precision. Nothing was saved and nothing was restored; the work was simply
        run in reverse.

        **The tape stays honest.** Those inverse gates physically ran, so they are
        appended to the history like any other op — the history is never rewritten or
        truncated. The *state* goes back; the *record* shows how it got back, the way
        an editor's undo appears in the edit log rather than erasing your keystrokes
        from it. So ``gate_counts()`` after a rewind includes the undoing, and it
        should: those gates would cost time on real hardware, and a circuit diagram
        that hid them would be a diagram of a program nobody ran.

        One consequence to know before you write a loop: rewinding twice to the *same*
        mark also undoes the first rewind, since the first rewind's gates are on the
        tape too. The state still comes out right — undoing an undo is a redo, and then
        the redo is undone in turn — but the op count doubles each time round. When
        sweeping a parameter, take a fresh mark after each rewind (the state there is
        the state at the old mark), and the tape grows one entry per pass.

        Raises :class:`~qsim.errors.QsimError` if the stretch being undone contains a
        measurement (the one operation with no inverse), if qubits were allocated or
        released since the mark, or if a combinator scope is open. Every check runs
        *before* anything touches the state, so a refused rewind changes nothing.
        """
        self._refuse_while_hooked()
        if self._record_stack:
            raise QsimError(
                "cannot rewind while a combinator scope is open. Inside "
                "`with qc.control(...)`, `with qc.adjoint():` or a @qsim.gate block, "
                "the body's ops have been recorded but not yet run: the tape and the "
                "state deliberately disagree until the scope closes, so there is no "
                "consistent position to rewind to. Close the scope — its ops execute "
                "on the way out — and rewind after that."
            )
        if mark._circuit is not self:
            raise QsimError(
                "this checkpoint belongs to a different circuit. A mark is a position "
                "on one circuit's tape together with the qubits that existed there; on "
                "another circuit, with its own history and its own qubits, it names "
                "nothing."
            )
        if mark._history_len > len(self._history):
            raise QsimError(
                f"this checkpoint points past the end of the tape: it marks op "
                f"{mark._history_len}, and the history is {len(self._history)} ops "
                "long. Marks are positions in a record that only ever grows — even a "
                "rewind appends to it — so a position beyond the end is one that never "
                "happened on this circuit."
            )
        if self._next_id != mark._next_id or self.n_qubits != mark._n_qubits:
            raise QsimError(
                f"cannot rewind across a change in which qubits exist. The circuit had "
                f"{mark._n_qubits} qubit(s) when the mark was taken and has "
                f"{self.n_qubits} now; {mark._next_id} had ever been allocated then, "
                f"{self._next_id} now.\n\n"
                "Ops on the tape name their qubits by id, and replaying their inverses "
                "only means something if those qubits are still there and still hold "
                "what they held. Allocating a qubit adds an axis to the state tensor "
                "and releasing an ancilla removes one, renumbering the axes underneath "
                "the recorded ops. Take the mark inside the allocation instead — or "
                "rewind before the ancilla scope closes, which is the usual intent "
                "anyway, since that is what makes the ancillas come back clean."
            )

        suffix = self._history[mark._history_len :]
        for position, op in enumerate(suffix, start=mark._history_len):
            if op.gate is None:
                raise QsimError(
                    f"cannot rewind across the measurement at op {position} of the "
                    "history. Every gate carries a rule for its own inverse, which is "
                    "what lets the tape be walked backwards at all — every gate except "
                    "one. Measurement has no inverse rule: it picked one branch of a "
                    "superposition at random and discarded the others, and nothing "
                    "brings back what was discarded. The measurement severed the tape, "
                    "exactly the way a non-differentiable operation severs an autograd "
                    "graph — everything before the cut is intact, but you cannot get "
                    "back through it.\n\n"
                    "The alternative is not to cut it. If you want the world to hold a "
                    "record of the qubit and still be able to undo the recording, "
                    "entangle instead of measuring: couple the qubit to another qubit "
                    "and leave that record coherent. The coupling is then a unitary "
                    "like any other and runs backwards perfectly — that is the quantum "
                    "eraser (06-decoherence.ipynb §9, test TD3). Otherwise, rewind only "
                    "to a mark taken after the measurement."
                )

        # Walk the suffix newest-first, executing each op's inverse. No new kernel
        # code: an inverse op is an ordinary op, and it goes through _execute like
        # every other, which is exactly why it lands on the tape.
        for op in reversed(suffix):
            self._execute(op.gate.adjoint_op(op))

    def on_op(self, fn: Callable[[Op, Circuit], None]) -> HookHandle:
        """Call ``fn(op, circuit)`` after every operation this circuit runs.

            entropies: list[float] = []
            handle = qc.on_op(lambda op, c: entropies.append(
                c.inspect.entanglement_entropy([q])))
            ...
            handle.remove()

        The hook sees gates *and* measurements, in the order they happened, with the
        state already updated — so it can ask the Inspector anything, and gets the
        answer as of just after that op. That is all ``viz.entropy_trace`` is: a hook
        that records one number per gate.

        Hooks observe and must not emit. Applying a gate or measuring from inside a
        hook raises, because those ops would appear on the tape with no line of the
        program accounting for them. To transform a program rather than watch one, use
        the combinators.

        There are deliberately no priorities and no filters: hooks fire in the order
        they were attached, and a hook that only cares about measurements checks
        ``op.name`` itself.
        """
        self._hooks.append(fn)
        return HookHandle(self, fn)

    # ---- combinator scopes ------------------------------------------------------

    def control(self, *controls: Qubit) -> Any:
        """Run a block of gates only where every control qubit is |1⟩.

            with qc.control(c):
                bell(a, b)

        Every gate inside the body is recorded rather than run, then lifted to its
        controlled form and executed on exit. Because the control may itself be in
        superposition, the result is a superposition of *the block having run and not
        having run* — not a coin flip deciding between them.

        At simulator level there is no decomposition: "conditioned on all controls
        being |1⟩" is just more sliced axes, however many controls there are. Real
        hardware is not so lucky — a multiply-controlled gate must be built out of one-
        and two-qubit gates, sometimes hundreds of them. The slice here is the
        mathematical meaning those decompositions work to implement.
        """
        from qsim.combinators import ControlScope

        return ControlScope(self, controls)

    def adjoint(self) -> Any:
        """Run a block of gates backwards.

            with qc.adjoint():
                H(a)
                CNOT(a, b)

        The body is recorded, then replayed in reverse with every gate replaced by its
        inverse. Every gate has one, because every gate is unitary — see
        ``Gate.adjoint``. Measurement does not, and attempting to measure inside this
        scope raises.
        """
        from qsim.combinators import AdjointScope

        return AdjointScope(self)

    def ancilla(self, count: int = 1) -> Any:
        """Borrow ``count`` scratch qubits, which must be given back clean.

            with qc.ancilla(2) as scratch:
                ...                      # use them
                ...                      # then uncompute them back to |00⟩

        On exit the scope **verifies numerically** that the scratch qubits are back in
        |0…0⟩ and unentangled, and raises :class:`~qsim.errors.DirtyAncillaError` if
        not. This check is the point of the whole construct, and it is a simulator
        superpower: real hardware cannot look. It is also the mechanism by which this
        library teaches uncomputation, so it is a hard requirement and not a debug
        option you can switch off.

        Note what is *not* offered: releasing a qubit by dropping its handle. Discarding
        an entangled qubit is a partial trace — it silently turns a pure state into a
        mixed one. There is no garbage collection for quantum memory; release requires
        uncomputation. That is physics, not a missing feature.
        """
        from qsim.combinators import AncillaScope

        return AncillaScope(self, count)

    def _deallocate(self, qubits: Sequence[Qubit]) -> None:
        """Remove verified-clean qubits from the state and renumber the surviving axes.

        This is the axis-lifecycle problem of design doc §2.4 in action, and the reason
        handles never store axis numbers.
        """
        axes = sorted(self._axis_of[q._id] for q in qubits)
        # Take the slice where every released qubit reads 0. Because the caller has
        # already verified they are exactly |0…0⟩, all the amplitude lives in this
        # slice — so dropping the axes loses nothing and the norm is preserved.
        selector: list[Any] = [slice(None)] * self._psi.ndim
        for axis in axes:
            selector[axis] = 0
        self._psi = np.ascontiguousarray(self._psi[tuple(selector)])

        for q in qubits:
            del self._axis_of[q._id]
            self._qubits.remove(q)
            self._is_env.discard(q._id)
            q._live = False

        # Renumber: the survivors keep their relative order and close the gaps.
        survivors = sorted(self._axis_of.items(), key=lambda item: item[1])
        self._axis_of = {qubit_id: new_axis for new_axis, (qubit_id, _) in enumerate(survivors)}

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

    def _validate(self, qubits: Sequence[Qubit]) -> None:
        """Check handles are usable and distinct, without resolving them to axes."""
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
        for q in qubits:
            self._axis(q)  # raises for a foreign or released handle

    def _record(self, op: Op) -> None:
        """The one place the history grows — and therefore the one place hooks fire.

        Gates reach it through :meth:`_execute` and measurements reach it directly from
        ``measure.py``, so an observer attached here sees the whole program, in the
        order it really happened, with the state already updated.
        """
        self._history.append(op)
        if self._hooks:
            # Iterate over a snapshot: a hook is allowed to remove itself (or another
            # hook) while it runs, and mutating the list we are walking would silently
            # skip the next hook.
            previously_in_hook = self._in_hook
            self._in_hook = True
            try:
                for fn in list(self._hooks):
                    fn(op, self)
            finally:
                self._in_hook = previously_in_hook

    def _refuse_while_hooked(self) -> None:
        """Raise if we are inside a hook. Called before anything that touches the tape."""
        if self._in_hook:
            raise QsimError(
                "a hook tried to apply an operation to the circuit. Hooks watch the "
                "tape; they do not write to it. An op emitted from inside a hook would "
                "land in the history with no line of your program accounting for it, "
                "and it would immediately fire every hook again — including the one "
                "that emitted it.\n\n"
                "If you want to *transform* a program rather than watch it, the "
                "combinators are the tools for that: qsim.within, qc.control, "
                "qc.adjoint and @qsim.gate blocks all put their ops on the tape at the "
                "point where you asked for them."
            )

    # ---- the emit / execute funnel ---------------------------------------------

    def _emit(self, op: Op) -> None:
        """Every gate arrives here. Run it now, or record it for a scope to transform."""
        self._refuse_while_hooked()
        if self._record_stack:
            self._record_stack[-1].append(op)
        else:
            self._execute(op)

    def _execute(self, op: Op) -> None:
        """Apply one recorded operation to the state, and add it to the history.

        Axes are resolved *here*, not when the gate was called. That is the whole
        reason ops carry qubit ids: a combinator scope may hold an op for a while, and
        an ancilla scope may renumber every axis in between.
        """
        from qsim import state

        gate = op.gate
        data = gate._data_for(op.params)
        # Match the circuit's precision. Without this, a complex128 gate matrix would
        # silently promote a complex64 state back to double and quietly undo the
        # single-precision experiment of design doc §9 (T17).
        data = data.astype(self._psi.dtype, copy=False)

        targets = [self._axis_of[i] for i in op.qubit_ids]
        controls = [self._axis_of[i] for i in op.controls]

        if controls:
            if gate._kind == "diag":
                self._psi = state.apply_controlled_diag(self._psi, data, controls, targets[0])
            else:
                self._psi = state.apply_controlled(self._psi, data, controls, targets)
        elif gate._kind == "diag":
            self._psi = state.apply_diag(self._psi, data, targets[0])
        elif gate._kind == "unitary1":
            self._psi = state.apply_1q(self._psi, data, targets[0])
        else:
            self._psi = state.apply_2q(self._psi, data, targets[0], targets[1])

        # One funnel: gates do not append to the history themselves. Everything that
        # runs — this gate, and every measurement in measure.py — goes on the tape
        # through _record, which is where hooks live.
        self._record(op)

    def _push_record(self) -> None:
        self._record_stack.append([])

    def _pop_record(self) -> list[Op]:
        return self._record_stack.pop()

    def _repr_html_(self) -> str:
        # Jupyter calls this when a Circuit is the last expression in a cell.
        from qsim.viz import circuit_html

        return circuit_html(self)

    def __repr__(self) -> str:
        label = f" {self._name!r}" if self._name else ""
        return f"<Circuit{label} with {self.n_qubits} qubits, {len(self._history)} ops>"

