"""Combinators: taking a block of gates and doing something to it as a whole.

**The physical fact this module makes concrete:** a quantum program is not just a list
of instructions to run, it is a *unitary* — an object you can invert, condition on
another qubit, and reason about as one thing. Three constructs follow from that, and
between them they are what every later algorithm is written with.

- ``with qc.adjoint():`` runs a block backwards. Possible because unitaries are always
  invertible, which is the same statement as "no information is lost".
- ``with qc.control(c):`` runs a block only where ``c`` is |1⟩ — and if ``c`` is in
  superposition, the result is a superposition of the block having run and not.
- ``with qc.ancilla(n) as scratch:`` borrows scratch qubits and *verifies* they come
  back clean, which is where uncomputation stops being advice and becomes enforced.

How they work: record mode
---------------------------
Execution in qsim is eager — a gate applied is a gate run — with exactly one
exception, and physics forces it. You cannot lift a block to its controlled form
without knowing the whole block first. So inside a scope the circuit records ops into
a buffer instead of executing them; on exit the buffer is transformed and *then* run.

Scopes nest, and the transformations compose in the order you would expect: an inner
scope's output is emitted into the outer scope's buffer rather than to the state.
"""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from qsim.circuit import Op, Qubit, Register
from qsim.errors import NoCloningError, QsimError

if TYPE_CHECKING:
    from qsim.circuit import Circuit


class _Scope:
    """Shared plumbing: push a record buffer on entry, transform and emit on exit."""

    def __init__(self, circuit: Circuit) -> None:
        self._circuit = circuit

    def __enter__(self) -> Any:
        self._circuit._push_record()
        return None

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        recorded = self._circuit._pop_record()
        if exc_type is not None:
            # The body blew up. Drop what was recorded and let the real error through,
            # rather than executing half a transformed block on the way out.
            return False
        for op in self._transform(recorded):
            self._circuit._emit(op)
        return False

    def _transform(self, ops: list[Op]) -> list[Op]:
        raise NotImplementedError  # pragma: no cover - every scope overrides it


class ControlScope(_Scope):
    """``with qc.control(c):`` — see :meth:`qsim.Circuit.control`."""

    def __init__(self, circuit: Circuit, controls: Sequence[Qubit]) -> None:
        super().__init__(circuit)
        circuit._validate(controls)
        self._controls = tuple(controls)

    def _transform(self, ops: list[Op]) -> list[Op]:
        control_ids = tuple(q._id for q in self._controls)
        for op in ops:
            for control in self._controls:
                if control._id in op.all_qubit_ids:
                    raise NoCloningError(
                        f"{control._name} controls this block and is also acted on "
                        "inside it. A qubit cannot control an operation on itself: "
                        "deciding what to do to it would mean reading its value, and "
                        "the result would be a copy of that value. Use a separate "
                        "qubit as the control."
                    )
        # Lifting a gate to its controlled form adds control ids; the gate itself is
        # untouched, because control is implemented by slicing the state rather than by
        # building a larger matrix.
        return [
            Op(
                name=op.name,
                qubit_ids=op.qubit_ids,
                params=op.params,
                controls=control_ids + op.controls,
                block=op.block,
                gate=op.gate,
            )
            for op in ops
        ]


class AdjointScope(_Scope):
    """``with qc.adjoint():`` — see :meth:`qsim.Circuit.adjoint`."""

    def _transform(self, ops: list[Op]) -> list[Op]:
        # Reverse the order and invert each step. Both halves are needed: undoing
        # "put on socks, then shoes" is "take off shoes, then socks".
        return [op.gate.adjoint_op(op) for op in reversed(ops)]


class AncillaScope:
    """``with qc.ancilla(n) as scratch:`` — see :meth:`qsim.Circuit.ancilla`."""

    def __init__(self, circuit: Circuit, count: int) -> None:
        if count < 1:
            raise ValueError(f"cannot borrow {count} ancilla qubits; ask for at least 1")
        self._circuit = circuit
        self._count = count
        self._register: Register | None = None

    def __enter__(self) -> Register:
        if self._circuit._record_stack:
            raise QsimError(
                "cannot open an ancilla scope inside a block, control or adjoint scope. "
                "Those scopes record their gates and run them later, so at this point "
                "the body has not happened yet: there would be nothing to verify on the "
                "way out, and the scratch qubits would be released while operations "
                "referring to them were still waiting to run.\n\n"
                "Allocate the scratch outside instead and pass it in, which also makes a "
                "block's qubit requirements visible in its signature:\n\n"
                "    with qc.ancilla(3) as scratch:\n"
                "        my_block(x, y, scratch)"
            )
        self._register = self._circuit.register(self._count, name="anc")
        return self._register

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        register = self._register
        assert register is not None  # set by __enter__; a scope cannot exit unentered
        if exc_type is not None:
            # Something already went wrong inside the body. Retire the handles so they
            # cannot be used again, but leave the axes alone: the state may be
            # entangled, and slicing it away would corrupt what is left while hiding
            # the error the user actually needs to see.
            for q in register:
                q._live = False
            return False

        # One number checks both conditions. If the probability of finding every
        # ancilla at 0 is 1, then every amplitude with any ancilla bit set is zero, so
        # the state factorizes as |0...0>_anc ⊗ |rest> — the ancillas are simultaneously
        # in |0> *and* unentangled. Anything less means something is still recorded over
        # there.
        #
        # If this raises, the handles are deliberately left alive and the axes in place,
        # so that a caller who catches DirtyAncillaError can go and look at the mess —
        # which is the whole point of the exercise in notebook 04.
        self._circuit.inspect.assert_zero(register)
        self._circuit._deallocate(list(register))
        return False


class Block:
    """A named, reusable block of gates — what ``@qsim.gate`` produces.

        @qsim.gate
        def bell(a, b):
            H(a)
            CNOT(a, b)

        bell(a, b)                 # runs now
        bell.adjoint()(a, b)       # runs backwards
        bell.controlled(c)(a, b)   # runs only where c is |1⟩

    ``adjoint`` and ``controlled`` are the same two methods gates have, so one
    vocabulary covers a single gate and a hundred-gate subroutine alike.

    A block's body may contain only gates and other blocks. There is no way for it to
    reach the state tensor directly — it receives qubit handles and nothing else —
    so this is enforced by construction rather than by a rule you have to remember.
    """

    def __init__(self, fn: Callable[..., None]) -> None:
        self._fn = fn
        self.name = fn.__name__
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        circuit, ops = self._record(args, kwargs)
        for op in ops:
            circuit._emit(op)

    def adjoint(self) -> Callable[..., None]:
        """A callable that runs this block backwards."""

        def run(*args: Any, **kwargs: Any) -> None:
            circuit, ops = self._record(args, kwargs)
            for op in reversed(ops):
                circuit._emit(op.gate.adjoint_op(op))

        return run

    def controlled(self, *controls: Qubit) -> Callable[..., None]:
        """A callable that runs this block only where every control qubit is |1⟩."""

        def run(*args: Any, **kwargs: Any) -> None:
            circuit = _circuit_of(args)
            with circuit.control(*controls):
                self(*args, **kwargs)

        return run

    def _record(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Circuit, list[Op]]:
        """Run the body in a private buffer, capturing its gates without executing them.

        Classical arguments are consumed *here*, at record time, which is what lets the
        same block be inverted or controlled later: by the time a transformation runs,
        every angle has already become a number in a recorded op.
        """
        circuit = _circuit_of(args)
        outer_block = circuit._current_block
        circuit._current_block = self.name
        circuit._push_record()
        try:
            self._fn(*args, **kwargs)
        finally:
            ops = circuit._pop_record()
            circuit._current_block = outer_block
        circuit._block_calls.append(self.name)
        return circuit, ops

    def __repr__(self) -> str:
        return f"<Block {self.name}>"


def gate(fn: Callable[..., None]) -> Block:
    """Decorator turning a function of qubits into a reusable, invertible block.

    See :class:`Block`.
    """
    return Block(fn)


def _circuit_of(args: Sequence[Any]) -> Circuit:
    """Find the circuit a block is operating on, from the handles it was given."""
    for arg in args:
        if isinstance(arg, Qubit):
            return arg._circuit
        if isinstance(arg, Register) and len(arg):
            return arg[0]._circuit
    raise QsimError(
        "a block must be given at least one qubit or non-empty register, so that it "
        "knows which circuit it is acting on."
    )
