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
- ``with qsim.within(V, q):`` does V, runs the body, then undoes V — the sandwich
  ``V U V†`` that quantum programs are built out of.

How they work: record mode
---------------------------
Execution in qsim is eager — a gate applied is a gate run — with exactly one
exception, and physics forces it. You cannot lift a block to its controlled form
without knowing the whole block first. So inside a scope the circuit records ops into
a buffer instead of executing them; on exit the buffer is transformed and *then* run.

Scopes nest, and the transformations compose in the order you would expect: an inner
scope's output is emitted into the outer scope's buffer rather than to the state.

``within`` is the exception that proves the rule: it records only *V*, never the body.
"Undo V later" needs V remembered and nothing else, so the body can stay eager and the
state stays watchable between its statements. "Run conditioned on c" cannot — it is a
counterfactual, and the ops have to execute *differently*, which means seeing them
first.
"""

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from qsim.circuit import Op, Qubit, Register
from qsim.errors import NoCloningError, QsimError

if TYPE_CHECKING:
    from qsim.circuit import Circuit

#: How a derived block rewrites the ops its body recorded, before they are emitted.
#: It gets the circuit too, so that it can validate handles at call time.
type _OpTransform = Callable[["Circuit", list[Op]], list[Op]]


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


def _lift_to_controlled(ops: list[Op], controls: Sequence[Qubit]) -> list[Op]:
    """Add ``controls`` to every op — the one implementation of "control this block".

    Both spellings of control go through here (``with qc.control(c):`` and
    ``block.controlled(c)``), so the self-control rule and the sentence explaining it
    exist once.
    """
    control_ids = tuple(q._id for q in controls)
    for op in ops:
        for control in controls:
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


def _invert(ops: list[Op]) -> list[Op]:
    """Reverse the order and invert each step — the one implementation of "undo this".

    Both halves are needed: undoing "put on socks, then shoes" is "take off shoes,
    then socks".
    """
    return [op.gate.adjoint_op(op) for op in reversed(ops)]


class ControlScope(_Scope):
    """``with qc.control(c):`` — see :meth:`qsim.Circuit.control`."""

    def __init__(self, circuit: Circuit, controls: Sequence[Qubit]) -> None:
        super().__init__(circuit)
        circuit._validate(controls)
        self._controls = tuple(controls)

    def _transform(self, ops: list[Op]) -> list[Op]:
        return _lift_to_controlled(ops, self._controls)


class AdjointScope(_Scope):
    """``with qc.adjoint():`` — see :meth:`qsim.Circuit.adjoint`."""

    def _transform(self, ops: list[Op]) -> list[Op]:
        return _invert(ops)


class WithinScope:
    """``with qsim.within(V, q):`` — do V, run the body, undo V.

        with qsim.within(H, q):                    # V = H(q), applied right now
            dephasing_coupling(q, e, theta=t)      # the body, run eagerly
                                                   # H(q) again on the way out

    The sandwich V·U·V† is the most common composite in quantum programming: change
    to a basis where the operation you want is easy, do it, change back. Dephasing a
    qubit in the |+⟩/|−⟩ basis is ordinary dephasing wrapped in H. Grover's oracle is
    a controlled-Z wrapped in X gates. The Fourier-space adder is phase rotations
    wrapped in a QFT. Writing the wrapper twice by hand works and is how it was done
    before this scope existed; the risk is that the two halves drift apart, and the
    bug that produces is a silent wrong answer rather than an error.

    What is recorded, and what is not
    ---------------------------------
    Only **V** is recorded. The scope runs it under a private buffer purely to learn
    which ops it consists of, emits them, and then gets out of the way: the body runs
    eagerly, gate by gate, and you can inspect the state between any two of its lines.
    That is the difference from ``qc.control``, which has no choice but to record its
    body — "run this only where c is |1⟩" is a counterfactual, so the ops must execute
    differently, and you cannot rewrite an op you have not seen yet. "Undo V later"
    needs V remembered and nothing more.

    (The same split exists in PyTorch, if that is a familiar landmark: ``backward()``
    is a tape operation, replaying a record of what already ran, while ``vmap`` is a
    function transform that has to intercept operations before they happen.)

    Two identities worth knowing, both of which this scope gets for free
    -------------------------------------------------------------------
    **Inverting a sandwich inverts only the filling.** (V U V†)† = V U† V†, because
    reversing the whole sequence puts V back at the front and V† back at the end.
    So a block built with ``within`` can be run backwards and the basis change still
    happens first — which is what you want, and not something you have to arrange.

    **Controlling a sandwich controls every layer, and that is correct**, because
    control distributes over a product: C(V U V†) = (CV)(CU)(CV†). If V happens not
    to touch the control qubit, the shorter form V (CU) V† does the same thing —
    conditioning the basis change is unnecessary work, since a basis change and its
    undo cancel in the branch where the control is |0⟩. qsim implements the uniform
    version; TT3 checks that both agree.

    Rules
    -----
    - ``V`` is any op-emitting callable — a gate, a ``@qsim.gate`` block, or a plain
      function — and its qubit arguments are how the scope finds the circuit.
    - **A named V is counted twice, once per half.** If V has a ``.name`` — every gate
      and every ``@qsim.gate`` block does — then ``block_counts()`` reports both
      ``name`` and ``name†`` after the scope closes, matching the ``bell†`` naming that
      ``Block.adjoint()`` already uses. So a conjugation shows up in the tally as the
      symmetric thing it is, instead of the basis change appearing once and its undo
      not at all. A plain function or lambda has no ``.name`` and stays unstamped: qsim
      will not invent a name for something the language did not name, and ``<lambda>``
      in a gate tally helps nobody. That is the boundary — if you want a conjugation
      counted, wrap V in a ``def`` and decorate it.
    - ``V`` may not measure. The whole construct rests on being able to undo V on the
      way out, and measurement is the one operation that cannot be undone.
    - If the body raises, **V† is not applied**: never run half a construct on the way
      out of an error. The same rule as the other scopes.
    - To make a conjugation reusable, wrap it in the abstraction mechanism the language
      already has, a ``def``::

          @qsim.gate
          def x_dephasing(q, e, theta):
              with qsim.within(H, q):
                  dephasing_coupling(q, e, theta=theta)
    """

    def __init__(
        self, v: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        self._v = v
        self._args = args
        self._kwargs = kwargs
        self._circuit: Circuit | None = None
        self._v_ops: list[Op] = []
        # Gates and Blocks both carry ``.name``; plain functions carry ``__name__`` and
        # not ``.name``, so this attribute is exactly the "did a human name this?" test
        # the stamping rule needs. See the class docstring.
        self._name: str = getattr(v, "name", "")

    def __enter__(self) -> None:
        circuit = _circuit_of(self._args, message=_WITHIN_NEEDS_A_QUBIT)
        self._circuit = circuit
        # Run V under a private buffer. This is not deferral — V is about to be
        # emitted, in this same method — it is how the scope learns what V *was*, so
        # that it can build the inverse later.
        circuit._push_record()
        try:
            self._v(*self._args, **self._kwargs)
        finally:
            v_ops = circuit._pop_record()

        for op in v_ops:
            if op.gate is None:
                raise QsimError(
                    f"the basis change of a `within` scope contains a {op.name}, which "
                    "has no inverse. Everything V does has to be undone on the way out "
                    "of the scope — that is the whole construct — and measurement is "
                    "the one operation that cannot be undone: it discards the branches "
                    "of the superposition it did not report. Nothing irreversible can "
                    "be part of a conjugation's wrapper. Measure in the body, or after "
                    "the scope closes."
                )
        self._v_ops = v_ops

        # A Block counts its own call while running, so counting it again here would
        # report `bell` twice for one conjugation. A bare gate counts nothing on its
        # own, so the forward half of `within(H, q)` is tallied here instead — leaving
        # both spellings symmetric with the `name†` added on the way out.
        if self._name and not isinstance(self._v, Block):
            circuit._block_calls.append(self._name)

        # Emit V now. If an enclosing scope is recording, these land in *its* buffer,
        # which is exactly right: a surrounding control or adjoint then sees V, body,
        # V† as three ordinary stretches of ops and transforms all of them uniformly.
        for op in v_ops:
            circuit._emit(op)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        circuit = self._circuit
        assert circuit is not None  # set by __enter__; a scope cannot exit unentered
        if exc_type is not None:
            # The body blew up. Leave V standing rather than undoing it: the state is
            # mid-construct and the user needs to see it that way to debug. Same rule
            # as the recording scopes, which drop their buffers on an exception.
            return False

        if self._name:
            circuit._block_calls.append(f"{self._name}†")
        # Stamp the undo ops only when V was a Block, because only then do the forward
        # ops carry V's own name in `op.block` and the two halves can match. A bare
        # gate's ops are stamped with whichever block is being recorded *around* the
        # scope, and overwriting that on one half alone would misreport where those
        # gates came from.
        undo_block = f"{self._name}†" if isinstance(self._v, Block) else ""
        for op in reversed(self._v_ops):
            undo = op.gate.adjoint_op(op)
            if undo_block:
                undo = replace(undo, block=undo_block)
            circuit._emit(undo)
        return False


def within(v: Callable[..., Any], *args: Any, **kwargs: Any) -> WithinScope:
    """Do ``v(*args, **kwargs)``, run the body, then undo it. See :class:`WithinScope`.

        with qsim.within(H, q):
            Z(q)                 # H Z H — a bit flip, spelled as a phase flip in the
                                 # basis where |+⟩ and |−⟩ are the "computational" states
    """
    return WithinScope(v, args, kwargs)


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

    **The algebra is closed**: both return another ``Block``, never a bare function.
    So they chain — ``bell.adjoint().controlled(c)`` — and the result is a first-class
    block like any other: it has a name (``"C-bell†"``), the ops it emits are stamped
    with that name, and ``block_counts()`` reports it. An operation on a block is a
    block, exactly as an operation on a gate is a gate (``T.adjoint().controlled()``).
    Nothing derived is second-class.

    Unlike gates, a block has only one name, not a short symbol plus a spelled-out
    form. It never needed two: ``H`` has to be called ``Hadamard`` somewhere because
    ``H`` is a symbol, whereas a block is already named by a ``def``.

    A block's body may contain only gates and other blocks. There is no way for it to
    reach the state tensor directly — it receives qubit handles and nothing else —
    so this is enforced by construction rather than by a rule you have to remember.
    """

    def __init__(
        self,
        fn: Callable[..., None],
        *,
        name: str = "",
        transform: _OpTransform | None = None,
    ) -> None:
        self._fn = fn
        self.name = name or fn.__name__
        #: How this block's recorded ops are rewritten before they are emitted.
        #: ``None`` for a block written with ``@qsim.gate``; a derived block sets it.
        self._transform = transform
        self.__doc__ = fn.__doc__
        self.__name__ = self.name

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        circuit, ops = self._record(args, kwargs)
        if self._transform is not None:
            ops = self._transform(circuit, ops)
        for op in ops:
            circuit._emit(op)

    def adjoint(self) -> Block:
        """This block run backwards — a ``Block`` named ``"bell†"``.

        The recorded body is reversed and every op replaced by its inverse. Note that
        ``bell.adjoint().adjoint()`` acts exactly as ``bell`` does, because two
        reversals compose to none; it is named ``"bell††"`` rather than being folded
        back into ``bell``, since the tape's job is to say what you asked for.
        """
        return Block(
            self._fn,
            name=f"{self.name}†",
            transform=_then(self._transform, lambda circuit, ops: _invert(ops)),
        )

    def controlled(self, *controls: Qubit) -> Block:
        """This block, run only where every control qubit is |1⟩ — a ``Block``.

        Named with one ``C`` per control: ``bell.controlled(c)`` is ``"C-bell"`` and
        ``bell.controlled(c1, c2)`` is ``"CC-bell"``, matching how gates spell theirs
        (``CNOT``, ``CZ``) with a hyphen added because block names are words.

        The controls are checked at *call* time, not here — that is when the circuit
        they belong to is known, from the qubit arguments of the call.
        """
        if not controls:
            raise QsimError(
                "controlled() needs at least one control qubit: "
                "`bell.controlled(c)`. With nothing to condition on there is no "
                "counterfactual to build — the block would simply be itself."
            )

        def control_them(circuit: Circuit, ops: list[Op]) -> list[Op]:
            # Rejects a released or foreign control handle, and the same qubit given
            # twice as a control, at the call site rather than deep in the transform.
            circuit._validate(controls)
            return _lift_to_controlled(ops, controls)

        return Block(
            self._fn,
            name=f"{'C' * len(controls)}-{self.name}",
            transform=_then(self._transform, control_them),
        )

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


def _then(first: _OpTransform | None, second: _OpTransform) -> _OpTransform:
    """Compose two op-transformations: ``first``, then ``second``.

    This is what makes derived blocks chain. ``bell.adjoint().controlled(c)`` reverses
    the recorded body and *then* adds the control to every op — the same order the
    method calls were written in, and the same order the two operations would be
    applied to the unitary itself.
    """
    if first is None:
        return second

    def composed(circuit: Circuit, ops: list[Op]) -> list[Op]:
        return second(circuit, first(circuit, ops))

    return composed


_BLOCK_NEEDS_A_QUBIT = (
    "a block must be given at least one qubit or non-empty register, so that it "
    "knows which circuit it is acting on."
)

_WITHIN_NEEDS_A_QUBIT = (
    "a `within` scope must be given at least one qubit or non-empty register among "
    "V's arguments — write `with qsim.within(H, q):`, not `with qsim.within(H):`. "
    "The scope has to know whose tape to capture V on before it can replay V backwards "
    "on the way out, and qubit handles are the only thing that says which circuit is "
    "meant."
)


def _circuit_of(args: Sequence[Any], *, message: str = _BLOCK_NEEDS_A_QUBIT) -> Circuit:
    """Find the circuit an op-emitting callable is operating on, from its arguments."""
    for arg in args:
        if isinstance(arg, Qubit):
            return arg._circuit
        if isinstance(arg, Register) and len(arg):
            return arg[0]._circuit
    raise QsimError(message)
