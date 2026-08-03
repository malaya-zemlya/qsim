"""The tape and the transform layer: ``within``, derived blocks, checkpoints, hooks.

Read this file as a usage guide. Four ideas live here, and they are all the same idea
seen from different sides — the history of a circuit is a *record of a program*, and a
record can be transformed:

- ``qsim.within(V, q)`` records V so that it can be undone on the way out of a scope;
- ``block.adjoint()`` / ``block.controlled(c)`` transform a recorded body into a new
  block, and the result is a block like any other;
- ``qc.checkpoint()`` / ``qc.rewind(mark)`` walk the record backwards;
- ``qc.on_op(fn)`` watches it grow.

The acceptance tests for the same phase are in ``test_acceptance_tt1_tt8.py``.
"""

import numpy as np
import pytest

import qsim
from qsim import Circuit
from qsim.circuit import Checkpoint, Op, Qubit, Register
from qsim.errors import DeadQubitError, NoCloningError, QsimError
from qsim.gates import CNOT, CZ, H, Rz, S, T, X, Z

# ---- within: do, act, undo ----------------------------------------------------


def test_within_applies_v_immediately_and_its_inverse_on_exit(qc: Circuit) -> None:
    """The two halves of the sandwich appear in the history around the body."""
    q = qc.alloc("q")

    with qsim.within(S, q):
        Z(q)

    assert [op.name for op in qc.history] == ["S", "Z", "S†"]


def test_the_body_of_a_within_scope_runs_eagerly(qc: Circuit) -> None:
    """Unlike control and adjoint, ``within`` does not record its body: each statement
    runs as it is reached, so the state can be inspected in the middle of the scope."""
    q = qc.alloc("q")

    with qsim.within(H, q):
        assert qc.inspect.probabilities() == pytest.approx([0.5, 0.5])
        Z(q)
        # The Z has already happened — H|0> = |+> became |->.
        assert qc.inspect.amplitude("1") == pytest.approx(-1 / np.sqrt(2))

    assert qc.inspect.probabilities() == pytest.approx([0.0, 1.0])


def test_within_accepts_a_register_and_finds_the_circuit_through_it(qc: Circuit) -> None:
    reg = qc.register(2, name="r")

    def spread(register: Register) -> None:
        for q in register:
            H(q)

    with qsim.within(spread, reg):
        Z(reg[0])

    assert [op.name for op in qc.history] == ["H", "H", "Z", "H", "H"]


def test_within_takes_keyword_arguments_for_v(qc: Circuit) -> None:
    q = qc.alloc("q")

    with qsim.within(Rz, q, theta=0.4):
        X(q)

    assert [(op.name, op.params) for op in qc.history] == [
        ("Rz", (0.4,)),
        ("X", ()),
        ("Rz", (-0.4,)),
    ]


def test_within_needs_a_qubit_to_know_which_circuit_it_is_on() -> None:
    def nothing() -> None:  # pragma: no cover - never reaches the body
        pass

    with pytest.raises(QsimError, match="at least one qubit"):
        with qsim.within(nothing):
            pass  # pragma: no cover - __enter__ raises before the body runs


def test_a_within_whose_basis_change_measures_is_refused(qc: Circuit) -> None:
    """Measurement cannot be part of a wrapper, because the wrapper has to be undone.

    The refusal happens twice over. Reached the ordinary way, V runs while the circuit
    is recording, and ``measure`` already refuses to run inside any recording scope.
    """
    q = qc.alloc("q")

    with pytest.raises(QsimError, match="irreversible"):
        with qsim.within(lambda target: qc.measure(target), q):
            pass  # pragma: no cover - __enter__ raises before the body runs


def test_a_within_that_captures_a_non_invertible_op_is_refused(qc: Circuit) -> None:
    """The scope also checks what it captured, for anything that reaches the buffer
    without going through a gate.

    There is no public way to emit a measurement into a record buffer — ``measure``
    refuses first — so this test hands the circuit the op directly. The guard is real
    all the same: it is what makes "V must be invertible" a property of the captured
    ops rather than of how they were produced.
    """
    q = qc.alloc("q")

    def sneaky(target: Qubit) -> None:
        target._circuit._emit(Op(name="measure", qubit_ids=(target._id,), result=0))

    with pytest.raises(QsimError) as raised:
        with qsim.within(sneaky, q):
            pass  # pragma: no cover - __enter__ raises before the body runs

    assert "no inverse" in str(raised.value)
    assert "undone" in str(raised.value)


def test_an_exception_in_the_body_leaves_the_basis_change_standing(qc: Circuit) -> None:
    """Never run half a construct on the way out of an error — the same rule the
    control and adjoint scopes follow. V stays applied, so the state at the moment of
    the failure is the state you get to look at."""
    q = qc.alloc("q")

    with pytest.raises(ValueError, match="boom"):
        with qsim.within(H, q):
            X(q)
            raise ValueError("boom")

    assert [op.name for op in qc.history] == ["H", "X"]


def test_within_nested_inside_control_lifts_every_layer(qc: Circuit) -> None:
    """A surrounding control sees V, body, V† as ordinary ops and controls all three.

    That is correct because control distributes over a product — see TT3, which checks
    it numerically against the shorter form that leaves V uncontrolled.
    """
    c, q = qc.alloc_many(2)

    with qc.control(c):
        with qsim.within(H, q):
            Z(q)

    assert [op.name for op in qc.history] == ["H", "Z", "H"]
    assert all(op.controls == (c._id,) for op in qc.history)


def test_within_inside_a_block_is_recorded_as_part_of_it(qc: Circuit) -> None:
    """Conjugation becomes reusable through the abstraction Python already has: a def."""

    @qsim.gate
    def flip_in_x(target: Qubit) -> None:
        with qsim.within(H, target):
            Z(target)

    q = qc.alloc("q")
    flip_in_x(q)

    assert [op.block for op in qc.history] == ["flip_in_x"] * 3
    # The conjugation is counted symmetrically, both halves (Phase 3's P7): V is the
    # named gate H, so the tally names H going in and H† coming out. The ops themselves
    # stay stamped with the enclosing block, which is where they really came from.
    assert qc.block_counts() == {"flip_in_x": 1, "H": 1, "H†": 1}
    # H Z H is X: the block flipped the qubit, spelled in the other basis.
    assert qc.inspect.probabilities() == pytest.approx([0.0, 1.0])


def test_nested_within_scopes_unwrap_in_the_right_order(qc: Circuit) -> None:
    """V W U W† V†: the inner wrapper is undone first, like closing brackets."""
    q = qc.alloc("q")

    with qsim.within(S, q):
        with qsim.within(H, q):
            Z(q)

    assert [op.name for op in qc.history] == ["S", "H", "Z", "H", "S†"]


def test_a_block_built_with_within_can_be_inverted(qc: Circuit) -> None:
    """(V U V†)† = V U† V†: the basis change stays where it is; the filling flips."""

    @qsim.gate
    def phase_in_x(target: Qubit) -> None:
        with qsim.within(H, target):
            T(target)

    q = qc.alloc("q")
    phase_in_x.adjoint()(q)

    assert [op.name for op in qc.history] == ["H", "T†", "H"]


# ---- the closed block algebra ---------------------------------------------------


@qsim.gate
def bell(a: Qubit, b: Qubit) -> None:
    """A Bell pair from |00>."""
    H(a)
    CNOT(a, b)


def test_a_derived_block_is_itself_a_block_and_chains(qc: Circuit) -> None:
    c, a, b = qc.alloc_many(3)
    derived = bell.adjoint().controlled(c)

    assert isinstance(derived, qsim.Block)
    assert derived.name == "C-bell†"
    assert derived.__name__ == "C-bell†"


def test_two_controls_are_spelled_with_two_cs(qc: Circuit) -> None:
    c1, c2, a, b = qc.alloc_many(4)
    twice = bell.controlled(c1, c2)

    assert twice.name == "CC-bell"

    X(c1)
    X(c2)
    twice(a, b)
    assert str(qc.inspect.ket()) == "0.707|1100⟩ + 0.707|1111⟩"


def test_a_derived_block_can_be_stored_and_called_later(qc: Circuit) -> None:
    """Classical arguments are consumed at call time, so a derived block is reusable —
    ``undo`` here works on whichever qubits it is handed, whenever it is handed them."""
    undo = bell.adjoint()
    a, b, c, d = qc.alloc_many(4)

    bell(a, b)
    bell(c, d)
    undo(a, b)
    undo(c, d)

    assert qc.inspect.probabilities()[0] == pytest.approx(1.0)


def test_the_adjoint_of_a_parametrized_block_negates_its_angles(qc: Circuit) -> None:
    @qsim.gate
    def turn(target: Qubit, *, angle: float) -> None:
        Rz(target, theta=angle)
        Rz(target, theta=2 * angle)

    q = qc.alloc("q")
    turn.adjoint()(q, angle=0.3)

    assert [op.params for op in qc.history] == [(-0.6,), (-0.3,)]


def test_controlling_a_block_with_no_control_qubit_is_refused() -> None:
    with pytest.raises(QsimError, match="at least one control"):
        bell.controlled()


def test_a_block_cannot_be_controlled_by_a_qubit_it_acts_on(qc: Circuit) -> None:
    """The no-cloning rule, in the one place it is implemented — the same message the
    ``with qc.control(...)`` scope gives, because both go through the same function."""
    a, b = qc.alloc_many(2)

    with pytest.raises(NoCloningError, match="cannot control an operation on itself"):
        bell.controlled(a)(a, b)


def test_a_released_qubit_cannot_control_a_block(qc: Circuit) -> None:
    """Controls are validated at call time, which is when the circuit is known."""
    a, b = qc.alloc_many(2)
    with qc.ancilla(1) as scratch:
        stale = scratch[0]

    with pytest.raises(DeadQubitError):
        bell.controlled(stale)(a, b)


def test_a_controlled_block_matches_the_control_scope_spelling() -> None:
    """``bell.controlled(c)(a, b)`` and ``with qc.control(c): bell(a, b)`` are the same
    program; only the name stamped on the ops differs."""
    method = Circuit(seed=1)
    c1, a1, b1 = method.alloc_many(3)
    H(c1)
    bell.controlled(c1)(a1, b1)

    scope = Circuit(seed=1)
    c2, a2, b2 = scope.alloc_many(3)
    H(c2)
    with scope.control(c2):
        bell(a2, b2)

    assert method.inspect.state_vector() == pytest.approx(scope.inspect.state_vector())
    assert method.block_counts() == {"C-bell": 1}
    assert scope.block_counts() == {"bell": 1}


# ---- checkpoint and rewind -------------------------------------------------------


def test_rewind_returns_the_state_and_appends_the_undoing(qc: Circuit) -> None:
    """The state goes back; the record shows how. An editor's undo appears in the edit
    log — it does not erase your keystrokes from it."""
    a, b = qc.alloc_many(2)
    mark = qc.checkpoint()

    H(a)
    CNOT(a, b)
    assert qc.inspect.entanglement_entropy([a]) == pytest.approx(1.0)

    qc.rewind(mark)

    assert qc.inspect.probabilities()[0] == pytest.approx(1.0)
    assert [op.name for op in qc.history] == ["H", "CNOT", "CNOT", "H"]
    assert qc.gate_counts() == {"H": 2, "CNOT": 2}


def test_a_mark_taken_before_anything_happened_rewinds_to_the_start(qc: Circuit) -> None:
    mark = qc.checkpoint()
    q = qc.alloc("q")

    # The mark was taken with zero qubits, so it cannot be used after alloc.
    with pytest.raises(QsimError, match="which qubits exist"):
        qc.rewind(mark)

    mark = qc.checkpoint()
    T(q)
    H(q)
    qc.rewind(mark)

    assert qc.inspect.probabilities()[0] == pytest.approx(1.0)
    assert len(qc.history) == 4


def test_rewinding_twice_to_the_same_mark_is_harmless(qc: Circuit) -> None:
    """The second rewind undoes the first rewind *and* redoes the original gates, which
    cancel exactly. The state does not move; the tape grows, honestly saying so."""
    q = qc.alloc("q")
    mark = qc.checkpoint()
    H(q)
    T(q)

    qc.rewind(mark)
    once = qc.inspect.state_vector()
    qc.rewind(mark)

    assert qc.inspect.state_vector() == pytest.approx(once)
    assert qc.inspect.probabilities()[0] == pytest.approx(1.0)
    assert len(qc.history) == 8


def test_rewinding_to_a_later_mark_after_an_earlier_one_redoes_the_work(qc: Circuit) -> None:
    """A consequence of the tape never being rewritten: rewinding past a rewind is a
    redo. The inverse ops are ordinary ops, so undoing *them* re-applies the originals,
    and a mark still names the state the circuit was in when it was taken."""
    q = qc.alloc("q")
    early = qc.checkpoint()
    H(q)
    late = qc.checkpoint()
    at_late = qc.inspect.state_vector()
    T(q)

    qc.rewind(early)
    assert qc.inspect.probabilities()[0] == pytest.approx(1.0)

    qc.rewind(late)
    assert qc.inspect.state_vector() == pytest.approx(at_late)


def test_checkpoint_and_rewind_can_drive_a_parameter_sweep(qc: Circuit) -> None:
    """The notebook pattern: prepare once, then try each parameter from the same state.

    Cheap because nothing is copied — going back is just running the gates backwards.
    """
    q, e = qc.alloc_many(2)
    H(q)
    mark = qc.checkpoint()

    coherences: list[float] = []
    for theta in (0.0, np.pi / 2, np.pi):
        with qc.control(q):
            qsim.gates.Ry(e, theta=theta)
        coherences.append(qc.inspect.coherence(q))
        qc.rewind(mark)
        # Take a fresh mark each pass. The state here is the state at the old mark, but
        # the tape has grown, and reusing the old mark would make the next rewind undo
        # this pass's undoing as well — see the test below.
        mark = qc.checkpoint()

    assert coherences == pytest.approx([0.5, 0.5 * np.cos(np.pi / 4), 0.0], abs=1e-12)
    # Every sweep step left both its gate and its undo on the tape.
    assert qc.gate_counts()["Ry"] == 6


def test_reusing_one_mark_in_a_loop_makes_the_tape_grow_fast(qc: Circuit) -> None:
    """The price of an honest tape, stated out loud.

    Rewinding to the *same* mark twice does not just undo the newest work: the previous
    rewind is itself on the tape, so it gets undone too and then redone. The state is
    always right, and the op count doubles each pass. Retake the mark after each rewind
    (as the sweep above does) and the growth is linear.
    """
    q = qc.alloc("q")
    mark = qc.checkpoint()

    for _ in range(4):
        H(q)
        qc.rewind(mark)

    assert qc.inspect.probabilities()[0] == pytest.approx(1.0)
    assert len(qc.history) == 2**5 - 2


def test_rewind_is_refused_while_a_combinator_scope_is_open(qc: Circuit) -> None:
    a = qc.alloc()
    mark = qc.checkpoint()

    with pytest.raises(QsimError, match="combinator scope is open"):
        with qc.adjoint():
            H(a)
            qc.rewind(mark)


def test_a_mark_from_another_circuit_is_refused(qc: Circuit) -> None:
    other = Circuit(seed=2)
    other.alloc()
    mark = other.checkpoint()

    qc.alloc()
    with pytest.raises(QsimError, match="different circuit"):
        qc.rewind(mark)


def test_a_mark_pointing_past_the_end_of_the_tape_is_refused(qc: Circuit) -> None:
    """Only reachable by building a Checkpoint by hand: because rewind *appends*, a
    circuit's own marks can never outrun its history. The guard says so plainly rather
    than walking off the end of the list."""
    q = qc.alloc("q")
    H(q)
    invented = Checkpoint(qc, len(qc.history) + 5, qc._next_id, qc.n_qubits)

    with pytest.raises(QsimError, match="past the end of the tape"):
        qc.rewind(invented)


def test_rewinding_across_an_allocation_explains_the_axis_problem(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    mark = qc.checkpoint()
    qc.alloc()

    with pytest.raises(QsimError) as raised:
        qc.rewind(mark)

    message = str(raised.value)
    assert "which qubits exist" in message
    assert "renumbering the axes" in message


def test_rewinding_across_a_measurement_names_it_as_the_one_operation_with_no_inverse(
    qc: Circuit,
) -> None:
    q = qc.alloc("q")
    mark = qc.checkpoint()
    H(q)
    qc.measure(q)

    with pytest.raises(QsimError) as raised:
        qc.rewind(mark)

    message = str(raised.value)
    assert "severed" in message
    assert "inverse" in message
    assert "eraser" in message


def test_a_mark_taken_after_a_measurement_still_rewinds(qc: Circuit) -> None:
    """The tape is severed *at* the measurement, not poisoned forever. Everything after
    the cut is ordinary unitary history and walks backwards as usual."""
    q = qc.alloc("q")
    H(q)
    qc.measure(q)
    mark = qc.checkpoint()
    after = qc.inspect.state_vector()

    H(q)
    T(q)
    qc.rewind(mark)

    assert qc.inspect.state_vector() == pytest.approx(after)


def test_a_checkpoint_says_where_it_points() -> None:
    named = Circuit(name="demo", seed=3)
    named.alloc()
    assert repr(named.checkpoint()) == "<Checkpoint at op 0 of Circuit 'demo', 1 qubits>"

    anonymous = Circuit(seed=3)
    anonymous.alloc_many(2)
    H(anonymous.qubits[0])
    assert repr(anonymous.checkpoint()) == "<Checkpoint at op 1, 2 qubits>"


# ---- hooks ------------------------------------------------------------------------


def test_hooks_fire_in_history_order_with_the_state_already_updated(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    seen: list[tuple[str, float]] = []

    def watch(op: Op, circuit: Circuit) -> None:
        seen.append((op.name, circuit.inspect.entanglement_entropy([a])))

    qc.on_op(watch)
    H(a)
    CNOT(a, b)

    assert [name for name, _ in seen] == ["H", "CNOT"]
    # After H the pair is still a product state; after CNOT it is a full Bell pair.
    assert [entropy for _, entropy in seen] == pytest.approx([0.0, 1.0])


def test_a_hook_receives_measurements_with_their_outcome(qc: Circuit) -> None:
    q = qc.alloc("q")
    results: list[int | None] = []
    qc.on_op(lambda op, circuit: results.append(op.result))

    H(q)
    outcome = qc.measure(q)

    assert results == [None, outcome]


def test_two_hooks_both_fire_in_the_order_they_were_attached(qc: Circuit) -> None:
    q = qc.alloc("q")
    order: list[str] = []
    qc.on_op(lambda op, circuit: order.append("first"))
    qc.on_op(lambda op, circuit: order.append("second"))

    H(q)

    assert order == ["first", "second"]


def test_a_hook_may_remove_itself_while_it_is_firing(qc: Circuit) -> None:
    """Hooks are iterated over a snapshot, so a hook that detaches mid-fire does not
    make the loop skip the hook after it."""
    q = qc.alloc("q")
    seen: list[str] = []
    also_seen: list[str] = []

    handle_box: list[qsim.HookHandle] = []

    def once(op: Op, circuit: Circuit) -> None:
        seen.append(op.name)
        handle_box[0].remove()

    handle_box.append(qc.on_op(once))
    qc.on_op(lambda op, circuit: also_seen.append(op.name))

    H(q)
    T(q)

    assert seen == ["H"]
    assert also_seen == ["H", "T"]


def test_removing_a_hook_twice_is_harmless(qc: Circuit) -> None:
    q = qc.alloc("q")
    seen: list[str] = []
    handle = qc.on_op(lambda op, circuit: seen.append(op.name))

    handle.remove()
    handle.remove()
    H(q)

    assert seen == []


def test_a_hook_handle_says_whether_it_is_still_attached(qc: Circuit) -> None:
    def watch(op: Op, circuit: Circuit) -> None:  # pragma: no cover - never fires
        pass

    handle = qc.on_op(watch)
    assert repr(handle) == "<HookHandle watch, attached>"

    handle.remove()
    assert repr(handle) == "<HookHandle watch, removed>"


def test_a_hook_that_measures_is_refused(qc: Circuit) -> None:
    """Measuring from a hook would collapse the state out from under the very op the
    hook was called about, and put an op on the tape that no program line asked for."""
    q, other = qc.alloc_many(2)

    def peek(op: Op, circuit: Circuit) -> None:
        circuit.measure(other)

    qc.on_op(peek)

    with pytest.raises(QsimError) as raised:
        H(q)

    assert "Hooks watch the tape" in str(raised.value)
    assert "combinators" in str(raised.value)


def test_a_hook_that_rewinds_is_refused(qc: Circuit) -> None:
    q = qc.alloc("q")
    mark = qc.checkpoint()
    qc.on_op(lambda op, circuit: circuit.rewind(mark))

    with pytest.raises(QsimError, match="Hooks watch the tape"):
        H(q)


def test_hooks_survive_the_op_that_raised_and_keep_working(qc: Circuit) -> None:
    """The reentrancy flag is restored even when a hook fails, so one bad hook does not
    leave the circuit permanently refusing to run gates."""
    q, other = qc.alloc_many(2)
    handle = qc.on_op(lambda op, circuit: CZ(q, other))

    with pytest.raises(QsimError):
        H(q)

    handle.remove()
    X(q)  # the circuit still works
    assert [op.name for op in qc.history] == ["H", "X"]
