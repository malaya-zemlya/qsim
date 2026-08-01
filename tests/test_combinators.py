"""Combinators: record mode, control, adjoint, ancilla scopes, and reusable blocks."""

import numpy as np
import pytest

import qsim
from qsim import Circuit
from qsim.errors import DeadQubitError, DirtyAncillaError, NoCloningError, QsimError
from qsim.gates import CNOT, CZ, H, Rz, S, T, X, Y, Z


@qsim.gate
def bell(a, b) -> None:
    """Prepare a Bell pair on two fresh qubits."""
    H(a)
    CNOT(a, b)


# ---- record mode --------------------------------------------------------------


def test_gates_do_not_run_until_the_scope_closes(qc: Circuit) -> None:
    """Inside a combinator the circuit records instead of executing — that deferral is
    what makes it possible to transform a block before it happens."""
    a, b = qc.alloc_many(2)
    with qc.adjoint():
        H(a)
        assert qc.inspect.probabilities()[0] == pytest.approx(1.0)  # nothing ran yet
        assert qc.history == []
    assert qc.inspect.probabilities()[0] == pytest.approx(0.5)  # now it has


def test_an_empty_scope_does_nothing(qc: Circuit) -> None:
    qc.alloc_many(2)
    with qc.adjoint():
        pass
    assert qc.history == []


def test_an_exception_inside_a_scope_discards_the_recorded_gates(qc: Circuit) -> None:
    """Half a transformed block is worse than none, so the buffer is dropped."""
    a, b = qc.alloc_many(2)
    with pytest.raises(ValueError, match="boom"):
        with qc.adjoint():
            H(a)
            raise ValueError("boom")

    assert qc.history == []
    assert qc.inspect.probabilities()[0] == pytest.approx(1.0)


# ---- adjoint ------------------------------------------------------------------


def test_adjoint_reverses_the_order_and_inverts_each_gate(qc: Circuit) -> None:
    """Undoing "socks then shoes" is "shoes off then socks off" — both halves matter."""
    a, b = qc.alloc_many(2)
    with qc.adjoint():
        T(a)
        CNOT(a, b)
        S(b)

    assert [op.name for op in qc.history] == ["S†", "CNOT", "T†"]


def test_adjoint_negates_a_rotation_angle(qc: Circuit) -> None:
    a = qc.alloc()
    with qc.adjoint():
        Rz(a, theta=0.4)

    assert qc.history[0].name == "Rz"
    assert qc.history[0].params == (-0.4,)


def test_a_block_followed_by_its_adjoint_is_the_identity(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    H(a)
    T(a)
    before = qc.inspect.state_vector()

    def body() -> None:
        Rz(a, theta=0.7)
        CNOT(a, b)
        S(b)
        H(b)

    body()
    with qc.adjoint():
        body()

    assert qc.inspect.state_vector() == pytest.approx(before, abs=1e-14)


def test_measuring_inside_an_adjoint_scope_raises(qc: Circuit) -> None:
    """Measurement is the one operation with no inverse, so there is nothing to replay."""
    a = qc.alloc()
    with pytest.raises(QsimError, match="irreversible"):
        with qc.adjoint():
            qc.measure(a)


def test_the_measurement_message_explains_why_scopes_need_unitaries(qc: Circuit) -> None:
    a = qc.alloc()
    with pytest.raises(QsimError, match="destroys the superposition"):
        with qc.control(qc.alloc()):
            qc.measure(a)


# ---- control ------------------------------------------------------------------


def test_controlling_x_reproduces_cnot(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    H(a)
    with qc.control(a):
        X(b)

    expected = Circuit()
    c, d = expected.alloc_many(2)
    H(c)
    CNOT(c, d)
    assert qc.inspect.state_vector() == pytest.approx(expected.inspect.state_vector())


def test_a_superposed_control_gives_a_superposition_of_having_run(qc: Circuit) -> None:
    """Not a coin flip choosing whether to run the block — both branches, at once."""
    c, a, b = qc.alloc_many(3)
    H(c)
    with qc.control(c):
        bell(a, b)

    # |0⟩ branch: nothing happened. |1⟩ branch: a Bell pair formed.
    assert str(qc.inspect.ket()) == "0.707|000⟩ + 0.500|100⟩ + 0.500|111⟩"


def test_multiple_controls_all_have_to_be_one(qc: Circuit) -> None:
    c1, c2, t = qc.alloc_many(3)
    X(c1)
    with qc.control(c1, c2):
        X(t)
    assert qc.inspect.probabilities()[0b100] == pytest.approx(1.0)

    X(c2)
    with qc.control(c1, c2):
        X(t)
    assert qc.inspect.probabilities()[0b111] == pytest.approx(1.0)


def test_a_control_qubit_cannot_be_used_inside_its_own_block(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    with pytest.raises(NoCloningError, match="controls this block"):
        with qc.control(a):
            H(a)
            CNOT(a, b)


def test_control_records_the_controls_on_every_op_in_the_block(qc: Circuit) -> None:
    c, a, b = qc.alloc_many(3)
    with qc.control(c):
        bell(a, b)

    # The scope's control is prepended to whatever controls the gate already had, so
    # the CNOT inside the block ends up with two: the scope's, and its own.
    assert [op.controls for op in qc.history] == [(c._id,), (c._id, a._id)]


def test_a_released_qubit_cannot_be_used_as_a_control(qc: Circuit) -> None:
    a = qc.alloc()
    a._live = False
    with pytest.raises(DeadQubitError):
        with qc.control(a):
            pass


# ---- nesting and composition --------------------------------------------------


def test_scopes_nest(qc: Circuit) -> None:
    c, a, b = qc.alloc_many(3)
    X(c)
    with qc.control(c):
        with qc.adjoint():
            T(a)
            CNOT(a, b)

    assert [op.name for op in qc.history[1:]] == ["CNOT", "T†"]
    assert all(c._id in op.controls for op in qc.history[1:])


def test_nested_controls_are_the_same_as_one_scope_with_both(qc: Circuit) -> None:
    def build(nested: bool) -> np.ndarray:
        circuit = Circuit()
        c1, c2, t = circuit.alloc_many(3)
        H(c1)
        H(c2)
        if nested:
            with circuit.control(c1):
                with circuit.control(c2):
                    X(t)
        else:
            with circuit.control(c1, c2):
                X(t)
        return circuit.inspect.state_vector()

    assert build(nested=True) == pytest.approx(build(nested=False))


def test_controlling_an_inverse_equals_inverting_a_controlled_block(qc: Circuit) -> None:
    """control ∘ adjoint == adjoint ∘ control — the two commute."""

    def build(control_outside: bool) -> np.ndarray:
        circuit = Circuit()
        c, a, b = circuit.alloc_many(3)
        H(c)
        H(a)
        T(b)

        def body() -> None:
            Rz(a, theta=0.5)
            CNOT(a, b)
            S(a)

        if control_outside:
            with circuit.control(c):
                with circuit.adjoint():
                    body()
        else:
            with circuit.adjoint():
                with circuit.control(c):
                    body()
        return circuit.inspect.state_vector()

    assert build(control_outside=True) == pytest.approx(build(control_outside=False))


# ---- blocks -------------------------------------------------------------------


def test_a_block_runs_its_body(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    bell(a, b)
    assert str(qc.inspect.ket()) == "0.707|00⟩ + 0.707|11⟩"


def test_gate_counts_reports_what_ran_and_block_counts_reports_how_it_was_written(
    qc: Circuit,
) -> None:
    a, b, c, d = qc.alloc_many(4)
    bell(a, b)
    bell(c, d)

    assert qc.gate_counts() == {"H": 2, "CNOT": 2}
    assert qc.block_counts() == {"bell": 2}


def test_each_op_remembers_the_block_it_came_from(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    H(a)
    bell(a, b)

    assert [op.block for op in qc.history] == ["", "bell", "bell"]


def test_a_block_can_be_inverted(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    bell(a, b)
    bell.adjoint()(a, b)

    assert qc.inspect.probabilities()[0] == pytest.approx(1.0)
    assert [op.name for op in qc.history[2:]] == ["CNOT", "H"]


def test_a_block_can_be_controlled(qc: Circuit) -> None:
    c, a, b = qc.alloc_many(3)
    X(c)
    bell.controlled(c)(a, b)

    assert str(qc.inspect.ket()) == "0.707|100⟩ + 0.707|111⟩"


def test_a_block_with_a_classical_parameter_inverts_by_negating_it(qc: Circuit) -> None:
    """The angle is captured when the body is recorded, which is what lets the block be
    inverted long after the call site."""

    @qsim.gate
    def turn(q, angle: float) -> None:
        Rz(q, theta=angle)

    a = qc.alloc()
    turn.adjoint()(a, 0.3)
    assert qc.history[0].params == (-0.3,)


def test_blocks_can_be_nested(qc: Circuit) -> None:
    @qsim.gate
    def two_pairs(a, b, c, d) -> None:
        bell(a, b)
        bell(c, d)

    a, b, c, d = qc.alloc_many(4)
    two_pairs(a, b, c, d)

    assert qc.gate_counts() == {"H": 2, "CNOT": 2}
    assert qc.block_counts() == {"bell": 2, "two_pairs": 1}


def test_a_block_accepts_a_register(qc: Circuit) -> None:
    @qsim.gate
    def spread(reg) -> None:
        for q in reg:
            H(q)

    reg = qc.register(3)
    spread(reg)
    assert qc.gate_counts() == {"H": 3}


def test_a_block_keeps_its_name_and_docstring() -> None:
    assert bell.name == "bell"
    assert bell.__doc__ is not None and "Bell pair" in bell.__doc__
    assert repr(bell) == "<Block bell>"


def test_a_block_needs_a_qubit_to_know_which_circuit_it_is_on() -> None:
    @qsim.gate
    def nothing(x: int) -> None:  # pragma: no cover - never reaches the body
        pass

    with pytest.raises(QsimError, match="at least one qubit"):
        nothing(3)


def test_an_empty_register_does_not_identify_a_circuit(qc: Circuit) -> None:
    @qsim.gate
    def nothing(reg) -> None:  # pragma: no cover - never reaches the body
        pass

    from qsim.circuit import Register

    with pytest.raises(QsimError, match="at least one qubit"):
        nothing(Register(()))


# ---- ancilla scopes -----------------------------------------------------------


def test_clean_ancillas_are_returned_and_the_axes_disappear(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    with qc.ancilla(2) as scratch:
        assert qc.n_qubits == 3
        CNOT(a, scratch[0])
        CNOT(a, scratch[0])  # uncompute

    assert qc.n_qubits == 1
    assert qc.inspect.norm() == pytest.approx(1.0)


def test_dirty_ancillas_raise_on_the_way_out(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    with pytest.raises(DirtyAncillaError, match="interfere"):
        with qc.ancilla(1) as scratch:
            CNOT(a, scratch[0])


def test_the_dirty_message_reports_how_dirty(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    with pytest.raises(DirtyAncillaError, match="0.5"):
        with qc.ancilla(1) as scratch:
            CNOT(a, scratch[0])


def test_an_ancilla_handle_dies_when_the_scope_closes(qc: Circuit) -> None:
    qc.alloc()
    with qc.ancilla(1) as scratch:
        escaped = scratch[0]

    with pytest.raises(DeadQubitError, match="released"):
        H(escaped)


def test_surviving_qubits_are_renumbered_after_a_release(qc: Circuit) -> None:
    """The axis-lifecycle stress test: handles keep working because they hold ids."""
    a = qc.alloc()
    with qc.ancilla(1) as scratch:
        assert qc._axis_of[a._id] == 0
        assert qc._axis_of[scratch[0]._id] == 1
    # The ancilla occupied axis 1; after release, a is still axis 0 and still usable.
    X(a)
    assert qc.inspect.probabilities()[1] == pytest.approx(1.0)


def test_a_released_axis_frees_room_for_later_allocations(qc: Circuit) -> None:
    a = qc.alloc()
    with qc.ancilla(2):
        pass
    b = qc.alloc()
    CNOT(a, b)
    assert qc.n_qubits == 2
    assert qc._axis_of == {a._id: 0, b._id: 1}


def test_ancilla_scopes_nest(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    with qc.ancilla(1) as outer:
        CNOT(a, outer[0])
        with qc.ancilla(1) as inner:
            CNOT(outer[0], inner[0])
            CNOT(outer[0], inner[0])
        CNOT(a, outer[0])

    assert qc.n_qubits == 1
    assert qc.inspect.norm() == pytest.approx(1.0)


def test_an_exception_inside_an_ancilla_scope_retires_the_handles(qc: Circuit) -> None:
    """The original error must survive: the axes are left alone rather than sliced away
    from a state that may be entangled."""
    a = qc.alloc()
    H(a)
    escaped: list = []
    with pytest.raises(ValueError, match="boom"):
        with qc.ancilla(1) as scratch:
            CNOT(a, scratch[0])
            escaped.append(scratch[0])
            raise ValueError("boom")

    with pytest.raises(DeadQubitError):
        H(escaped[0])


def test_an_ancilla_scope_cannot_be_opened_inside_a_recording_scope(qc: Circuit) -> None:
    """Inside a control, adjoint or block scope the body has not run yet, so there
    would be nothing to verify on the way out — and the scratch would be released while
    ops referring to it were still queued. Allocate outside and pass it in."""
    a, b = qc.alloc_many(2)

    with pytest.raises(QsimError, match="pass it in"):
        with qc.control(a):
            with qc.ancilla(1):
                pass


def test_the_same_guard_protects_blocks(qc: Circuit) -> None:
    @qsim.gate
    def borrows_scratch(q) -> None:
        with q._circuit.ancilla(1):
            pass

    a = qc.alloc()
    with pytest.raises(QsimError, match="cannot open an ancilla scope"):
        borrows_scratch(a)


def test_a_failed_ancilla_check_leaves_the_mess_inspectable(qc: Circuit) -> None:
    """After DirtyAncillaError the handles stay live on purpose, so you can look at what
    went wrong — notebook 04 does exactly that."""
    a = qc.alloc()
    H(a)
    scratch_handle: list = []

    with pytest.raises(DirtyAncillaError):
        with qc.ancilla(1) as scratch:
            scratch_handle.append(scratch[0])
            CNOT(a, scratch[0])

    assert qc.inspect.entanglement_entropy([scratch_handle[0]]) == pytest.approx(1.0)


def test_borrowing_no_ancillas_is_refused(qc: Circuit) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        qc.ancilla(0)


def test_ancillas_start_in_the_zero_state(qc: Circuit) -> None:
    qc.alloc()
    with qc.ancilla(2) as scratch:
        assert qc.inspect.is_product(scratch)
        assert qc.inspect.bloch_vector(scratch[0])[2] == pytest.approx(1.0)


# ---- gate-level adjoint -------------------------------------------------------


def test_most_gates_are_their_own_inverse() -> None:
    for g in (H, X, Y, Z, CNOT, CZ):
        assert g.adjoint() is g


def test_the_three_exceptions_name_their_partner() -> None:
    from qsim.gates import SX

    assert S.adjoint().name == "S†"
    assert T.adjoint().name == "T†"
    assert SX.adjoint().name == "SX†"


def test_an_inverse_is_stable_and_mutual() -> None:
    """Asking twice gives the same object, and the partner points back."""
    assert S.adjoint() is S.adjoint()
    assert S.adjoint().adjoint() is S


def test_a_rotation_inverts_by_reading_its_angle_backwards(qc: Circuit) -> None:
    a = qc.alloc()
    Rz.adjoint()(a, theta=0.3)

    assert qc.history[0].name == "Rz"
    assert qc.history[0].params == (-0.3,)
    assert Rz.adjoint().adjoint() is Rz


def test_controlling_a_self_inverse_gate_keeps_it_self_inverse() -> None:
    controlled_h = H.controlled()
    assert controlled_h.adjoint() is controlled_h


def test_controlling_a_gate_controls_its_inverse_too() -> None:
    """Controlled-S inverts to controlled-S†, built on demand."""
    assert S.controlled().adjoint().name == "CS†"
    assert S.controlled().adjoint().n_controls == 1


def test_a_controlled_rotation_inverts_by_negating_its_angle(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    X(a)
    Rz.controlled().adjoint()(a, b, theta=0.5)
    assert qc.history[-1].params == (-0.5,)


def test_undoing_a_gate_returns_the_state(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    before = qc.inspect.state_vector()
    T(a)
    T.adjoint()(a)
    assert qc.inspect.state_vector() == pytest.approx(before)
