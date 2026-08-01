"""Circuits, qubit handles and registers: who owns the state, and what a handle is."""

import copy

import numpy as np
import pytest

from qsim import Circuit
from qsim.errors import DeadQubitError, NoCloningError, QsimError
from qsim.gates import CNOT, H, X

# ---- allocation ---------------------------------------------------------------


def test_a_new_circuit_has_no_qubits() -> None:
    assert Circuit().n_qubits == 0


def test_allocating_a_qubit_adds_an_axis_to_the_state(qc: Circuit) -> None:
    qc.alloc()
    assert qc._psi.shape == (2,)
    qc.alloc()
    assert qc._psi.shape == (2, 2)


def test_a_freshly_allocated_qubit_is_in_the_zero_state(qc: Circuit) -> None:
    qc.alloc()
    assert qc.inspect.probabilities() == pytest.approx([1.0, 0.0])


def test_allocating_mid_circuit_leaves_existing_qubits_undisturbed(qc: Circuit) -> None:
    """A new qubit is tensored on, so it is unentangled with everything already there."""
    a = qc.alloc()
    H(a)
    b = qc.alloc()

    assert qc.inspect.is_product([b])
    assert qc.inspect.probabilities() == pytest.approx([0.5, 0.0, 0.5, 0.0])


def test_the_constructor_can_preallocate_qubits() -> None:
    qc = Circuit(3)
    assert qc.n_qubits == 3
    assert len(qc.qubits) == 3


def test_preallocated_qubits_are_reachable_through_the_qubits_property() -> None:
    a, b = Circuit(2).qubits
    assert a is not b


def test_alloc_many_returns_one_handle_per_qubit(qc: Circuit) -> None:
    handles = qc.alloc_many(3)
    assert len(handles) == 3
    assert qc.n_qubits == 3


def test_allocating_zero_qubits_is_refused(qc: Circuit) -> None:
    """alloc_many always returns a tuple, so asking for none is a mistake, not an empty tuple."""
    with pytest.raises(ValueError, match="at least 1"):
        qc.alloc_many(0)


def test_a_register_of_no_qubits_is_refused(qc: Circuit) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        qc.register(0)


# ---- qubit handles ------------------------------------------------------------


def test_copying_a_qubit_handle_raises_the_no_cloning_error(qc: Circuit) -> None:
    """The no-cloning theorem, surfaced at the API: an unknown state cannot be duplicated."""
    a = qc.alloc()
    with pytest.raises(NoCloningError, match="no-cloning theorem"):
        copy.copy(a)


def test_deep_copying_a_qubit_handle_raises_too(qc: Circuit) -> None:
    a = qc.alloc()
    with pytest.raises(NoCloningError, match="no-cloning theorem"):
        copy.deepcopy(a)


def test_a_qubit_has_no_state_attribute(qc: Circuit) -> None:
    """Deliberate: an entangled qubit has no state of its own to report."""
    a = qc.alloc()
    assert not hasattr(a, "state")
    assert not hasattr(a, "value")
    assert not hasattr(a, "amplitude")


def test_qubit_equality_is_identity(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    assert a == a
    assert a != b
    assert len({a, b, a}) == 2


def test_qubit_repr_names_the_qubit_and_its_circuit(qc: Circuit) -> None:
    a = qc.alloc()
    assert repr(a) == "<Qubit q0 of Circuit 'test'>"


def test_qubit_repr_omits_an_unnamed_circuit() -> None:
    a = Circuit().alloc()
    assert repr(a) == "<Qubit q0>"


def test_qubit_repr_says_so_once_released(qc: Circuit) -> None:
    a = qc.alloc()
    a._live = False  # Phase 2 preview: this is what releasing an ancilla will do.
    assert "released" in repr(a)


def test_qubits_can_be_given_names(qc: Circuit) -> None:
    a = qc.alloc("control")
    assert a.name == "control"


# ---- handle validation --------------------------------------------------------


def test_using_a_released_handle_raises_a_dead_qubit_error(qc: Circuit) -> None:
    a = qc.alloc()
    a._live = False  # Phase 2 preview: ancilla scope exit will do this.
    with pytest.raises(DeadQubitError, match="no longer refers to a qubit"):
        H(a)


def test_mixing_qubits_from_two_circuits_raises(qc: Circuit) -> None:
    """Two circuits are two separate physical systems; a gate across them is not an operation."""
    other = Circuit()
    a = qc.alloc()
    b = other.alloc()
    with pytest.raises(QsimError, match="different circuit"):
        CNOT(a, b)


def test_giving_the_same_qubit_twice_raises_the_no_cloning_error(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    with pytest.raises(NoCloningError, match="no-cloning theorem"):
        CNOT(a, a)


# ---- registers ----------------------------------------------------------------


def test_a_register_indexes_like_a_sequence(qc: Circuit) -> None:
    reg = qc.register(3, name="x")
    assert len(reg) == 3
    assert list(reg)[0] is reg[0]
    assert reg[0].name == "x0"


def test_slicing_a_register_gives_a_register_over_the_same_qubits(qc: Circuit) -> None:
    reg = qc.register(4)
    half = reg[:2]
    assert len(half) == 2
    assert half[0] is reg[0]


def test_reversing_a_register_reverses_the_bit_order(qc: Circuit) -> None:
    """The QFT needs this: its output comes out with the bits the other way round."""
    reg = qc.register(3)
    assert list(reg.reversed()) == list(reg)[::-1]


def test_registers_can_be_concatenated(qc: Circuit) -> None:
    a, b = qc.register(2), qc.register(3)
    assert len(a.concat(b)) == 5


def test_an_unnamed_register_still_names_its_qubits(qc: Circuit) -> None:
    reg = qc.register(2)
    assert reg[0].name == "q0"


def test_a_register_keeps_the_name_it_was_given(qc: Circuit) -> None:
    reg = qc.register(2, name="counting")
    assert reg.name == "counting"
    assert reg[:1].name == "counting"


def test_register_repr_reports_its_size(qc: Circuit) -> None:
    assert repr(qc.register(2, name="x")) == "<Register 'x' of 2 qubits>"
    assert repr(qc.register(2)) == "<Register of 2 qubits>"


# ---- encoding integers --------------------------------------------------------


def test_encode_prepares_a_register_in_a_basis_state(qc: Circuit) -> None:
    reg = qc.register(3)
    reg.encode(5)
    assert str(qc.inspect.ket()) == "1.000|101⟩"


def test_encode_puts_the_most_significant_bit_in_the_first_qubit(qc: Circuit) -> None:
    reg = qc.register(2)
    reg.encode(2)
    assert str(qc.inspect.ket()) == "1.000|10⟩"


def test_encoding_a_value_too_large_for_the_register_is_refused(qc: Circuit) -> None:
    reg = qc.register(2)
    with pytest.raises(ValueError, match="holds values 0 to 3"):
        reg.encode(4)


def test_encoding_a_negative_value_is_refused(qc: Circuit) -> None:
    with pytest.raises(ValueError, match="holds values 0 to 3"):
        qc.register(2).encode(-1)


def test_encoding_a_register_that_is_not_in_the_zero_state_is_refused(qc: Circuit) -> None:
    """X gates only 'set' a register that started at zero; otherwise they scramble it."""
    reg = qc.register(2)
    H(reg[0])
    with pytest.raises(Exception, match="not in"):
        reg.encode(1)


# ---- history ------------------------------------------------------------------


def test_the_history_records_every_operation_in_order(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    H(a)
    CNOT(a, b)

    names = [op.name for op in qc.history]
    assert names == ["H", "CNOT"]


def test_a_controlled_op_records_its_controls_separately_from_its_targets(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    CNOT(a, b)

    op = qc.history[0]
    assert op.controls == (a._id,)
    assert op.qubit_ids == (b._id,)
    assert op.all_qubit_ids == (a._id, b._id)


def test_a_rotation_records_its_angle(qc: Circuit) -> None:
    from qsim.gates import Rz

    a = qc.alloc()
    Rz(a, theta=0.5)
    assert qc.history[0].params == (0.5,)


def test_the_history_cannot_be_edited_from_outside(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    qc.history.clear()
    assert len(qc.history) == 1


def test_gate_counts_tallies_the_history(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    H(a)
    H(b)
    CNOT(a, b)
    assert qc.gate_counts() == {"H": 2, "CNOT": 1}


def test_an_empty_circuit_has_zero_depth(qc: Circuit) -> None:
    assert qc.depth() == 0


def test_gates_on_disjoint_qubits_share_a_layer(qc: Circuit) -> None:
    """Depth, not gate count, is what sets how long a circuit takes to run."""
    a, b = qc.alloc_many(2)
    H(a)
    H(b)
    assert qc.depth() == 1


def test_gates_sharing_a_qubit_stack_into_layers(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    H(a)
    CNOT(a, b)
    X(b)
    assert qc.depth() == 3


# ---- misc ---------------------------------------------------------------------


def test_circuit_repr_summarizes_size_and_history(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    assert repr(qc) == "<Circuit 'test' with 1 qubits, 1 ops>"


def test_an_unnamed_circuit_reprs_without_a_label() -> None:
    assert repr(Circuit()) == "<Circuit with 0 qubits, 0 ops>"


def test_a_circuit_can_be_built_in_single_precision() -> None:
    qc = Circuit(1, dtype=np.complex64)
    H(qc.qubits[0])
    # The gate matrices are complex128; applying one must not silently upgrade the
    # circuit's precision, or the T17 experiment would measure nothing.
    assert qc._psi.dtype == np.complex64
    assert qc.inspect.norm() == pytest.approx(1.0, abs=1e-6)
