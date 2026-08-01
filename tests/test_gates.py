"""The gate set: what each gate does, and the guarantees that hold across all of them.

Full 4x4 and 8x8 matrices appear in this file (built with ``np.kron``) purely to
check the library against an independent construction. The library itself never
builds one — that is the point of the tensor-contraction kernels.
"""

import numpy as np
import pytest

from qsim import Circuit
from qsim.errors import QsimError
from qsim.gates import (
    CNOT,
    CZ,
    GATES,
    SWAP,
    SX,
    CPhase,
    Fredkin,
    H,
    Phase,
    Rx,
    Ry,
    Rz,
    S,
    T,
    Toffoli,
    X,
    Y,
    Z,
)

I2 = np.eye(2, dtype=np.complex128)
P0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)  # |0><0|
P1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)  # |1><1|
X_M = np.array([[0, 1], [1, 0]], dtype=np.complex128)


def state_of(qc: Circuit) -> np.ndarray:
    return qc.inspect.state_vector()


# ---- single-qubit gates -------------------------------------------------------


def test_hadamard_creates_an_equal_superposition(qc: Circuit) -> None:
    H(qc.alloc())
    assert state_of(qc) == pytest.approx([1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_x_flips_a_bit(qc: Circuit) -> None:
    X(qc.alloc())
    assert state_of(qc) == pytest.approx([0, 1])


def test_y_flips_the_bit_and_the_phase(qc: Circuit) -> None:
    Y(qc.alloc())
    assert state_of(qc) == pytest.approx([0, 1j])


def test_z_leaves_the_zero_state_completely_alone(qc: Circuit) -> None:
    """A phase flip does nothing observable to |0> — it needs a superposition to act on."""
    Z(qc.alloc())
    assert state_of(qc) == pytest.approx([1, 0])


def test_z_turns_plus_into_minus(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    Z(a)
    assert state_of(qc) == pytest.approx([1 / np.sqrt(2), -1 / np.sqrt(2)])


def test_s_is_a_quarter_turn_and_squares_to_z(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    S(a)
    S(a)
    assert state_of(qc) == pytest.approx([1 / np.sqrt(2), -1 / np.sqrt(2)])


def test_t_applied_twice_is_s(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    T(a)
    T(a)
    assert state_of(qc) == pytest.approx([1 / np.sqrt(2), 1j / np.sqrt(2)])


def test_sx_applied_twice_is_a_bit_flip(qc: Circuit) -> None:
    """There is no 'half of a classical NOT' — the quantum operation space is bigger."""
    a = qc.alloc()
    SX(a)
    SX(a)
    assert np.abs(state_of(qc)) == pytest.approx([0, 1])


def test_applying_hadamard_twice_is_the_identity(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    H(a)
    assert state_of(qc) == pytest.approx([1, 0])


# ---- rotations ----------------------------------------------------------------


def test_a_rotation_by_zero_does_nothing(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    before = state_of(qc)
    Rx(a, theta=0.0)
    Ry(a, theta=0.0)
    Rz(a, theta=0.0)
    assert state_of(qc) == pytest.approx(before)


def test_a_rotation_by_two_pi_returns_the_state_with_a_minus_sign(qc: Circuit) -> None:
    """Spin-1/2 systems really do behave this way: a full turn is not the identity."""
    a = qc.alloc()
    H(a)
    before = state_of(qc)
    Rx(a, theta=2 * np.pi)
    assert state_of(qc) == pytest.approx(-before)


def test_rx_by_pi_is_a_bit_flip_up_to_phase(qc: Circuit) -> None:
    a = qc.alloc()
    Rx(a, theta=np.pi)
    assert np.abs(state_of(qc)) == pytest.approx([0, 1])


def test_ry_moves_amplitude_without_introducing_complex_phases(qc: Circuit) -> None:
    a = qc.alloc()
    Ry(a, theta=np.pi / 2)
    assert state_of(qc) == pytest.approx([1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_rz_cannot_change_any_measurement_probability(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    before = qc.inspect.probabilities()
    Rz(a, theta=0.7)
    assert qc.inspect.probabilities() == pytest.approx(before)


def test_phase_leaves_the_zero_amplitude_alone(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    Phase(a, theta=np.pi / 2)
    assert state_of(qc) == pytest.approx([1 / np.sqrt(2), 1j / np.sqrt(2)])


def test_the_angle_must_be_passed_by_keyword(qc: Circuit) -> None:
    """Every positional argument to a gate is a qubit, so an angle must be named."""
    a = qc.alloc()
    with pytest.raises(TypeError):
        Rz(a, 0.5)  # type: ignore[call-arg]


# ---- two-qubit gates ----------------------------------------------------------


def test_cnot_flips_the_target_only_when_the_control_is_one(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    CNOT(a, b)
    assert state_of(qc) == pytest.approx([1, 0, 0, 0])

    X(a)
    CNOT(a, b)
    assert state_of(qc) == pytest.approx([0, 0, 0, 1])


def test_cnot_on_a_superposition_creates_entanglement(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    H(a)
    CNOT(a, b)
    assert state_of(qc) == pytest.approx([1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])


def test_cnot_matches_an_independently_built_matrix(qc: Circuit) -> None:
    """Checked against |0><0| (x) I + |1><1| (x) X, assembled with np.kron in the test."""
    a, b = qc.alloc_many(2)
    H(a)
    H(b)
    T(b)
    before = state_of(qc)
    CNOT(a, b)

    cnot_matrix = np.kron(P0, I2) + np.kron(P1, X_M)
    assert state_of(qc) == pytest.approx(cnot_matrix @ before)


def test_cz_negates_only_the_all_ones_amplitude(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    H(a)
    H(b)
    CZ(a, b)
    assert state_of(qc) == pytest.approx(np.array([1, 1, 1, -1]) / 2)


def test_cz_is_symmetric_in_its_two_qubits(qc: Circuit) -> None:
    """For a diagonal gate, 'control' and 'target' are our labels, not nature's."""
    a, b = qc.alloc_many(2)
    H(a)
    H(b)
    CZ(a, b)
    forwards = state_of(qc)

    other = Circuit()
    c, d = other.alloc_many(2)
    H(c)
    H(d)
    CZ(d, c)
    assert forwards == pytest.approx(other.inspect.state_vector())


def test_cphase_rotates_only_the_all_ones_amplitude(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    H(a)
    H(b)
    CPhase(a, b, theta=np.pi / 2)
    assert state_of(qc) == pytest.approx(np.array([1, 1, 1, 1j]) / 2)


def test_swap_exchanges_two_qubits(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    X(a)
    SWAP(a, b)
    assert state_of(qc) == pytest.approx([0, 1, 0, 0])


def test_swap_equals_three_alternating_cnots(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    H(a)
    T(a)
    SWAP(a, b)
    swapped = state_of(qc)

    other = Circuit()
    c, d = other.alloc_many(2)
    H(c)
    T(c)
    CNOT(c, d)
    CNOT(d, c)
    CNOT(c, d)
    assert swapped == pytest.approx(other.inspect.state_vector())


# ---- three-qubit gates --------------------------------------------------------


def test_toffoli_flips_the_target_only_when_both_controls_are_one(qc: Circuit) -> None:
    """The reversible AND — this is how classical logic embeds into quantum circuits."""
    a, b, c = qc.alloc_many(3)
    X(a)
    Toffoli(a, b, c)
    assert state_of(qc) == pytest.approx([0, 0, 0, 0, 1, 0, 0, 0])  # |100>

    X(b)
    Toffoli(a, b, c)
    assert state_of(qc) == pytest.approx([0, 0, 0, 0, 0, 0, 0, 1])  # |111>


def test_fredkin_swaps_its_targets_only_when_the_control_is_one(qc: Circuit) -> None:
    a, b, c = qc.alloc_many(3)
    X(b)
    Fredkin(a, b, c)
    assert state_of(qc) == pytest.approx([0, 0, 1, 0, 0, 0, 0, 0])  # |010>, unchanged

    X(a)
    Fredkin(a, b, c)
    assert state_of(qc) == pytest.approx([0, 0, 0, 0, 0, 1, 0, 0])  # |101>


def test_toffoli_matches_an_independently_built_matrix(qc: Circuit) -> None:
    a, b, c = qc.alloc_many(3)
    for q in (a, b, c):
        H(q)
    T(c)
    before = state_of(qc)
    Toffoli(a, b, c)

    # |11><11| (x) X on the target, identity elsewhere.
    toffoli = np.kron(np.kron(P1, P1), X_M) + (
        np.eye(8, dtype=np.complex128) - np.kron(np.kron(P1, P1), I2)
    )
    assert state_of(qc) == pytest.approx(toffoli @ before)


# ---- gate algebra -------------------------------------------------------------


def test_any_gate_can_be_given_extra_controls(qc: Circuit) -> None:
    """Control is slicing, so controlling a gate does not change its matrix."""
    controlled_h = H.controlled()
    assert controlled_h.name == "CH"
    assert controlled_h.n_controls == 1

    a, b = qc.alloc_many(2)
    X(a)
    controlled_h(a, b)
    assert state_of(qc) == pytest.approx([0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)])


def test_a_parametrized_gate_can_be_given_extra_controls(qc: Circuit) -> None:
    crz = Rz.controlled()
    assert crz.name == "CRz"

    a, b = qc.alloc_many(2)
    X(a)
    H(b)
    crz(a, b, theta=np.pi)
    assert state_of(qc) == pytest.approx([0, 0, -1j / np.sqrt(2), 1j / np.sqrt(2)])


def test_a_gate_reports_how_many_qubits_it_needs() -> None:
    assert H.n_qubits == 1
    assert CNOT.n_qubits == 2
    assert Toffoli.n_qubits == 3
    assert Fredkin.n_qubits == 3


def test_giving_a_gate_the_wrong_number_of_qubits_raises(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    with pytest.raises(QsimError, match="acts on 3 qubit"):
        Toffoli(a, b)


def test_gate_repr_names_it_and_its_arity() -> None:
    assert repr(CNOT) == "<Gate CNOT on 2 qubit(s)>"


# ---- inverses (declared now, used by Phase 2's adjoint) -----------------------


def test_most_fixed_gates_are_their_own_inverse(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    op = qc.history[0]
    assert H.adjoint_op(op).name == "H"


def test_s_and_t_name_their_daggered_partner(qc: Circuit) -> None:
    a = qc.alloc()
    S(a)
    T(a)
    SX(a)
    assert S.adjoint_op(qc.history[0]).name == "S†"
    assert T.adjoint_op(qc.history[1]).name == "T†"
    assert SX.adjoint_op(qc.history[2]).name == "SX†"


def test_the_daggered_gates_undo_their_partners(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    before = state_of(qc)
    T(a)
    GATES["T†"](a)  # type: ignore[operator]
    assert state_of(qc) == pytest.approx(before)

    S(a)
    GATES["S†"](a)  # type: ignore[operator]
    SX(a)
    GATES["SX†"](a)  # type: ignore[operator]
    assert state_of(qc) == pytest.approx(before)


def test_a_rotation_is_undone_by_the_opposite_angle(qc: Circuit) -> None:
    a = qc.alloc()
    Rz(a, theta=0.4)
    inverse = Rz.adjoint_op(qc.history[0])
    assert inverse.name == "Rz"
    assert inverse.params == (-0.4,)


def test_an_inverted_op_keeps_the_qubits_it_acted_on(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    CPhase(a, b, theta=0.3)
    inverse = CPhase.adjoint_op(qc.history[0])
    assert inverse.controls == (a._id,)
    assert inverse.qubit_ids == (b._id,)


def test_every_public_gate_is_in_the_registry() -> None:
    """Phase 2 replays recorded history by name, so every name must resolve."""
    for gate in (H, X, Y, Z, S, T, SX, SWAP, CNOT, CZ, Toffoli, Fredkin,
                 Rx, Ry, Rz, Phase, CPhase):
        assert GATES[gate.name] is gate
