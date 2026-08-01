"""The state tensor and the gate kernels — including the bit-ordering convention.

Bit-ordering mistakes are the most common bug in quantum simulators, so the
convention is pinned here first and by example.
"""

import numpy as np
import pytest

from qsim import state
from qsim.gates import _H_MATRIX, _SWAP_TENSOR, _X_MATRIX, _Z_PHASES


def cnot_tensor() -> np.ndarray:
    """CNOT as a (2,2,2,2) tensor indexed [out_control, out_target, in_control, in_target]."""
    tensor = np.zeros((2, 2, 2, 2), dtype=np.complex128)
    for control in (0, 1):
        for target in (0, 1):
            tensor[control, control ^ target, control, target] = 1.0
    return tensor


# ---- shape and starting state -------------------------------------------------


def test_zero_state_has_one_axis_per_qubit() -> None:
    psi = state.zero_state(3)
    assert psi.shape == (2, 2, 2)


def test_zero_state_puts_all_amplitude_on_the_all_zeros_basis_state() -> None:
    psi = state.zero_state(3)
    assert psi[0, 0, 0] == 1.0
    assert np.sum(np.abs(psi) ** 2) == pytest.approx(1.0)


def test_zero_qubits_is_a_one_dimensional_space_holding_the_number_one() -> None:
    """Not a degenerate case: 0 qubits really do span a 1-dimensional space."""
    psi = state.zero_state(0)
    assert psi.shape == ()
    assert psi[()] == 1.0


def test_a_negative_number_of_qubits_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot have -1 qubits"):
        state.zero_state(-1)


# ---- the bit convention -------------------------------------------------------


def test_qubit_zero_is_the_most_significant_bit() -> None:
    """X on qubit 0 of |00> gives |10>, which as an integer is 2 — not 1."""
    psi = state.apply_1q(state.zero_state(2), _X_MATRIX, 0)

    assert psi[1, 0] == pytest.approx(1.0)
    # The same fact read off the flat vector: C-order reshape makes axis 0 slowest.
    assert np.argmax(np.abs(psi.reshape(-1))) == 0b10


def test_flipping_the_last_qubit_gives_the_least_significant_bit() -> None:
    psi = state.apply_1q(state.zero_state(2), _X_MATRIX, 1)

    assert psi[0, 1] == pytest.approx(1.0)
    assert np.argmax(np.abs(psi.reshape(-1))) == 0b01


# ---- single-qubit kernel ------------------------------------------------------


def test_hadamard_on_zero_gives_an_equal_superposition() -> None:
    psi = state.apply_1q(state.zero_state(1), _H_MATRIX, 0)
    assert psi == pytest.approx(np.array([1, 1]) / np.sqrt(2))


def test_a_gate_on_one_qubit_leaves_the_others_alone() -> None:
    psi = state.zero_state(3)
    psi = state.apply_1q(psi, _H_MATRIX, 1)

    # Only qubit 1 moved: amplitude is split between |000> and |010> and nowhere else.
    assert psi[0, 0, 0] == pytest.approx(1 / np.sqrt(2))
    assert psi[0, 1, 0] == pytest.approx(1 / np.sqrt(2))
    assert np.count_nonzero(np.abs(psi) > 1e-12) == 2


# ---- two-qubit kernel ---------------------------------------------------------


def test_apply_2q_with_a_cnot_tensor_matches_apply_controlled_with_x(random_state) -> None:
    """The two routes to a controlled gate must agree — one builds a 2-qubit tensor,
    the other slices the control axis and applies a 2x2 matrix."""
    psi = random_state(3)

    via_tensor = state.apply_2q(psi, cnot_tensor(), 0, 2)
    via_slicing = state.apply_controlled(psi, _X_MATRIX, [0], [2])

    assert via_tensor == pytest.approx(via_slicing)


def test_swap_exchanges_two_qubits() -> None:
    psi = state.apply_1q(state.zero_state(2), _X_MATRIX, 0)  # |10>
    psi = state.apply_2q(psi, _SWAP_TENSOR, 0, 1)
    assert psi[0, 1] == pytest.approx(1.0)


# ---- diagonal kernel ----------------------------------------------------------


def test_a_diagonal_gate_never_changes_any_amplitude_magnitude(random_state) -> None:
    """Structural, not approximate: diagonal gates only rotate phases."""
    psi = random_state(3)
    after = state.apply_diag(psi, _Z_PHASES, 1)

    # Exact equality, because multiplying by +-1 cannot change a magnitude at all.
    assert np.array_equal(np.abs(psi), np.abs(after))


def test_z_negates_only_the_amplitudes_where_the_qubit_is_one() -> None:
    psi = np.array([[1, 2], [3, 4]], dtype=np.complex128)
    after = state.apply_diag(psi, _Z_PHASES, 1)
    assert after == pytest.approx(np.array([[1, -2], [3, -4]]))


# ---- controlled kernel --------------------------------------------------------


def test_a_controlled_gate_does_nothing_when_the_control_is_zero() -> None:
    psi = state.zero_state(2)  # control (qubit 0) is |0>
    after = state.apply_controlled(psi, _X_MATRIX, [0], [1])
    assert np.array_equal(psi, after)


def test_a_controlled_gate_fires_when_the_control_is_one() -> None:
    psi = state.apply_1q(state.zero_state(2), _X_MATRIX, 0)  # |10>
    after = state.apply_controlled(psi, _X_MATRIX, [0], [1])
    assert after[1, 1] == pytest.approx(1.0)


def test_two_controls_require_both_to_be_one() -> None:
    psi = state.apply_1q(state.zero_state(3), _X_MATRIX, 0)  # |100>
    after = state.apply_controlled(psi, _X_MATRIX, [0, 1], [2])
    assert np.array_equal(psi, after)

    psi = state.apply_1q(psi, _X_MATRIX, 1)  # |110>
    after = state.apply_controlled(psi, _X_MATRIX, [0, 1], [2])
    assert after[1, 1, 1] == pytest.approx(1.0)


def test_a_controlled_two_qubit_gate_swaps_only_in_the_control_one_subspace() -> None:
    """Fredkin: controlled-SWAP, the two-target branch of the controlled kernel."""
    psi = state.zero_state(3)
    psi = state.apply_1q(psi, _X_MATRIX, 0)  # control on
    psi = state.apply_1q(psi, _X_MATRIX, 1)  # |110>
    after = state.apply_controlled(psi, _SWAP_TENSOR, [0], [1, 2])
    assert after[1, 0, 1] == pytest.approx(1.0)


def test_target_axes_are_renumbered_around_dropped_control_axes() -> None:
    """With the control at axis 1, the target at axis 2 sits at axis 1 inside the slice."""
    psi = state.apply_1q(state.zero_state(3), _X_MATRIX, 1)  # |010>
    after = state.apply_controlled(psi, _X_MATRIX, [1], [2])
    assert after[0, 1, 1] == pytest.approx(1.0)


def test_controlled_diagonal_negates_only_the_all_ones_amplitude() -> None:
    psi = np.ones((2, 2), dtype=np.complex128) / 2.0
    after = state.apply_controlled_diag(psi, _Z_PHASES, [0], 1)
    assert after == pytest.approx(np.array([[0.5, 0.5], [0.5, -0.5]]))


# ---- unitarity ----------------------------------------------------------------


@pytest.mark.parametrize("apply_gate", ["1q", "2q", "diag", "controlled", "controlled_diag"])
def test_every_kernel_preserves_the_norm(random_state, apply_gate: str) -> None:
    """Unitary means norm-preserving, and the norm is the total probability, which is 1."""
    psi = random_state(4)
    kernels = {
        "1q": lambda p: state.apply_1q(p, _H_MATRIX, 2),
        "2q": lambda p: state.apply_2q(p, _SWAP_TENSOR, 0, 3),
        "diag": lambda p: state.apply_diag(p, _Z_PHASES, 1),
        "controlled": lambda p: state.apply_controlled(p, _H_MATRIX, [0], [2]),
        "controlled_diag": lambda p: state.apply_controlled_diag(p, _Z_PHASES, [0], [1][0]),
    }
    after = kernels[apply_gate](psi)
    assert np.linalg.norm(after) == pytest.approx(1.0, abs=1e-14)


# ---- measurement kernel -------------------------------------------------------


def test_measuring_a_definite_qubit_always_returns_that_value() -> None:
    psi = state.apply_1q(state.zero_state(1), _X_MATRIX, 0)  # |1>
    rng = np.random.default_rng(0)
    for _ in range(10):
        outcome, psi = state.measure_axis(psi, 0, rng)
        assert outcome == 1


def test_measurement_leaves_the_state_normalized() -> None:
    psi = state.apply_1q(state.zero_state(2), _H_MATRIX, 0)
    _, after = state.measure_axis(psi, 0, np.random.default_rng(7))
    assert np.linalg.norm(after) == pytest.approx(1.0)


def test_measurement_zeroes_the_branch_that_did_not_happen() -> None:
    psi = state.apply_1q(state.zero_state(1), _H_MATRIX, 0)
    outcome, after = state.measure_axis(psi, 0, np.random.default_rng(7))
    assert after[1 - outcome] == 0.0
    assert abs(after[outcome]) == pytest.approx(1.0)


# ---- precision ----------------------------------------------------------------


def test_the_default_amplitude_dtype_is_double_precision() -> None:
    assert state.get_dtype() == np.dtype(np.complex128)
    assert state.zero_state(1).dtype == np.complex128


def test_single_precision_can_be_selected_globally() -> None:
    """complex64 exists so you can watch rounding error accumulate (design doc §9, T17)."""
    try:
        state.set_dtype(np.complex64)
        assert state.zero_state(2).dtype == np.complex64
    finally:
        state.set_dtype(np.complex128)


def test_a_real_dtype_is_refused_because_phase_is_what_interferes() -> None:
    with pytest.raises(ValueError, match="amplitudes must be complex"):
        state.set_dtype(np.float64)
    assert state.get_dtype() == np.dtype(np.complex128)
