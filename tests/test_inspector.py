"""The Inspector — the things a real quantum computer would never let you see."""

import numpy as np
import pytest

from qsim import Circuit
from qsim.errors import DirtyAncillaError
from qsim.gates import CNOT, H, S, T, X, Y, Z

# ---- raw state ----------------------------------------------------------------


def test_the_state_tensor_has_one_axis_per_qubit(qc: Circuit) -> None:
    qc.alloc_many(3)
    assert qc.inspect.state_tensor().shape == (2, 2, 2)


def test_the_state_vector_is_indexed_with_qubit_zero_as_the_high_bit(qc: Circuit) -> None:
    a, _ = qc.alloc_many(2)
    X(a)
    assert np.argmax(np.abs(qc.inspect.state_vector())) == 0b10


def test_editing_the_returned_state_cannot_corrupt_the_circuit(qc: Circuit) -> None:
    qc.alloc()
    stolen = qc.inspect.state_vector()
    stolen[0] = 99.0
    assert qc.inspect.norm() == pytest.approx(1.0)


def test_amplitude_looks_up_one_basis_state_by_its_bitstring(bell_pair) -> None:
    qc, _, _ = bell_pair
    assert qc.inspect.amplitude("00") == pytest.approx(1 / np.sqrt(2))
    assert qc.inspect.amplitude("01") == 0


def test_a_malformed_bitstring_is_refused(bell_pair) -> None:
    qc, _, _ = bell_pair
    with pytest.raises(ValueError, match="expected a string of 2 characters"):
        qc.inspect.amplitude("0")
    with pytest.raises(ValueError, match="expected a string of 2 characters"):
        qc.inspect.amplitude("0x")


def test_probabilities_are_squared_amplitude_magnitudes(bell_pair) -> None:
    qc, _, _ = bell_pair
    assert qc.inspect.probabilities() == pytest.approx([0.5, 0, 0, 0.5])


def test_the_norm_is_one_and_stays_one(bell_pair) -> None:
    qc, a, _ = bell_pair
    assert qc.inspect.norm() == pytest.approx(1.0)
    T(a)
    assert qc.inspect.norm() == pytest.approx(1.0)


def test_sampling_does_not_disturb_the_measurements_that_follow_it() -> None:
    """Sampling is a simulator cheat, so it draws from its own random stream. Adding a
    sample() call to a seeded notebook must not rewrite the measurements below it."""

    def run(with_sampling: bool) -> list[int]:
        qc = Circuit(seed=5)
        reg = qc.register(3)
        for q in reg:
            H(q)
        if with_sampling:
            qc.inspect.sample(100)
        return [qc.measure(q) for q in reg]

    assert run(with_sampling=True) == run(with_sampling=False)


def test_a_product_state_reports_zero_entropy_not_negative_zero(qc: Circuit) -> None:
    """-1 * log(1) is negative zero in floating point, which would print alarmingly."""
    qc.alloc()
    assert str(qc.inspect.entanglement_entropy(list(qc.qubits))) == "0.0"


def test_sampling_does_not_collapse_the_state(bell_pair) -> None:
    """The cheat: a real machine would have to rerun the circuit for every shot."""
    qc, _, _ = bell_pair
    counts = qc.inspect.sample(500)

    assert set(counts) <= {"00", "11"}
    assert sum(counts.values()) == 500
    assert qc.inspect.probabilities() == pytest.approx([0.5, 0, 0, 0.5])


# ---- subsystems ---------------------------------------------------------------


def test_an_unentangled_qubit_has_a_pure_reduced_density_matrix(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    rho = qc.inspect.reduced_density_matrix([a])

    assert rho == pytest.approx(np.full((2, 2), 0.5))
    # Pure means rho^2 == rho.
    assert rho @ rho == pytest.approx(rho)


def test_half_a_bell_pair_is_the_maximally_mixed_state(bell_pair) -> None:
    """The whole pair is in a perfectly definite state; each half is as uninformative
    as a coin flip. That is what entanglement costs."""
    qc, a, _ = bell_pair
    rho = qc.inspect.reduced_density_matrix([a])

    assert rho == pytest.approx(np.eye(2) / 2)
    # The off-diagonal coherences are gone: no superposition survives locally.
    assert rho[0, 1] == pytest.approx(0)


def test_the_density_matrix_of_the_whole_system_is_a_pure_projector(bell_pair) -> None:
    qc, a, b = bell_pair
    rho = qc.inspect.reduced_density_matrix([a, b])
    assert np.trace(rho) == pytest.approx(1.0)
    assert rho @ rho == pytest.approx(rho)


def test_a_product_state_has_zero_entanglement_entropy(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    H(a)
    T(b)
    assert qc.inspect.entanglement_entropy([a]) == pytest.approx(0.0, abs=1e-12)
    assert qc.inspect.is_product([a])


def test_a_bell_pair_carries_exactly_one_bit_of_entanglement(bell_pair) -> None:
    qc, a, _ = bell_pair
    assert qc.inspect.entanglement_entropy([a]) == pytest.approx(1.0, abs=1e-12)
    assert not qc.inspect.is_product([a])


def test_entropy_can_be_measured_in_other_bases(bell_pair) -> None:
    qc, a, _ = bell_pair
    assert qc.inspect.entanglement_entropy([a], base=np.e) == pytest.approx(np.log(2))


def test_the_schmidt_spectrum_of_a_product_state_has_one_nonzero_value(qc: Circuit) -> None:
    a, _ = qc.alloc_many(2)
    H(a)
    spectrum = qc.inspect.schmidt_spectrum([a])
    assert spectrum == pytest.approx([1.0, 0.0], abs=1e-12)


def test_the_schmidt_spectrum_of_a_bell_pair_is_evenly_split(bell_pair) -> None:
    qc, a, _ = bell_pair
    assert qc.inspect.schmidt_spectrum([a]) == pytest.approx([1 / np.sqrt(2)] * 2)


def test_mutual_information_is_two_bits_for_a_bell_pair(bell_pair) -> None:
    """S(A) + S(B) - S(AB) = 1 + 1 - 0. Counts classical and quantum correlation together."""
    qc, a, b = bell_pair
    assert qc.inspect.mutual_information([a], [b]) == pytest.approx(2.0, abs=1e-12)


def test_uncorrelated_qubits_share_no_mutual_information(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    H(a)
    H(b)
    assert qc.inspect.mutual_information([a], [b]) == pytest.approx(0.0, abs=1e-12)


def test_a_ghz_state_is_entangled_across_every_cut(qc: Circuit) -> None:
    reg = qc.register(3)
    H(reg[0])
    CNOT(reg[0], reg[1])
    CNOT(reg[0], reg[2])

    for cut in ([reg[0]], [reg[1]], [reg[2]], [reg[0], reg[1]]):
        assert qc.inspect.entanglement_entropy(cut) == pytest.approx(1.0, abs=1e-12)


# ---- assert_zero --------------------------------------------------------------


def test_assert_zero_passes_for_qubits_still_in_the_zero_state(qc: Circuit) -> None:
    reg = qc.register(2)
    qc.inspect.assert_zero(reg)


def test_assert_zero_rejects_a_qubit_in_superposition(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    with pytest.raises(DirtyAncillaError, match="not in"):
        qc.inspect.assert_zero([a])


def test_the_dirty_ancilla_message_explains_why_it_matters(qc: Circuit) -> None:
    """The error is a teaching surface: leftover entanglement is a *record*, and a
    recorded branch can no longer interfere."""
    a = qc.alloc()
    X(a)
    with pytest.raises(DirtyAncillaError, match="interfere"):
        qc.inspect.assert_zero([a])


def test_assert_zero_rejects_an_entangled_qubit_even_though_it_could_read_zero(
    bell_pair,
) -> None:
    qc, a, _ = bell_pair
    with pytest.raises(DirtyAncillaError):
        qc.inspect.assert_zero([a])


# ---- Bloch vectors ------------------------------------------------------------


def test_the_zero_state_sits_at_the_north_pole(qc: Circuit) -> None:
    a = qc.alloc()
    assert qc.inspect.bloch_vector(a) == pytest.approx((0, 0, 1))


def test_the_one_state_sits_at_the_south_pole(qc: Circuit) -> None:
    a = qc.alloc()
    X(a)
    assert qc.inspect.bloch_vector(a) == pytest.approx((0, 0, -1))


def test_plus_and_minus_sit_on_opposite_sides_of_the_equator(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    H(a)
    X(b)
    H(b)
    assert qc.inspect.bloch_vector(a) == pytest.approx((1, 0, 0))
    assert qc.inspect.bloch_vector(b) == pytest.approx((-1, 0, 0))


def test_s_on_plus_rotates_a_quarter_turn_around_the_equator(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    S(a)
    assert qc.inspect.bloch_vector(a) == pytest.approx((0, 1, 0), abs=1e-12)


def test_an_entangled_qubit_sits_at_the_centre_of_the_sphere(bell_pair) -> None:
    """Length zero: nothing whatsoever can be known about it locally."""
    qc, a, _ = bell_pair
    assert qc.inspect.bloch_vector(a) == pytest.approx((0, 0, 0), abs=1e-12)


# ---- observables --------------------------------------------------------------


def test_expectation_of_z_is_plus_one_in_the_zero_state(qc: Circuit) -> None:
    qc.alloc()
    assert qc.inspect.expectation("Z") == pytest.approx(1.0)


def test_expectation_of_z_is_zero_in_a_superposition(qc: Circuit) -> None:
    """Half the time +1, half the time -1: the average is 0."""
    a = qc.alloc()
    H(a)
    assert qc.inspect.expectation("Z") == pytest.approx(0.0, abs=1e-15)
    assert qc.inspect.expectation("X") == pytest.approx(1.0)


def test_expectation_of_y_detects_a_complex_phase(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    S(a)
    assert qc.inspect.expectation("Y") == pytest.approx(1.0)


def test_a_bell_pair_is_perfectly_correlated_in_z_and_in_x(bell_pair) -> None:
    qc, _, _ = bell_pair
    assert qc.inspect.expectation("ZZ") == pytest.approx(1.0)
    assert qc.inspect.expectation("XX") == pytest.approx(1.0)
    assert qc.inspect.expectation("YY") == pytest.approx(-1.0)


def test_identity_letters_ignore_their_qubit(bell_pair) -> None:
    qc, _, _ = bell_pair
    assert qc.inspect.expectation("ZI") == pytest.approx(0.0, abs=1e-15)
    assert qc.inspect.expectation("II") == pytest.approx(1.0)


def test_expectation_can_be_restricted_to_a_register(qc: Circuit) -> None:
    reg = qc.register(2)
    other = qc.alloc()
    X(other)
    assert qc.inspect.expectation("ZZ", reg) == pytest.approx(1.0)


def test_a_pauli_string_of_the_wrong_length_is_refused(bell_pair) -> None:
    qc, _, _ = bell_pair
    with pytest.raises(ValueError, match="one letter per qubit"):
        qc.inspect.expectation("Z")


def test_an_unknown_pauli_letter_is_refused(bell_pair) -> None:
    qc, _, _ = bell_pair
    with pytest.raises(ValueError, match="not a Pauli operator"):
        qc.inspect.expectation("ZQ")


# ---- comparisons --------------------------------------------------------------


def test_a_state_has_fidelity_one_with_itself(bell_pair) -> None:
    qc, _, _ = bell_pair
    assert qc.inspect.fidelity(qc.inspect.state_vector()) == pytest.approx(1.0)


def test_orthogonal_states_have_zero_fidelity(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    minus = np.array([1, -1]) / np.sqrt(2)
    assert qc.inspect.fidelity(minus) == pytest.approx(0.0, abs=1e-15)


def test_orthogonal_states_have_zero_overlap(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    Z(a)  # |->
    plus = np.array([1, 1]) / np.sqrt(2)
    assert qc.inspect.overlap(plus) == pytest.approx(0.0, abs=1e-15)


def test_overlap_keeps_the_phase_that_fidelity_squares_away(qc: Circuit) -> None:
    """A full 2*pi rotation multiplies the state by -1. No measurement can see that,
    and fidelity cannot either — but the phase is exactly what decides how two states
    add when they interfere, so overlap reports it."""
    from qsim.gates import Rz

    a = qc.alloc()
    H(a)
    plus = qc.inspect.state_vector()

    Rz(a, theta=2 * np.pi)  # now -|+>

    assert qc.inspect.overlap(plus) == pytest.approx(-1.0)
    assert qc.inspect.fidelity(plus) == pytest.approx(1.0)


def test_overlap_accepts_a_state_in_tensor_form(bell_pair) -> None:
    qc, _, _ = bell_pair
    assert qc.inspect.overlap(qc.inspect.state_tensor()) == pytest.approx(1.0)


# ---- Dirac notation -----------------------------------------------------------


def test_a_bell_pair_prints_as_two_terms(bell_pair) -> None:
    qc, _, _ = bell_pair
    assert str(qc.inspect.ket()) == "0.707|00⟩ + 0.707|11⟩"


def test_a_negative_amplitude_prints_as_a_subtraction(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    Z(a)
    assert str(qc.inspect.ket()) == "0.707|0⟩ - 0.707|1⟩"


def test_a_leading_negative_amplitude_keeps_its_sign(qc: Circuit) -> None:
    from qsim.gates import Ry

    a = qc.alloc()
    # Ry(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>. At theta = 1.75*pi the cosine
    # is both negative and the larger of the two, so the leading term carries a minus.
    Ry(a, theta=1.75 * np.pi)
    assert str(qc.inspect.ket()) == "-0.924|0⟩ + 0.383|1⟩"


def test_a_complex_amplitude_prints_in_parentheses(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    T(a)
    assert str(qc.inspect.ket()) == "0.707|0⟩ + (0.500+0.500i)|1⟩"


def test_a_negative_imaginary_part_prints_with_a_minus(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    from qsim.gates import GATES

    GATES["T†"](a)  # type: ignore[operator]
    assert str(qc.inspect.ket()) == "0.707|0⟩ + (0.500-0.500i)|1⟩"


def test_the_display_truncates_and_says_how_many_terms_are_hidden(qc: Circuit) -> None:
    reg = qc.register(3)
    for q in reg:
        H(q)
    text = str(qc.inspect.ket(max_terms=2))
    assert text.endswith("… (6 more terms)")


def test_one_hidden_term_is_described_in_the_singular(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    H(a)
    H(b)
    assert str(qc.inspect.ket(max_terms=3)).endswith("… (1 more term)")


def test_a_circuit_with_no_qubits_has_one_basis_state_with_no_name(qc: Circuit) -> None:
    """Zero qubits span a 1-dimensional space: one basis state, and no bits to label it."""
    assert str(qc.inspect.ket()) == "1.000|⟩"


def test_a_state_with_no_significant_amplitudes_prints_as_zero() -> None:
    ket = type(Circuit().inspect.ket())([], 0)
    assert str(ket) == "0"


def test_ket_repr_matches_its_string_form(bell_pair) -> None:
    qc, _, _ = bell_pair
    assert repr(qc.inspect.ket()) == str(qc.inspect.ket())


def test_ket_renders_as_latex_in_a_notebook(bell_pair) -> None:
    qc, _, _ = bell_pair
    latex = qc.inspect.ket()._repr_latex_()
    assert latex.startswith("$") and latex.endswith("$")
    assert r"\left|" in latex


def test_truncated_latex_uses_a_typeset_ellipsis(qc: Circuit) -> None:
    reg = qc.register(3)
    for q in reg:
        H(q)
    assert r"\dots" in qc.inspect.ket(max_terms=1)._repr_latex_()


# ---- bras ---------------------------------------------------------------------


def test_a_bra_is_written_the_other_way_round(bell_pair) -> None:
    qc, _, _ = bell_pair
    assert str(qc.inspect.bra()) == "0.707⟨00| + 0.707⟨11|"


def test_a_bra_conjugates_every_amplitude(qc: Circuit) -> None:
    """The one reason to print a bra: the conjugation is visible."""
    a = qc.alloc()
    H(a)
    T(a)

    assert str(qc.inspect.ket()) == "0.707|0⟩ + (0.500+0.500i)|1⟩"
    assert str(qc.inspect.bra()) == "0.707⟨0| + (0.500-0.500i)⟨1|"


def test_bra_repr_matches_its_string_form(bell_pair) -> None:
    qc, _, _ = bell_pair
    assert repr(qc.inspect.bra()) == str(qc.inspect.bra())


def test_a_bra_renders_as_latex_too(bell_pair) -> None:
    qc, _, _ = bell_pair
    latex = qc.inspect.bra()._repr_latex_()
    assert r"\left\langle" in latex


def test_a_bra_meeting_its_own_ket_gives_one(qc: Circuit) -> None:
    """⟨ψ|ψ⟩ = 1 is the normalization condition, written in Dirac notation."""
    a = qc.alloc()
    H(a)
    Y(a)
    assert qc.inspect.overlap(qc.inspect.state_vector()) == pytest.approx(1.0)
