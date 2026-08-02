"""Acceptance tests TD1–TD7 from `qsim-design.md` §9.

These pin down the claim the whole decoherence phase rests on: that unitary coupling to
extra qubits, plus a decision not to look at them, *is* noise — not an imitation of it.

Tolerances are part of the specification and must not be loosened to make a test pass.
"""

import numpy as np
import pytest

from qsim import Circuit
from qsim.circuit import Qubit
from qsim.decoherence import (
    amplitude_damping_coupling,
    dephasing_coupling,
    depolarizing_coupling,
    pointer_coupling,
)
from qsim.gates import H, Ry, Rz

# Single-qubit matrices, written out so the tests check the library against an
# independent construction rather than against itself.
I2 = np.eye(2, dtype=complex)
PAULI_X = np.array([[0, 1], [1, 0]], dtype=complex)
PAULI_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
PAULI_Z = np.array([[1, 0], [0, -1]], dtype=complex)
HADAMARD = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def apply_kraus(kraus: list[np.ndarray], rho: np.ndarray) -> np.ndarray:
    """The textbook form of a noise channel: rho -> sum_k K_k rho K_k†."""
    out = np.zeros_like(rho)
    for K in kraus:
        out = out + K @ rho @ K.conj().T
    return out


def prepare_random_qubit(qc: Circuit, q: Qubit, seed: int) -> None:
    """Put ``q`` into an arbitrary pure state — two rotations reach the whole sphere."""
    rng = np.random.default_rng(seed)
    Ry(q, theta=float(rng.uniform(0, np.pi)))
    Rz(q, theta=float(rng.uniform(0, 2 * np.pi)))


# ---- TD1: coherence decays with coupling strength ----------------------------


@pytest.mark.parametrize("theta", [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi])
def test_td1_the_bloch_vector_shrinks_as_the_cosine_of_half_the_coupling_angle(
    theta: float,
) -> None:
    """TD1: for |+>, the Bloch x-component after dephasing is exactly cos(θ/2).

    The qubit was pushed by nothing and measured by no one. Its Bloch vector shrinks
    because another qubit now holds a partial record of which branch it is in, and a
    recorded branch can no longer interfere with the others.
    """
    qc = Circuit(seed=1)
    q = qc.alloc("q")
    env = qc.environment(1)
    H(q)
    dephasing_coupling(q, env[0], theta=theta)

    x, y, z = qc.inspect.bloch_vector(q)
    assert x == pytest.approx(np.cos(theta / 2), abs=1e-12)
    assert y == pytest.approx(0.0, abs=1e-12)
    assert z == pytest.approx(0.0, abs=1e-12)


def test_td1_a_perfect_record_leaves_the_bloch_vector_at_the_origin() -> None:
    """TD1: at θ = π the vector is the origin — the maximally mixed state.

    Nothing whatsoever can be learned from this qubit alone. Every measurement of it,
    in every basis, gives 50/50. That is what "maximally mixed" means, and it is the
    single-qubit shadow of a state that is still perfectly pure globally.
    """
    qc = Circuit(seed=1)
    q = qc.alloc("q")
    env = qc.environment(1)
    H(q)
    dephasing_coupling(q, env[0], theta=np.pi)

    assert np.linalg.norm(qc.inspect.bloch_vector(q)) == pytest.approx(0.0, abs=1e-12)
    # ... while the global state is untouched in purity: still a unit vector, still pure.
    assert qc.inspect.norm() == pytest.approx(1.0, abs=1e-12)
    assert qc.inspect.entanglement_entropy(list(qc.qubits)) == pytest.approx(0.0, abs=1e-12)


# ---- TD2: interference is destroyed ------------------------------------------


@pytest.mark.parametrize("theta", [0.0, np.pi / 6, np.pi / 3, np.pi / 2, 2.4, np.pi])
def test_td2_interference_visibility_follows_the_cosine_of_half_the_coupling_angle(
    theta: float,
) -> None:
    """TD2: H, couple, H — the visibility P(0) − P(1) is cos(θ/2).

    This *is* the double-slit experiment. The two H gates are the beam splitter that
    opens two paths and the screen where they recombine; the environment qubit is a
    detector at the slits. The fringes fade exactly as fast as the detector becomes
    able to say which slit was taken — no faster, and not at all if it cannot.
    """
    qc = Circuit(seed=2)
    q = qc.alloc("q")
    env = qc.environment(1)
    H(q)
    dephasing_coupling(q, env[0], theta=theta)
    H(q)

    rho = qc.inspect.system_density_matrix()
    visibility = float((rho[0, 0] - rho[1, 1]).real)
    assert visibility == pytest.approx(np.cos(theta / 2), abs=1e-12)


def test_td2_with_no_environment_watching_the_outcome_is_certain() -> None:
    """TD2: at θ = 0 the photon lands in |0> with probability 1 — full fringes."""
    qc = Circuit(seed=2)
    q = qc.alloc("q")
    env = qc.environment(1)
    H(q)
    dephasing_coupling(q, env[0], theta=0.0)
    H(q)

    assert qc.inspect.probabilities()[0] == pytest.approx(1.0, abs=1e-12)


def test_td2_with_a_perfect_record_the_outcome_is_a_coin_flip() -> None:
    """TD2: at θ = π the interference is gone and the two outcomes are equally likely.

    Note what did *not* happen: no one measured the environment, and no random number
    was drawn. Merely making the which-path information *available* is enough.
    """
    qc = Circuit(seed=2)
    q = qc.alloc("q")
    env = qc.environment(1)
    H(q)
    dephasing_coupling(q, env[0], theta=np.pi)
    H(q)

    # The system's own distribution, not the joint one over qubit-and-environment.
    populations = np.real(np.diag(qc.inspect.system_density_matrix()))
    assert populations == pytest.approx([0.5, 0.5], abs=1e-12)


# ---- TD3: the quantum eraser -------------------------------------------------


@pytest.mark.parametrize("erase_with", ["scope", "method"])
def test_td3_uncomputing_the_coupling_restores_coherence_exactly(erase_with: str) -> None:
    """TD3: decoherence is reversible if you kept the environment.

    Couple at θ = π — a perfect record, coherence zero, one full bit of entropy — and
    then simply run the interaction backwards. The coherence returns to 0.5, the
    entropy to zero, and the interference to full visibility, all to 1e-12.

    Nothing was repaired, because nothing had broken. The information had moved into
    correlations with the environment and was brought back. This is the same phenomenon
    as T18's dirty ancilla: an ancilla you failed to uncompute *is* an environment, and
    uncomputation *is* erasure. A quantum computer's whole job is to keep this
    reversible until the very last step.
    """
    qc = Circuit(seed=3)
    q = qc.alloc("q")
    env = qc.environment(1)
    H(q)
    dephasing_coupling(q, env[0], theta=np.pi)

    assert qc.inspect.coherence(q) == pytest.approx(0.0, abs=1e-12)
    assert qc.inspect.system_entropy() == pytest.approx(1.0, abs=1e-12)

    # Both spellings of "run that block backwards" must work — the scope form and the
    # block's own adjoint. They cross-check Phase 2's capture of the theta argument.
    if erase_with == "scope":
        with qc.adjoint():
            dephasing_coupling(q, env[0], theta=np.pi)
    else:
        dephasing_coupling.adjoint()(q, env[0], theta=np.pi)

    assert qc.inspect.coherence(q) == pytest.approx(0.5, abs=1e-12)
    assert qc.inspect.system_entropy() == pytest.approx(0.0, abs=1e-12)

    # And the interference comes back: the final H returns |0> with certainty.
    H(q)
    assert qc.inspect.probabilities()[0] == pytest.approx(1.0, abs=1e-12)


# ---- TD4: populations versus coherences --------------------------------------


@pytest.mark.parametrize("theta", [np.pi / 5, np.pi / 2, np.pi])
def test_td4_dephasing_destroys_coherences_while_leaving_populations_untouched(
    theta: float,
) -> None:
    """TD4: dephasing shrinks |ρ₀₁| and moves the diagonal by less than 1e-12.

    The environment did not kick the qubit. Had it done so, the probability of finding
    1 would have changed — and it does not, at any coupling strength, including a
    perfect record. Decoherence is *leaked information*, not disturbance. That
    distinction is the difference between the popular account of quantum measurement
    and the actual one.
    """
    qc = Circuit(seed=4)
    q = qc.alloc("q")
    env = qc.environment(1)
    Ry(q, theta=0.7)  # an uneven superposition, so the populations are distinguishable
    before = qc.inspect.reduced_density_matrix([q]).copy()

    dephasing_coupling(q, env[0], theta=theta)
    after = qc.inspect.system_density_matrix()

    assert np.real(np.diag(after)) == pytest.approx(np.real(np.diag(before)), abs=1e-12)
    assert abs(after[0, 1]) == pytest.approx(abs(before[0, 1]) * np.cos(theta / 2), abs=1e-12)


def test_td4_amplitude_damping_changes_populations_too() -> None:
    """TD4: damping moves the diagonal, which is how it differs from dephasing.

    Energy actually leaves the qubit here — an excited atom decays and the excitation
    ends up in the environment — so the probability of finding 1 genuinely drops.
    """
    qc = Circuit(seed=4)
    q = qc.alloc("q")
    env = qc.environment(1)
    Ry(q, theta=0.7)
    before = qc.inspect.reduced_density_matrix([q]).copy()

    amplitude_damping_coupling(q, env[0], theta=np.pi / 3)
    after = qc.inspect.system_density_matrix()

    assert after[1, 1].real < before[1, 1].real - 1e-3
    # The qubit decays toward |0>, never away from it.
    assert after[0, 0].real > before[0, 0].real


# ---- TD5: einselection -------------------------------------------------------


def test_td5_coupling_through_z_makes_the_computational_basis_survive() -> None:
    """TD5: with basis="z", |0> comes through untouched and |+> is destroyed."""
    for prepare, expected_entropy in ((None, 0.0), (H, 1.0)):
        qc = Circuit(seed=5)
        q = qc.alloc("q")
        env = qc.environment(1)
        if prepare is not None:
            prepare(q)
        pointer_coupling(q, env[0], theta=np.pi, basis="z")
        assert qc.inspect.system_entropy() == pytest.approx(expected_entropy, abs=1e-12)


def test_td5_coupling_through_x_makes_the_plus_minus_basis_survive() -> None:
    """TD5: with basis="x" it is exactly reversed — |+> survives, |0> is destroyed.

    Same qubit, same states, same coupling strength; only the interaction changed. So
    nothing intrinsic to |0> makes it the "classical" state. The environment picks the
    surviving basis — that is einselection, and it is why the world we see is made of
    definite positions rather than definite momenta: the interactions that dominate
    around us couple through position.
    """
    for prepare, expected_entropy in ((None, 1.0), (H, 0.0)):
        qc = Circuit(seed=5)
        q = qc.alloc("q")
        env = qc.environment(1)
        if prepare is not None:
            prepare(q)
        pointer_coupling(q, env[0], theta=np.pi, basis="x")
        assert qc.inspect.system_entropy() == pytest.approx(expected_entropy, abs=1e-12)


def test_td5_z_coupling_preserves_populations_and_destroys_plus_minus_coherence() -> None:
    """TD5, stated on the density matrix: which coherences die depends on the basis.

    Read in the computational basis, a z-coupling leaves the diagonal alone and kills
    the off-diagonal. Read in the |+>/|-> basis — conjugate by H — it is the diagonal
    that is left alone and the off-diagonal that dies, for an x-coupling.
    """
    qc = Circuit(seed=5)
    q = qc.alloc("q")
    env = qc.environment(1)
    Ry(q, theta=0.9)
    before_z = qc.inspect.reduced_density_matrix([q]).copy()
    pointer_coupling(q, env[0], theta=np.pi, basis="z")
    after_z = qc.inspect.system_density_matrix()

    assert np.real(np.diag(after_z)) == pytest.approx(np.real(np.diag(before_z)), abs=1e-12)
    assert abs(after_z[0, 1]) == pytest.approx(0.0, abs=1e-12)

    qc = Circuit(seed=5)
    q = qc.alloc("q")
    env = qc.environment(1)
    Ry(q, theta=0.9)
    before_x = HADAMARD @ qc.inspect.reduced_density_matrix([q]) @ HADAMARD
    pointer_coupling(q, env[0], theta=np.pi, basis="x")
    after_x = HADAMARD @ qc.inspect.system_density_matrix() @ HADAMARD

    assert np.real(np.diag(after_x)) == pytest.approx(np.real(np.diag(before_x)), abs=1e-12)
    assert abs(after_x[0, 1]) == pytest.approx(0.0, abs=1e-12)


# ---- TD6: the dilations really are the named channels ------------------------
#
# The most important tests in this group. Each one builds the channel the honest way —
# couple unitarily, then trace — and compares against sum_k K_k rho K_k† computed here
# from the matrices written in the coupling's docstring. If these pass, the docstrings
# are true and the phase's central claim holds.


@pytest.mark.parametrize("theta", [0.0, 0.3, np.pi / 2, 2.2, np.pi])
@pytest.mark.parametrize("seed", [11, 12, 13])
def test_td6_dephasing_dilation_reproduces_its_kraus_channel(theta: float, seed: int) -> None:
    """TD6: tracing out our dephasing environment gives exactly the dephasing channel."""
    qc = Circuit()
    q = qc.alloc("q")
    env = qc.environment(1)
    prepare_random_qubit(qc, q, seed)
    rho = qc.inspect.reduced_density_matrix([q]).copy()

    dephasing_coupling(q, env[0], theta=theta)

    kraus = [
        np.diag([1.0, np.cos(theta / 2)]).astype(complex),
        np.diag([0.0, np.sin(theta / 2)]).astype(complex),
    ]
    assert qc.inspect.system_density_matrix() == pytest.approx(apply_kraus(kraus, rho), abs=1e-12)


@pytest.mark.parametrize("theta", [0.0, 0.3, np.pi / 2, 2.2, np.pi])
@pytest.mark.parametrize("seed", [11, 12, 13])
def test_td6_amplitude_damping_dilation_reproduces_its_kraus_channel(
    theta: float, seed: int
) -> None:
    """TD6: tracing out our damping environment gives exactly the damping channel."""
    qc = Circuit()
    q = qc.alloc("q")
    env = qc.environment(1)
    prepare_random_qubit(qc, q, seed)
    rho = qc.inspect.reduced_density_matrix([q]).copy()

    amplitude_damping_coupling(q, env[0], theta=theta)

    gamma = np.sin(theta / 2) ** 2
    kraus = [
        np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - gamma)]], dtype=complex),
        np.array([[0.0, np.sqrt(gamma)], [0.0, 0.0]], dtype=complex),
    ]
    assert qc.inspect.system_density_matrix() == pytest.approx(apply_kraus(kraus, rho), abs=1e-12)


@pytest.mark.parametrize("p", [0.0, 0.05, 0.25, 0.75, 1.0])
@pytest.mark.parametrize("seed", [11, 12, 13])
def test_td6_depolarizing_dilation_reproduces_its_kraus_channel(p: float, seed: int) -> None:
    """TD6: tracing out our two-qubit environment gives exactly the Pauli mixture.

    Worth noticing: the library never drew a random number. The "random Pauli" is a
    superposition over which Pauli was applied, and it only *looks* random because the
    four environment states recording the choice are orthogonal.
    """
    qc = Circuit()
    q = qc.alloc("q")
    env = qc.environment(2)
    prepare_random_qubit(qc, q, seed)
    rho = qc.inspect.reduced_density_matrix([q]).copy()

    depolarizing_coupling(q, env, p=p)

    kraus = [np.sqrt(1.0 - p) * I2] + [np.sqrt(p / 3.0) * P for P in (PAULI_X, PAULI_Y, PAULI_Z)]
    assert qc.inspect.system_density_matrix() == pytest.approx(apply_kraus(kraus, rho), abs=1e-12)


@pytest.mark.parametrize("basis,change", [("z", I2), ("x", HADAMARD)])
@pytest.mark.parametrize("theta", [0.4, np.pi])
def test_td6_pointer_coupling_is_its_dephasing_channel_conjugated(
    basis: str, change: np.ndarray, theta: float
) -> None:
    """TD6: pointer_coupling realizes dephasing conjugated into the chosen basis.

    Written out: U† K_k U for the dephasing Kraus operators, with U the rotation that
    carries the chosen basis onto the computational one.
    """
    qc = Circuit()
    q = qc.alloc("q")
    env = qc.environment(1)
    prepare_random_qubit(qc, q, 17)
    rho = qc.inspect.reduced_density_matrix([q]).copy()

    pointer_coupling(q, env[0], theta=theta, basis=basis)

    dephasing = [
        np.diag([1.0, np.cos(theta / 2)]).astype(complex),
        np.diag([0.0, np.sin(theta / 2)]).astype(complex),
    ]
    kraus = [change.conj().T @ K @ change for K in dephasing]
    assert qc.inspect.system_density_matrix() == pytest.approx(apply_kraus(kraus, rho), abs=1e-12)


# ---- TD7: the two halves of a pure state are equally mixed -------------------


@pytest.mark.parametrize("seed", range(20))
def test_td7_system_and_environment_always_have_equal_entropy(seed: int) -> None:
    """TD7: S(system) == S(environment) to 1e-10, for any coupling and any state.

    A genuinely surprising fact. The system is one qubit and the environment may be
    two, they were prepared differently and coupled asymmetrically — and their
    entropies agree exactly. It holds because the *global* state is pure: entropy here
    measures how entangled the two halves are with each other, and being entangled is
    something a pair does together, so both halves must report the same number.

    It is also a free audit of the partial trace: two independent traces over different
    axes have to land on the same value, and they do.
    """
    rng = np.random.default_rng(seed)
    qc = Circuit()
    q = qc.alloc("q")
    env = qc.environment(2)
    prepare_random_qubit(qc, q, seed)

    # A different random coupling each time, using both environment qubits.
    dephasing_coupling(q, env[0], theta=float(rng.uniform(0, np.pi)))
    amplitude_damping_coupling(q, env[1], theta=float(rng.uniform(0, np.pi)))

    assert qc.inspect.system_entropy() == pytest.approx(qc.inspect.environment_entropy(), abs=1e-10)


def test_td7_holds_for_the_two_qubit_depolarizing_environment_too() -> None:
    """TD7: still equal when the environment is larger than the system."""
    qc = Circuit()
    q = qc.alloc("q")
    env = qc.environment(2)
    prepare_random_qubit(qc, q, 99)
    depolarizing_coupling(q, env, p=0.4)

    assert qc.inspect.system_entropy() == pytest.approx(qc.inspect.environment_entropy(), abs=1e-10)
