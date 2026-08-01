"""Acceptance tests TB1–TB3 from design doc §9 — the entanglement demonstrations.

Tolerances are part of the specification and must not be loosened to make a test pass.
"""

import numpy as np
import pytest

from qsim.algorithms.chsh import (
    OPTIMAL_SETTINGS,
    chsh_expectation,
    chsh_S,
    chsh_sampled,
    classical_bound,
    classical_strategies,
)
from qsim.algorithms.teleportation import superdense_send, teleport
from qsim.gates import H, Rx, Ry, Rz, T, X

# ---- TB1: CHSH ---------------------------------------------------------------


def test_tb1_a_bell_pair_scores_two_root_two_on_the_chsh_test() -> None:
    """TB1: S = 2√2 ≈ 2.828 at the optimal measurement angles.

    No local hidden-variable model reaches S > 2; quantum mechanics does. This is the
    experimentally confirmed (Nobel Prize in Physics, 2022) sense in which entanglement
    is not classical correlation — the two qubits did not agree on their answers in
    advance, because no set of pre-agreed answers can score above 2.
    """
    assert abs(chsh_S() - 2 * np.sqrt(2)) < 1e-12


def test_tb1_no_local_strategy_beats_an_s_of_two() -> None:
    """TB1: the classical bound is *computed*, by brute-forcing all 16 deterministic
    local strategies, not asserted."""
    assert classical_bound() == 2.0


def test_tb1_all_sixteen_deterministic_strategies_are_searched() -> None:
    """TB1: every combination of four fixed ±1 answers, and none of them exceeds 2."""
    strategies = classical_strategies()

    assert len(strategies) == 16
    assert all(abs(score) <= 2 for _, score in strategies)
    assert max(score for _, score in strategies) == 2


def test_tb1_sampled_measurements_still_break_the_classical_bound() -> None:
    """TB1: S > 2.7 from 100,000 actual measurement outcomes per setting.

    The point of doing it this way is that a real experiment never sees an amplitude —
    only a pile of ±1 answers. The violation has to survive the statistical noise, and it
    does, by a wide margin.
    """
    assert chsh_sampled(shots=100_000, seed=1234) > 2.7


def test_tb1_the_correlator_follows_the_cosine_of_the_angle_difference() -> None:
    """TB1: E(α, β) = cos(α − β) — perfect agreement along the same axis, perfect
    disagreement at right angles, and a smooth curve in between."""
    for alpha, beta in [(0.0, 0.0), (0.0, np.pi / 4), (np.pi / 3, -np.pi / 5), (0.2, 1.9)]:
        assert chsh_expectation(alpha, beta) == pytest.approx(np.cos(alpha - beta), abs=1e-12)


def test_tb1_measuring_along_the_same_axis_always_agrees() -> None:
    """TB1: the Bell pair's defining property, stated as a correlator."""
    assert chsh_expectation(0.0, 0.0) == pytest.approx(1.0)
    assert chsh_expectation(np.pi / 2, np.pi / 2) == pytest.approx(1.0)


def test_tb1_measuring_at_right_angles_gives_no_correlation_at_all() -> None:
    assert chsh_expectation(0.0, np.pi / 2) == pytest.approx(0.0, abs=1e-15)


def test_tb1_badly_chosen_angles_score_below_the_classical_bound() -> None:
    """TB1: the violation is not automatic — it takes the right measurement settings.
    Aligning every angle makes the quantum strategy no better than a classical one."""
    assert chsh_S((0.0, 0.0, 0.0, 0.0)) == pytest.approx(2.0)


def test_tb1_the_optimal_settings_are_actually_optimal() -> None:
    """TB1: sweeping an offset across Bob's angles peaks exactly at the default."""
    a, a_prime, b, b_prime = OPTIMAL_SETTINGS
    scores = [
        chsh_S((a, a_prime, b + offset, b_prime + offset))
        for offset in np.linspace(-0.6, 0.6, 25)
    ]
    assert max(scores) == pytest.approx(chsh_S(), abs=1e-12)


def test_tb1_sampling_without_a_seed_still_violates_the_bound() -> None:
    """TB1: the violation is a property of the state, not of a lucky seed."""
    assert chsh_sampled(shots=20_000) > 2.5


# ---- TB2: teleportation ------------------------------------------------------


def random_state_prep(seed: int):
    """Build a state_prep callable placing a qubit somewhere random on the Bloch sphere."""
    rng = np.random.default_rng(seed)
    theta = float(rng.uniform(0, np.pi))
    phi = float(rng.uniform(0, 2 * np.pi))

    def prep(q) -> None:
        Ry(q, theta=theta)
        Rz(q, theta=phi)

    return prep


def test_tb2_a_teleported_state_arrives_perfectly_intact() -> None:
    """TB2: fidelity 1 to within 1e-12, for states scattered over the Bloch sphere."""
    seeds = np.random.default_rng(0).integers(0, 2**32, size=25)
    for seed in seeds:
        result = teleport(random_state_prep(int(seed)), seed=int(seed))
        assert abs(result.fidelity - 1.0) < 1e-12


def test_tb2_teleportation_destroys_the_original() -> None:
    """TB2: after the protocol Alice's qubits sit at the poles of the Bloch sphere —
    definite classical bits, holding no trace of the message. This is the no-cloning
    theorem's fingerprint: you end with one copy, exactly as you began."""
    seeds = np.random.default_rng(1).integers(0, 2**32, size=20)
    for seed in seeds:
        result = teleport(random_state_prep(int(seed)), seed=int(seed))

        for z in result.source_bloch_z:
            assert abs(abs(z) - 1.0) < 1e-12
        assert result.source_bits == (result.m1, result.m2)


def test_tb2_all_four_measurement_branches_occur_and_are_corrected() -> None:
    """TB2: Alice's two bits are uniformly random, so all four (m1, m2) outcomes happen —
    and each needs a different fixup from Bob. Every branch must reach fidelity 1."""
    branches: dict[tuple[int, int], str] = {}
    seeds = np.random.default_rng(2).integers(0, 2**32, size=40)
    for seed in seeds:
        result = teleport(random_state_prep(7), seed=int(seed))
        branches[(result.m1, result.m2)] = result.corrections
        assert abs(result.fidelity - 1.0) < 1e-12

    assert set(branches) == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert branches == {(0, 0): "", (0, 1): "X", (1, 0): "Z", (1, 1): "X then Z"}


def test_tb2_teleportation_works_for_basis_states_too() -> None:
    """TB2: edge cases — |0⟩ needs no preparation at all, and |1⟩ is a single X."""
    for prep in (lambda q: None, X, H, lambda q: Rx(q, theta=np.pi / 3)):
        result = teleport(prep, seed=5)
        assert abs(result.fidelity - 1.0) < 1e-12


def test_tb2_teleporting_a_state_with_a_complex_phase_preserves_the_phase() -> None:
    """TB2: fidelity would still be high if only the magnitudes survived, so check a
    state whose whole content is in its phase."""

    def prep(q) -> None:
        H(q)
        T(q)

    result = teleport(prep, seed=11)
    assert abs(result.fidelity - 1.0) < 1e-12


def test_tb2_the_same_seed_reproduces_the_same_run() -> None:
    first = teleport(random_state_prep(3), seed=42)
    second = teleport(random_state_prep(3), seed=42)
    assert (first.m1, first.m2, first.corrections) == (second.m1, second.m2, second.corrections)


def test_tb2_teleportation_runs_without_a_seed() -> None:
    result = teleport(random_state_prep(4))
    assert abs(result.fidelity - 1.0) < 1e-12


# ---- TB3: superdense coding --------------------------------------------------


@pytest.mark.parametrize("message", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_tb3_every_two_bit_message_survives_the_trip(message: tuple[int, int]) -> None:
    """TB3: all four messages decode exactly, carried by a single transmitted qubit.

    Two classical bits arrive having handed over one qubit — because Bob already held the
    other half of an entangled pair, prepared before there was any message to send.
    """
    assert superdense_send(message) == message


def test_tb3_decoding_is_deterministic_across_seeds() -> None:
    """TB3: "with probability 1" means every seed, not most of them."""
    for seed in range(25):
        for message in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            assert superdense_send(message, seed=seed) == message


def test_tb3_a_message_that_is_not_two_bits_is_refused() -> None:
    with pytest.raises(ValueError, match="two classical bits"):
        superdense_send((0, 2))
