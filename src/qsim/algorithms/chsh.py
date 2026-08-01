"""The CHSH game — the empirical proof that entanglement is not classical correlation.

**The physical fact this module makes concrete:** there is a number you can measure that
no theory in which the particles agreed on their answers in advance can push above 2, and
quantum mechanics reaches 2√2 ≈ 2.828. Real experiments measure the quantum value. This is
the single most important demonstration in the library for someone learning what
entanglement actually is, and it won the 2022 Nobel Prize in Physics.

The game
--------
Alice and Bob are separated and cannot communicate. Each round, a referee hands Alice a
random question bit x and Bob a random question bit y. Each must answer ±1 — call the
answers A and B. They win the round if

    A · B = +1   when x and y are not both 1,
    A · B = −1   when x = y = 1.

Playing the best possible *classical* strategy — including any amount of shared planning,
shared random numbers, or shared physical objects prepared together beforehand — they win
at most 75% of rounds. Sharing an entangled pair, they win about 85.4%. Nothing is
signalled, nothing travels faster than light, and no local story about pre-agreed answers
reproduces that number.

The score S, and where the number 2√2 comes from
-------------------------------------------------
Instead of counting wins we track the four **correlators** E(α, β): the average of A · B
when Alice measures at angle α and Bob at angle β. The CHSH score combines the four
question combinations into one number:

    S = E(a, b) + E(a, b′) + E(a′, b) − E(a′, b′)

For a Bell pair, E(α, β) = cos(α − β). With the optimal settings
(a, a′, b, b′) = (0, π/2, π/4, −π/4), the first three correlators are each cos(π/4) = 1/√2
and the last is cos(3π/4) = −1/√2, which is subtracted. So

    S = 1/√2 + 1/√2 + 1/√2 − (−1/√2) = 4/√2 = 2√2 ≈ 2.828.

Measuring at an angle
---------------------
This is the one new physical idea in this module. So far every measurement has asked "is
this qubit |0⟩ or |1⟩?" — a measurement along the z-axis of the Bloch sphere. Alice and Bob
need to ask questions along *tilted* axes: the observable cos θ · Z + sin θ · X, which is
the z-axis rotated by θ towards x.

There is no separate machinery for that, and there does not need to be. Rotating the state
and then measuring z is the same experiment as leaving the state alone and measuring along
a tilted axis — the difference is only whether you turn the apparatus or the sample.
Concretely, applying ``Ry(q, theta=-θ)`` and then measuring Z measures cos θ · Z + sin θ · X.
Every "measure at angle θ" below is that one line.

Cross-reference: CHSH shows the classical picture *failing*. Decoherence (Phase 2.5) and
einselection show where the classical picture *comes from* — the two halves of the same
story, approached from opposite ends.
"""

from itertools import product

import numpy as np

from qsim.circuit import Circuit
from qsim.gates import CNOT, H, Ry

#: (a, a′, b, b′) — Alice's two measurement angles then Bob's two. These maximize S.
OPTIMAL_SETTINGS: tuple[float, float, float, float] = (0.0, np.pi / 2, np.pi / 4, -np.pi / 4)

#: The largest S any local hidden-variable theory can produce. Computed, not asserted, by
#: :func:`classical_bound`.
CLASSICAL_LIMIT = 2.0

#: The largest S quantum mechanics allows (Tsirelson's bound), reached by a Bell pair at
#: :data:`OPTIMAL_SETTINGS`.
QUANTUM_LIMIT = 2.0 * np.sqrt(2.0)


def _bell_pair(seed: int | None = None) -> tuple[Circuit, tuple]:
    """A fresh circuit holding (|00⟩ + |11⟩)/√2, with handles to Alice's and Bob's halves."""
    qc = Circuit(name="chsh", seed=seed)
    alice, bob = qc.alloc_many(2)
    H(alice)
    CNOT(alice, bob)
    return qc, (alice, bob)


def _rotate_to_measure_at(qubit, angle: float) -> None:
    """Turn the apparatus: after this, measuring Z measures cos θ·Z + sin θ·X."""
    # Ry(-θ) followed by a z-measurement is exactly a measurement along the axis tilted
    # by θ from z towards x. See the module docstring — the minus sign is because
    # rotating the *state* by -θ is the same as rotating the *apparatus* by +θ.
    Ry(qubit, theta=-angle)


def chsh_expectation(angle_a: float, angle_b: float) -> float:
    """The correlator E(α, β): the average of Alice's answer times Bob's.

    Both answers are ±1, so E runs from −1 (they always disagree) through 0 (no
    correlation) to +1 (they always agree). For a Bell pair the answer is cos(α − β):
    perfectly correlated when the two measure along the same axis, perfectly
    anti-correlated at right angles, and smoothly in between.

    Computed analytically here, by asking the Inspector for ⟨ZZ⟩ — which no real
    experiment could do. :func:`chsh_sampled` does it the honest way.
    """
    qc, (alice, bob) = _bell_pair()
    _rotate_to_measure_at(alice, angle_a)
    _rotate_to_measure_at(bob, angle_b)
    # With both bases rotated, ⟨ZZ⟩ is the correlation between the two tilted
    # measurements. Each Z has eigenvalues ±1, so their product averages to E.
    return qc.inspect.expectation("ZZ")


def chsh_S(settings: tuple[float, float, float, float] = OPTIMAL_SETTINGS) -> float:
    """The CHSH score S for the four measurement angles (a, a′, b, b′).

    At the default settings this returns 2√2 ≈ 2.828, comfortably above the classical
    ceiling of 2 that :func:`classical_bound` computes.
    """
    a, a_prime, b, b_prime = settings
    return (
        chsh_expectation(a, b)
        + chsh_expectation(a, b_prime)
        + chsh_expectation(a_prime, b)
        - chsh_expectation(a_prime, b_prime)
    )


def _sampled_correlator(angle_a: float, angle_b: float, shots: int, seed: int | None) -> float:
    """Estimate E(α, β) from ``shots`` measurement outcomes rather than from the amplitudes."""
    qc, (alice, bob) = _bell_pair(seed)
    _rotate_to_measure_at(alice, angle_a)
    _rotate_to_measure_at(bob, angle_b)

    # inspect.sample() draws independent outcomes from exactly the distribution that
    # rerunning the experiment `shots` times would produce — a real lab would have to
    # rebuild the pair for every single shot, since measuring destroys it.
    counts = qc.inspect.sample(shots)

    # Outcome 0 means the answer +1 and outcome 1 means −1, so the product A·B is +1 when
    # the two bits agree and −1 when they differ. E is the average of that product.
    agree = counts["00"] + counts["11"]
    disagree = counts["01"] + counts["10"]
    return (agree - disagree) / shots


def chsh_sampled(
    shots: int,
    *,
    seed: int | None = None,
    settings: tuple[float, float, float, float] = OPTIMAL_SETTINGS,
) -> float:
    """The CHSH score estimated from actual measurements — what an experiment reports.

    Each of the four correlators is estimated from ``shots`` measurement outcomes, so the
    answer carries statistical noise of order 1/√shots and lands *near* 2√2 rather than on
    it. That is the honest situation: a real experiment never measures an amplitude, only
    a pile of ±1 answers, and the violation has to be visible through the noise.
    """
    a, a_prime, b, b_prime = settings
    # The four settings are four separate experiments, so they get four separate streams.
    # Drawing the seeds from one generator says that more plainly than seed, seed+1,
    # seed+2, seed+3 would -- though either is sound: NumPy runs a seed through
    # SeedSequence precisely so that neighbouring seeds still give independent streams.
    if seed is None:
        seeds: list[int | None] = [None, None, None, None]
    else:
        seeds = [int(s) for s in np.random.default_rng(seed).integers(0, 2**32, size=4)]

    return (
        _sampled_correlator(a, b, shots, seeds[0])
        + _sampled_correlator(a, b_prime, shots, seeds[1])
        + _sampled_correlator(a_prime, b, shots, seeds[2])
        - _sampled_correlator(a_prime, b_prime, shots, seeds[3])
    )


def classical_bound() -> float:
    """The best S reachable by any local hidden-variable theory. Returns exactly 2.0.

    "Local hidden variable" is the pair-of-gloves picture: the two particles carry
    pre-agreed answers, fixed when they were created, and measuring merely reads one off.
    Any such theory — however elaborate, with any amount of shared randomness — is a
    probabilistic mixture of *deterministic* strategies, and a deterministic strategy is
    just four fixed answers: A and A′ for Alice's two questions, B and B′ for Bob's. Since
    S is linear in those probabilities, the best mixture can do no better than the best
    single strategy, so brute-forcing all 16 settles it.

    The algebra says the same thing in one line:

        S = A·B + A·B′ + A′·B − A′·B′ = A(B + B′) + A′(B − B′)

    B and B′ are each ±1, so either they are equal — making (B − B′) zero and (B + B′)
    equal to ±2 — or they differ, making (B + B′) zero and (B − B′) equal to ±2. Either
    way exactly one term survives and it is at most 2 in magnitude. No cleverness escapes
    it, which is what makes the quantum value of 2.828 so hard to explain away.
    """
    return float(
        max(
            a * b + a * b_prime + a_prime * b - a_prime * b_prime
            for a, a_prime, b, b_prime in product((-1, 1), repeat=4)
        )
    )


def classical_strategies() -> list[tuple[tuple[int, int, int, int], int]]:
    """Every deterministic local strategy paired with its score: 16 rows, best S = 2.

    Returned so the notebook can *show* the exhaustive search rather than assert its
    result. Each row is ((A, A′, B, B′), S).
    """
    return [
        ((a, a_prime, b, b_prime), a * b + a * b_prime + a_prime * b - a_prime * b_prime)
        for a, a_prime, b, b_prime in product((-1, 1), repeat=4)
    ]
