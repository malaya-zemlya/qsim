"""Acceptance tests T11–T15 from design doc §9 — the QFT and phase estimation.

Tolerances are part of the specification and must not be loosened to make a test pass.

T11 is where this project's bit convention is pinned down against an outside authority
(``np.fft``) and documented by example. If exactly one test in this file is worth
reading slowly, it is that one.
"""

from collections import Counter

import numpy as np
import pytest

import qsim
from qsim import Circuit
from qsim.algorithms.phase_estimation import (
    phase_estimation,
    semiclassical_phase_estimation,
)
from qsim.algorithms.qft import iqft, qft
from qsim.gates import Phase

# ---- T11: the QFT is the DFT, and here is the bit convention ------------------


def test_t11_the_qft_matches_numpys_inverse_fft(random_state) -> None:
    """T11: for a random 5-qubit state, ``qft`` reproduces ``np.fft.ifft`` scaled by
    sqrt(2^n), to 1e-12. The physical fact: the QFT is not *like* a Fourier transform,
    it *is* one — the same linear map on the same 2^n numbers.

    Three conventions have to line up for that sentence to be checkable at all, and
    getting any of them wrong is the classic quantum-simulator bug. Spelled out:

    **1. Which sign of the exponent.** qsim's QFT (design doc §8.1) is

        QFT|j> = (1/sqrt N) sum_k exp(+2*pi*i*j*k/N) |k>

    with a *plus* in the exponent. NumPy puts the minus sign in ``np.fft.fft`` and the
    plus in ``np.fft.ifft``. So the QFT corresponds to numpy's **inverse** transform,
    which reads backwards but is only a naming convention: signal processing calls
    "forward" the direction that goes from samples to spectrum, and quantum computing
    inherited the opposite convention from physics.

    **2. Which normalization.** ``np.fft.ifft`` carries a factor 1/N, chosen so that
    ``ifft(fft(x)) == x``. The QFT must be unitary — it is a physical operation, so it
    has to preserve total probability — which forces the symmetric factor 1/sqrt N.
    Multiplying numpy's output by sqrt(N) converts one to the other.

    **3. Which bit is which.** ``inspect.state_vector()`` flattens the (2,)*n state
    tensor in C order, so axis 0 varies slowest — i.e. qubit 0 is the most significant
    bit, the convention stated in ``state.py`` and used everywhere in qsim. That means
    index i of the flat vector is the amplitude of the basis state whose bits read as
    the integer i, which is exactly the indexing ``np.fft`` assumes. **No reindexing is
    needed anywhere in this test.** The reversal the circuit produces internally is
    already undone by ``qft``'s SWAP network; the test below shows what happens without
    it.
    """
    n = 5
    psi = random_state(n)

    qc = Circuit(n, seed=1)
    # Install a generic state directly. There is no gate sequence that would make the
    # point better, and tests may reach into privates (see tests/CLAUDE.md).
    qc._psi = psi.astype(np.complex128)

    qft(qc.qubits)

    expected = np.fft.ifft(psi.reshape(-1)) * np.sqrt(2**n)
    assert np.abs(qc.inspect.state_vector() - expected).max() < 1e-12


def test_t11_without_the_swap_network_the_output_is_bit_reversed(random_state) -> None:
    """T11: ``qft(reg, swap=False)`` gives the right transform with the qubit order
    reversed, and *reversing the qubit order is a transpose of the state tensor*.

    This is the whole bit-ordering story in one assertion. The state is an array of
    shape (2,)*n whose axes are the qubits, so "read the register backwards" is
    literally "walk the axes backwards" — ``np.transpose`` with the axis list reversed.
    Flattening the transposed tensor then gives the same numbers a SWAP network would
    have produced, because a SWAP network *is* a permutation of the axes, done with
    gates instead of with an index list.

    Keeping ``swap=True`` as the default is therefore a convenience, not a correction:
    nothing is wrong with the unswapped state, it is just written in the other order.
    """
    n = 5
    psi = random_state(n)

    qc = Circuit(n, seed=1)
    qc._psi = psi.astype(np.complex128)

    qft(qc.qubits, swap=False)

    # Reverse the axis order — qubit n-1 becomes axis 0, and so on — then flatten. The
    # C-order flatten reads the *new* axis 0 as the most significant bit, so this is
    # "re-read the same amplitudes with the register held the other way round".
    reversed_axes = np.transpose(qc.inspect.state_tensor(), list(reversed(range(n))))
    undone = reversed_axes.reshape(-1)

    expected = np.fft.ifft(psi.reshape(-1)) * np.sqrt(2**n)
    assert np.abs(undone - expected).max() < 1e-12


def test_t11_a_basis_state_transforms_into_a_flat_pattern_of_phases() -> None:
    """T11, the hand-checkable case: QFT|j> has 2^n amplitudes all of magnitude
    1/sqrt(2^n), differing only in phase — and the k-th phase is exp(2*pi*i*j*k/2^n).

    A definite number spreads into every basis state at once. The information about
    which number it was survives entirely in the *phases*, which no measurement of this
    state can see. That is the catch in the module docstring, made concrete.
    """
    n = 3
    for j in range(2**n):
        qc = Circuit(seed=1)
        reg = qc.register(n)
        reg.encode(j)
        qft(reg)

        amps = qc.inspect.state_vector()
        expected = np.exp(2j * np.pi * j * np.arange(2**n) / 2**n) / np.sqrt(2**n)
        assert np.abs(amps - expected).max() < 1e-12
        assert np.abs(np.abs(amps) - 1 / np.sqrt(2**n)).max() < 1e-12


# ---- T12: the inverse really is the inverse -----------------------------------


@pytest.mark.parametrize("swap", [True, False])
def test_t12_qft_followed_by_iqft_is_the_identity(random_state, swap: bool) -> None:
    """T12: on a random 6-qubit state, ``qft`` then ``iqft`` returns the original state
    to 1e-13, with the SWAP network on and off.

    ``iqft`` is built as ``qft.adjoint()`` — the recorded body replayed backwards with
    every gate inverted — so this test is also a check on the block algebra: a
    hundred-gate subroutine inverts by the same rule a single gate does.
    """
    n = 6
    psi = random_state(n)

    qc = Circuit(n, seed=2)
    qc._psi = psi.astype(np.complex128)

    qft(qc.qubits, swap=swap)
    iqft(qc.qubits, swap=swap)

    assert np.abs(qc.inspect.state_tensor() - psi).max() < 1e-13


def test_t12_the_qft_alone_is_not_the_identity(random_state) -> None:
    """T12: a guard against the round trip passing because nothing happened."""
    n = 6
    psi = random_state(n)

    qc = Circuit(n, seed=2)
    qc._psi = psi.astype(np.complex128)
    qft(qc.qubits)

    assert qc.inspect.fidelity(psi.reshape(-1)) < 0.5


# ---- T13: how much the approximate QFT gives up -------------------------------


def approximate_qft_fidelities(t: int, levels: range, seed: int) -> list[float]:
    """Fidelity of ``qft(approx=m)`` against the exact ``qft``, for each m in ``levels``."""
    rng = np.random.default_rng(seed)
    psi = rng.normal(size=(2,) * t) + 1j * rng.normal(size=(2,) * t)
    psi /= np.linalg.norm(psi)

    def transformed(approx: int | None) -> Circuit:
        qc = Circuit(t, seed=0)
        # Same input state every time, so the only thing varying is the truncation.
        qc._psi = psi.astype(np.complex128)
        qft(qc.qubits, approx=approx)
        return qc

    exact = transformed(None).inspect.state_vector()
    return [transformed(m).inspect.fidelity(exact) for m in levels]


def test_t13_the_approximate_qft_improves_monotonically_with_the_truncation_level() -> None:
    """T13: for t = 12 qubits and truncation levels m = 2..10, the fidelity against the
    exact QFT never decreases as m grows, and m = 8 already exceeds 0.999.

    The physical fact: the QFT's smallest rotations are doing almost no work. At t = 12
    the finest exact rotation is by 2*pi/4096 — a fifteenth of a degree — and dropping
    every rotation below 1/256 of a turn costs about two parts in ten thousand of
    fidelity. Real hardware cannot apply a fifteenth of a degree accurately anyway, so
    the approximate QFT is not a compromise forced on us: the circuit was carrying gates
    that were never earning their keep.

    Fidelity here is |<exact|approx>|², so 0.999 means the two output states are
    indistinguishable by any single measurement 99.9% of the time.
    """
    fidelities = approximate_qft_fidelities(t=12, levels=range(2, 11), seed=11)

    # Monotone non-decreasing: each extra binary place of rotation can only help.
    for coarser, finer in zip(fidelities, fidelities[1:], strict=False):
        assert finer >= coarser

    m8 = fidelities[8 - 2]
    assert m8 > 0.999


def test_t13_truncating_hard_enough_really_does_damage() -> None:
    """T13: the other end of the same curve — a guard that the ``approx`` argument is
    connected to anything at all. Keeping only two binary places of rotation destroys
    the transform (fidelity well under 0.1 at t = 12)."""
    fidelities = approximate_qft_fidelities(t=12, levels=range(2, 4), seed=11)
    assert fidelities[0] < 0.1


# ---- T14: phase estimation, the exactly representable case --------------------


def phase_unitary(phi: float) -> qsim.Block:
    """A one-qubit block whose eigenvalue on |1> is exp(2*pi*i*phi).

    ``Phase(theta)`` multiplies the |1> amplitude by e^{i*theta} and leaves |0> alone,
    so |1> is an eigenvector with eigenvalue e^{i*theta} and |0> is an eigenvector with
    eigenvalue 1. Setting theta = 2*pi*phi makes the first of those the phase we are
    asking phase estimation to find.
    """

    @qsim.gate
    def u(reg: qsim.Register) -> None:
        Phase(reg[0], theta=2 * np.pi * phi)

    return u


def test_t14_phase_estimation_reads_an_exact_three_bit_phase_with_certainty() -> None:
    """T14: with phi = 3/8 — exactly 0.011 in binary — a 3-qubit phase register holds
    the outcome 0b011 with probability greater than 0.999.

    "Greater than 0.999" understates it: when phi is exactly representable in t bits the
    inverse QFT lands on one basis state and the probability is 1 to floating-point
    precision. The interference is perfect, because every one of the 2^t paths through
    the circuit arrives at the right answer in phase and at every wrong answer out of
    phase. The probability is read from ``inspect``, not sampled, so this is the exact
    number rather than an estimate of it.
    """
    qc = Circuit(seed=1)
    target = qc.register(1, name="v")
    out = qc.register(3, name="out")

    # Put the target in |1>, the eigenvector whose eigenvalue we want to read.
    target.encode(1)

    phase_estimation(phase_unitary(3 / 8), target, out)

    # marginal() over just the phase register: the probability of each 3-bit outcome,
    # MSB first. (probabilities()[3] would be a different and wrong number — it is the
    # probability of the whole 4-qubit state |0011>.)
    outcomes = qc.inspect.marginal(out)
    assert outcomes[0b011] > 0.999
    assert qc.measure_all(out) == 0b011


def test_t14_the_target_is_returned_untouched_by_phase_kickback() -> None:
    """T14: the eigenstate is a catalyst. After phase estimation, the target is still
    exactly |1>, unentangled with the register that now holds the answer — the whole
    effect of applying U 2^t - 1 times has landed on the qubits that only controlled it.
    That is what "phase kickback" names."""
    qc = Circuit(seed=1)
    target = qc.register(1, name="v")
    out = qc.register(3, name="out")
    target.encode(1)

    phase_estimation(phase_unitary(3 / 8), target, out)

    assert qc.inspect.is_product(target)
    assert qc.inspect.marginal(target) == pytest.approx([0.0, 1.0], abs=1e-12)


def test_t14_a_phase_of_zero_reads_back_as_zero() -> None:
    """T14, the edge case: U = identity has phase 0, and the register must say so.

    Worth having because it is the one answer a broken circuit is most likely to give by
    accident — so seeing it appear *correctly* here, alongside 0b011 above, is what
    makes the pair of results meaningful.
    """
    qc = Circuit(seed=1)
    target = qc.register(1)
    out = qc.register(3)
    target.encode(1)

    phase_estimation(phase_unitary(0.0), target, out)

    assert qc.inspect.marginal(out)[0] > 0.999


# ---- T15: the semiclassical version agrees ------------------------------------

#: A phase that is *not* exactly representable in binary — the interesting case, and the
#: one every real use of phase estimation is in. 0.3 in binary is 0.0100110011... , so
#: no finite register can hold it and the outcome distribution is genuinely spread.
NON_REPRESENTABLE_PHI = 0.3

#: Digits of precision for T15. Small enough that 500 shots resolve the distribution
#: well (see the tolerance note in the test), large enough that the answer is spread
#: over several outcomes rather than concentrated on one.
T15_DIGITS = 4

#: Shots, fixed by the design doc's "over 500 seeded shots".
T15_SHOTS = 500


def coherent_distribution(phi: float, t: int) -> np.ndarray:
    """The exact outcome distribution of the coherent circuit, read from ``inspect``."""
    qc = Circuit(seed=0)
    target = qc.register(1, name="v")
    out = qc.register(t, name="out")
    target.encode(1)
    phase_estimation(phase_unitary(phi), target, out)
    return qc.inspect.marginal(out)


def semiclassical_counts(phi: float, t: int, shots: int) -> Counter[int]:
    """Run the one-qubit version ``shots`` times, each on its own seeded circuit."""
    counts: Counter[int] = Counter()
    for seed in range(shots):
        qc = Circuit(seed=seed)
        target = qc.register(1, name="v")
        target.encode(1)
        counts[semiclassical_phase_estimation(phase_unitary(phi), target, t)] += 1
    return counts


def test_t15_the_semiclassical_version_reproduces_the_coherent_distribution() -> None:
    """T15: 500 seeded runs of ``semiclassical_phase_estimation`` match the coherent
    circuit's exact distribution to a total-variation distance below 0.05.

    The physical fact — the **deferred measurement principle**. The coherent circuit
    keeps all t qubits in superposition until the very end; the semiclassical one
    measures each digit as soon as it is available and feeds the result back as an
    ordinary Python number controlling an ordinary rotation. Those look like different
    experiments, and quantum mechanics is famously unforgiving about when you measure.
    They are nonetheless provably the same, and this test is the empirical form of the
    proof.

    **The statistic.** Total-variation distance between two distributions is
    TVD = (1/2) * sum_i |p_i - q_i|: the largest difference in probability the two can
    assign to any one event. It is 0 for identical distributions and 1 for
    distributions with disjoint support.

    **Why 0.05 is the right bound for 500 shots.** Even two *identical* distributions
    disagree at this sample size, because 500 draws only pin each probability down to
    about sqrt(p(1-p)/500) — a scale of 1/sqrt(500) ~ 0.045. Summing that noise over the
    handful of outcomes carrying real probability here gives an expected TVD near 0.026,
    so 0.05 sits roughly two standard deviations out: tight enough that a genuinely
    different distribution would fail it, loose enough that agreement passes. (A real
    disagreement would not be subtle. Getting the correction angles' sign wrong, for
    instance, gives TVD near 1.)
    """
    exact = coherent_distribution(NON_REPRESENTABLE_PHI, T15_DIGITS)
    counts = semiclassical_counts(NON_REPRESENTABLE_PHI, T15_DIGITS, T15_SHOTS)
    sampled = np.array([counts[y] / T15_SHOTS for y in range(2**T15_DIGITS)])

    tvd = 0.5 * float(np.abs(exact - sampled).sum())
    assert tvd < 0.05


def test_t15_a_non_representable_phase_gives_a_peaked_but_spread_distribution() -> None:
    """T15: phi = 0.3 with t = 4 digits. 2^4 * 0.3 = 4.8, so no outcome is exactly
    right and the distribution piles up on the two neighbours 4 and 5, with a tail.

    This is the normal situation, and it is why Shor's algorithm needs continued
    fractions to turn the measured integer back into a fraction: what comes out is the
    nearest representable phase, not the phase.
    """
    exact = coherent_distribution(NON_REPRESENTABLE_PHI, T15_DIGITS)

    assert int(np.argmax(exact)) == 5  # 5/16 = 0.3125, the closest 4-bit phase
    assert exact[5] > 0.8
    assert exact[4] > exact[6]  # 4.8 is nearer 5 than 4, and nearer 4 than 6
    assert exact.max() < 0.99  # ... but no outcome is certain, unlike T14


def test_t15_the_semiclassical_version_agrees_exactly_when_the_phase_is_representable() -> None:
    """T15: when the answer is certain, "the same distribution" becomes "the same
    answer", every time. phi = 3/8 with 3 digits gives 0b011 from T14's coherent circuit
    and 0b011 from every seeded semiclassical run — a sharper check of the same claim,
    with no statistics in the way."""
    for seed in range(20):
        qc = Circuit(seed=seed)
        target = qc.register(1)
        target.encode(1)
        assert semiclassical_phase_estimation(phase_unitary(3 / 8), target, 3) == 0b011
