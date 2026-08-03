"""The QFT, phase estimation, and the Phase 3 papercut fixes.

Read this as a usage guide. The first half is the Fourier machinery — what the circuit
is made of and what it does to the simplest possible inputs. The second half covers the
small library-wide fixes that shipped alongside it (phase plan §0, P1–P8), each one
tested where its behaviour is easiest to state.
"""

from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pytest

import qsim
from qsim import Circuit, Register, viz
from qsim.algorithms.phase_estimation import (
    phase_estimation,
    semiclassical_phase_estimation,
)
from qsim.algorithms.qft import iqft, qft
from qsim.circuit import Op, Qubit
from qsim.errors import DirtyAncillaError, QsimError
from qsim.gates import CNOT, H, Phase, Ry, X

# ---- the QFT circuit ----------------------------------------------------------


def test_the_qft_of_all_zeros_is_the_uniform_superposition() -> None:
    """|00...0> is the number 0, and 0 has no periodicity for the transform to find —
    so every frequency is present in equal measure and every amplitude comes out
    1/sqrt(2^n), with no phases at all. The Fourier transform of nothing is everything,
    flat."""
    n = 4
    qc = Circuit(seed=1)
    reg = qc.register(n)

    qft(reg)

    amps = qc.inspect.state_vector()
    assert amps == pytest.approx(np.full(2**n, 1 / np.sqrt(2**n)))


def test_the_qft_of_a_single_qubit_is_exactly_the_hadamard_gate() -> None:
    """With one qubit there are no controlled rotations left to apply and no bits to
    reverse, so the whole construction collapses to its first line: H. The Hadamard
    *is* the two-point Fourier transform."""
    for prep in (lambda q: None, X, H):
        by_qft = Circuit(seed=1)
        reg = by_qft.register(1)
        prep(reg[0])
        qft(reg)

        by_hand = Circuit(seed=1)
        q = by_hand.alloc()
        prep(q)
        H(q)

        assert by_qft.inspect.state_vector() == pytest.approx(by_hand.inspect.state_vector())


def test_truncating_to_one_binary_place_leaves_only_the_hadamards() -> None:
    """``approx=1`` drops every controlled rotation, because the coarsest of them is
    already a half turn — two binary places. What is left is n independent H gates,
    which is a perfectly good unitary, just not the Fourier transform."""
    n = 4
    qc = Circuit(seed=1)
    reg = qc.register(n)

    qft(reg, approx=1, swap=False)

    assert qc.gate_counts() == {"H": n}


def test_the_qft_costs_n_hadamards_and_n_choose_two_controlled_phases() -> None:
    """The structural claim behind the algorithm's whole reputation: one H per qubit and
    one controlled rotation per *pair* of qubits, so about n²/2 gates in total — against
    the 2^n * n arithmetic operations a classical FFT of the same 2^n amplitudes needs.

    The SWAP network adds n/2 more gates, which is why it is optional: on hardware you
    can often just relabel the wires instead.
    """
    for n in (1, 2, 5, 8):
        qc = Circuit(seed=1)
        reg = qc.register(n)
        qft(reg)

        counts = qc.gate_counts()
        assert counts.get("H", 0) == n
        assert counts.get("CPhase", 0) == n * (n - 1) // 2
        assert counts.get("SWAP", 0) == n // 2


def test_the_swap_network_can_be_switched_off() -> None:
    """``swap=False`` emits no SWAP gates at all — the point being that you can then see
    the bit reversal for yourself (T11 does exactly that)."""
    qc = Circuit(seed=1)
    reg = qc.register(5)
    qft(reg, swap=False)

    assert "SWAP" not in qc.gate_counts()


def test_the_qft_on_an_empty_register_refuses_clearly() -> None:
    """The edge case, and the error a slice that selected nothing produces."""
    qc = Circuit(seed=1)
    reg = qc.register(3)

    with pytest.raises(QsimError, match="at least one qubit"):
        qft(reg[1:1])


def test_the_inverse_transform_is_the_forward_one_run_backwards() -> None:
    """``iqft`` is built as ``qft.adjoint()``, so its recorded ops are the forward
    circuit reversed and daggered: swaps first, then the rotations undone, then the
    Hadamards. Reading the two op lists side by side is the clearest possible statement
    of what "inverse" means for a circuit."""
    forward = Circuit(seed=1)
    qft(forward.register(3))

    backward = Circuit(seed=1)
    iqft(backward.register(3))

    assert [op.name for op in backward.history] == [op.name for op in forward.history][::-1]
    # Every angle is negated; SWAP and H are their own inverses so theirs are empty.
    forward_angles = [op.params for op in forward.history if op.params]
    backward_angles = [op.params for op in backward.history if op.params]
    assert backward_angles == [(-theta,) for (theta,) in reversed(forward_angles)]


def test_the_inverse_transform_is_recorded_as_a_block_of_its_own() -> None:
    """``iqft`` stays a named public function even though it delegates: the tape shows
    both the name you called and the derived block that did the work."""
    qc = Circuit(seed=1)
    iqft(qc.register(2))

    assert qc.block_counts() == {"iqft": 1, "qft†": 1}


# ---- phase estimation ---------------------------------------------------------


def phase_unitary(phi: float) -> qsim.Block:
    """A one-qubit block with eigenvalue exp(2*pi*i*phi) on |1> and 1 on |0>."""

    @qsim.gate
    def u(reg: Register) -> None:
        Phase(reg[0], theta=2 * np.pi * phi)

    return u


def test_phase_estimation_needs_somewhere_to_put_the_answer() -> None:
    """The size of ``out`` is the precision being asked for, so an empty ``out`` is a
    request for zero digits."""
    qc = Circuit(seed=1)
    target = qc.register(1)
    out = qc.register(2)
    target.encode(1)

    with pytest.raises(QsimError, match="at least one output qubit"):
        phase_estimation(phase_unitary(0.25), target, out[0:0])


def test_phase_estimation_applies_the_unitary_two_to_the_t_minus_one_times() -> None:
    """Where the cost lives. Three output qubits mean 4 + 2 + 1 = 7 applications of U,
    each one a controlled block. The repetition is honest — no shortcut powers — because
    that exponential is the number any real use of the algorithm has to engineer away.

    Counted through ``block_counts()``, which reports the derived block ``C-u`` by name.
    """
    qc = Circuit(seed=1)
    target = qc.register(1)
    out = qc.register(3)
    target.encode(1)

    phase_estimation(phase_unitary(0.25), target, out)

    assert qc.block_counts()["C-u"] == 2**3 - 1
    # Growing the register by one doubles the work: 15 applications for 4 digits.
    bigger = Circuit(seed=1)
    bigger_target = bigger.register(1)
    bigger_target.encode(1)
    phase_estimation(phase_unitary(0.25), bigger_target, bigger.register(4))
    assert bigger.block_counts()["C-u"] == 2**4 - 1


def test_a_target_that_is_not_an_eigenstate_gives_a_superposition_of_answers() -> None:
    """The case the argument is named ``target`` rather than ``eigenstate`` for.

    ``Phase`` has eigenvalue 1 on |0> (phase 0) and e^{2*pi*i*phi} on |1> (phase phi).
    Feed it |+>, an equal mix of the two, and the phase register comes out an equal mix
    of the two answers — each entangled with the eigenvector that produced it. That is
    exactly the mechanism Shor's algorithm runs on, since it cannot prepare an
    eigenstate of its multiplication map without already knowing the answer.
    """
    qc = Circuit(seed=1)
    target = qc.register(1, name="v")
    out = qc.register(3, name="out")
    H(target[0])

    phase_estimation(phase_unitary(3 / 8), target, out)

    outcomes = qc.inspect.marginal(out)
    assert outcomes[0b000] == pytest.approx(0.5, abs=1e-9)
    assert outcomes[0b011] == pytest.approx(0.5, abs=1e-9)
    # And the answer is entangled with which eigenvector it came from: measuring the
    # register collapses the target onto the matching one.
    assert qc.inspect.entanglement_entropy(target) == pytest.approx(1.0, abs=1e-9)


def test_the_semiclassical_version_asks_for_at_least_one_digit() -> None:
    qc = Circuit(seed=1)
    target = qc.register(1)
    target.encode(1)

    with pytest.raises(QsimError, match="at least 1"):
        semiclassical_phase_estimation(phase_unitary(0.25), target, 0)


def test_the_semiclassical_version_uses_exactly_one_extra_qubit() -> None:
    """The deliverable of the Griffiths–Niu construction: t digits of precision from
    *one* phase qubit, reused, instead of t of them.

    Counted with an ``on_op`` hook, so the number is the peak qubit count during the
    run rather than a claim about it. The coherent circuit for the same 8 digits is
    measured alongside for contrast.
    """
    coherent = Circuit(seed=1)
    coherent_target = coherent.register(1)
    coherent_out = coherent.register(8)
    coherent_target.encode(1)
    phase_estimation(phase_unitary(0.25), coherent_target, coherent_out)
    assert coherent.n_qubits == 1 + 8

    qc = Circuit(seed=1)
    target = qc.register(1, name="v")
    target.encode(1)

    peak: list[int] = []
    handle = qc.on_op(lambda op, circuit: peak.append(circuit.n_qubits))
    semiclassical_phase_estimation(phase_unitary(0.25), target, 8)
    handle.remove()

    assert max(peak) == 1 + 1
    # The phase qubit was borrowed with qc.ancilla(1), so it has been verified clean
    # and handed back — after 8 rounds of use the circuit is the size it started.
    assert qc.n_qubits == 1


def test_the_semiclassical_version_reads_a_single_digit_phase() -> None:
    """t = 1 is the smallest case and the one with no classical feedback at all: with
    no digits measured yet there is nothing to correct for, so the loop's correction
    step never runs. phi = 1/2 gives the digit 1."""
    qc = Circuit(seed=1)
    target = qc.register(1)
    target.encode(1)

    assert semiclassical_phase_estimation(phase_unitary(0.5), target, 1) == 1


def test_the_semiclassical_version_leaves_the_target_untouched() -> None:
    """Phase kickback again: after 2^t - 1 applications of U the eigenstate is still
    exactly |1>, and the circuit is back to holding it alone."""
    qc = Circuit(seed=1)
    target = qc.register(1)
    target.encode(1)

    semiclassical_phase_estimation(phase_unitary(3 / 8), target, 3)

    assert qc.inspect.marginal(target) == pytest.approx([0.0, 1.0], abs=1e-12)


# ==============================================================================
# Papercut fixes (phase plan §0). Small library-wide changes that shipped with
# this phase; each is tested here for the behaviour it states.
# ==============================================================================


# ---- P1: inspect.marginal -----------------------------------------------------


def test_the_marginal_of_one_qubit_is_the_number_probabilities_zero_is_mistaken_for() -> None:
    """P1: the trap this method exists to close.

    ``probabilities()[0]`` reads like "the chance qubit 0 comes out 0". It is the chance
    that *every* qubit comes out 0 — here 1/8, not 1/2. Three demo notebooks were
    written against the wrong one before ``marginal`` existed.
    """
    qc = Circuit(seed=1)
    a, b, c = qc.alloc_many(3)
    for q in (a, b, c):
        H(q)

    assert qc.inspect.probabilities()[0] == pytest.approx(1 / 8)
    assert qc.inspect.marginal([a]) == pytest.approx([0.5, 0.5])


def test_a_marginal_of_half_a_bell_pair_is_a_fair_coin(bell_pair) -> None:
    """P1: each half of a Bell pair is individually random, which is what "no state of
    its own" looks like as a distribution."""
    qc, a, b = bell_pair

    assert qc.inspect.marginal([a]) == pytest.approx([0.5, 0.5])
    assert qc.inspect.marginal([b]) == pytest.approx([0.5, 0.5])
    # ... while the *pair* is perfectly correlated: |01> and |10> never happen.
    assert qc.inspect.marginal([a, b]) == pytest.approx([0.5, 0.0, 0.0, 0.5])


def test_the_order_of_the_subset_is_the_bit_order_of_the_result() -> None:
    """P1: ``marginal([b, a])`` is ``marginal([a, b])`` with its two axes swapped —
    ``subset[0]`` is the most significant bit of the index, as everywhere in qsim."""
    qc = Circuit(seed=1)
    a, b = qc.alloc_many(2)
    Ry(a, theta=np.pi / 3)  # P(a = 1) = sin²(pi/6) = 0.25
    H(b)  # P(b = 1) = 0.5

    assert qc.inspect.marginal([a, b]) == pytest.approx([0.375, 0.375, 0.125, 0.125])
    assert qc.inspect.marginal([b, a]) == pytest.approx([0.375, 0.125, 0.375, 0.125])


def test_a_marginal_over_every_qubit_is_just_the_probabilities() -> None:
    """P1: with nothing left to sum over, the marginal is the full distribution."""
    qc = Circuit(seed=1)
    a, b, c = qc.alloc_many(3)
    H(a)
    CNOT(a, b)
    Ry(c, theta=0.7)

    assert qc.inspect.marginal([a, b, c]) == pytest.approx(qc.inspect.probabilities())


def test_environment_qubits_appear_in_a_marginal_only_when_asked_for() -> None:
    """P1: ``marginal`` takes the subset you name and assumes nothing about which
    qubits count as "the system" — unlike ``system_density_matrix``, which uses the
    environment marking."""
    from qsim.decoherence import dephasing_coupling

    qc = Circuit(seed=1)
    q = qc.alloc("q")
    e = qc.environment_qubit()
    H(q)
    dephasing_coupling(q, e, theta=np.pi)

    assert qc.inspect.marginal([q]) == pytest.approx([0.5, 0.5])
    assert len(qc.inspect.marginal([q, e])) == 4


def test_asking_for_the_same_qubit_twice_in_a_marginal_is_refused() -> None:
    """P1: a qubit has one outcome, not two, so it cannot occupy two index positions."""
    qc = Circuit(seed=1)
    a, _ = qc.alloc_many(2)

    with pytest.raises(ValueError, match="listed twice"):
        qc.inspect.marginal([a, a])


# ---- P2: DirtyAncillaError grammar --------------------------------------------


def test_one_dirty_ancilla_is_described_in_the_singular() -> None:
    """P2: "anc0 are not in |0>" was distracting to read at exactly the moment the
    reader is trying to absorb what uncomputation is for."""
    qc = Circuit(seed=1)
    a = qc.alloc("a")
    H(a)

    with pytest.raises(DirtyAncillaError) as raised:
        with qc.ancilla(1) as scratch:
            CNOT(a, scratch[0])

    message = str(raised.value)
    assert "anc0 is not in |0>" in message
    assert "dirtied this qubit" in message


def test_two_dirty_ancillas_are_described_in_the_plural() -> None:
    qc = Circuit(seed=1)
    a = qc.alloc("a")
    H(a)

    with pytest.raises(DirtyAncillaError) as raised:
        with qc.ancilla(2) as scratch:
            CNOT(a, scratch[0])
            CNOT(a, scratch[1])

    message = str(raised.value)
    assert "anc0, anc1 are not in |0>" in message
    assert "dirtied these qubits" in message


# ---- P3: no negative zero in a Bloch vector -----------------------------------


def test_the_north_pole_has_no_negative_zeros_in_it() -> None:
    """P3: |0> sits at (0, 0, 1). The y-component is computed as -2 * Im(rho01), and
    -2 * 0.0 is negative zero in IEEE arithmetic — numerically identical to 0.0 and
    needlessly alarming printed next to a component reading 1.0."""
    qc = Circuit(seed=1)
    q = qc.alloc()

    components = qc.inspect.bloch_vector(q)

    assert components == (0.0, 0.0, 1.0)
    assert not any(repr(component).startswith("-0") for component in components)


# ---- P4: Qubit.circuit and Register.circuit -----------------------------------


def test_a_qubit_handle_knows_which_circuit_it_belongs_to() -> None:
    """P4: bookkeeping, not physics — it is how ``H(q)`` has always found its circuit.
    Making it public means a block can open a scope without a redundant ``qc``
    parameter."""
    qc = Circuit(seed=1)
    q = qc.alloc()

    assert q.circuit is qc


def test_a_block_can_open_a_control_scope_through_its_own_qubits() -> None:
    """P4: the friction this fixes. Before, this block's signature had to start with a
    circuit its own arguments already knew."""

    @qsim.gate
    def flip_if(control: Qubit, target: Qubit) -> None:
        with control.circuit.control(control):
            X(target)

    qc = Circuit(seed=1)
    c, t = qc.alloc_many(2)
    H(c)
    flip_if(c, t)

    assert qc.inspect.probabilities() == pytest.approx([0.5, 0.0, 0.0, 0.5])


def test_a_register_knows_its_circuit_too() -> None:
    qc = Circuit(seed=1)
    reg = qc.register(3)

    assert reg.circuit is qc
    assert reg[1:].circuit is qc


def test_an_empty_register_cannot_say_which_circuit_it_belongs_to() -> None:
    """P4: a Register is its handles and nothing else, so with no handles there is
    nothing to ask."""
    qc = Circuit(seed=1)
    reg = qc.register(3)

    with pytest.raises(QsimError, match="empty register"):
        _ = reg[1:1].circuit


# ---- P5: viz.bloch takes the qubit alone --------------------------------------


def test_the_bloch_plot_resolves_its_circuit_from_the_qubit() -> None:
    """P5: ``viz.bloch(q)``, matching how gates are called. ``H(q)`` is never written
    ``H(qc, q)``, so neither is this."""
    qc = Circuit(seed=1)
    q = qc.alloc()
    H(q)

    fig = viz.bloch(q)
    assert "length 1.00" in fig.axes[0].get_title()
    plt.close(fig)


# ---- P6: environment_qubit ----------------------------------------------------


def test_a_single_environment_qubit_can_be_allocated_directly() -> None:
    """P6: ``qc.environment_qubit()`` next to ``qc.environment(n)``, the same split as
    ``alloc()`` next to ``alloc_many(n)`` — every call site used to write
    ``qc.environment(1)[0]``."""
    qc = Circuit(seed=1)
    q = qc.alloc("q")
    e = qc.environment_qubit()

    assert isinstance(e, Qubit)
    assert e.name == "E"
    assert list(qc.environment_qubits) == [e]
    assert list(qc.system_qubits) == [q]


def test_an_environment_qubit_can_be_named() -> None:
    qc = Circuit(seed=1)
    bath = qc.environment_qubit(name="bath")

    assert bath.name == "bath"
    assert list(qc.environment_qubits) == [bath]


def test_a_single_environment_qubit_decoheres_exactly_like_a_register_of_one() -> None:
    """P6: the helper changes the spelling and nothing else."""
    from qsim.decoherence import dephasing_coupling

    singular = Circuit(seed=1)
    q1 = singular.alloc("q")
    H(q1)
    dephasing_coupling(q1, singular.environment_qubit(), theta=np.pi / 3)

    plural = Circuit(seed=1)
    q2 = plural.alloc("q")
    H(q2)
    dephasing_coupling(q2, plural.environment(1)[0], theta=np.pi / 3)

    assert singular.inspect.coherence(q1) == pytest.approx(plural.inspect.coherence(q2))


# ---- P7: within stamps both halves of the conjugation --------------------------


def test_a_conjugation_by_a_named_block_is_counted_symmetrically() -> None:
    """P7: ``within(bell, a, b)`` reports ``bell`` going in and ``bell†`` coming out.

    Before this, the tally showed the basis change once and its undo not at all, which
    made a conjugation look lopsided in exactly the place a reader goes to check that it
    is not. The naming matches ``block.adjoint()``, which has produced ``bell†`` since
    Phase 2.75.
    """

    @qsim.gate
    def bell(x: Qubit, y: Qubit) -> None:
        H(x)
        CNOT(x, y)

    qc = Circuit(seed=1)
    a, b, t = qc.alloc_many(3)

    with qsim.within(bell, a, b):
        X(t)

    assert qc.block_counts() == {"bell": 1, "bell†": 1}
    # The undo half's ops carry the derived name, so the two halves match on the tape.
    assert [op.block for op in qc.history] == ["bell", "bell", "", "bell†", "bell†"]


def test_a_conjugation_by_a_bare_gate_is_counted_but_does_not_restamp_its_ops() -> None:
    """P7: a gate has a name, so both halves are counted. Its ops keep whichever block
    is being recorded *around* the scope, because that is where they really came from —
    overwriting one half alone would misreport the other."""
    qc = Circuit(seed=1)
    q = qc.alloc()

    with qsim.within(H, q):
        X(q)

    assert qc.block_counts() == {"H": 1, "H†": 1}
    assert [op.block for op in qc.history] == ["", "", ""]


def test_an_anonymous_conjugation_is_not_counted_at_all() -> None:
    """P7's boundary: qsim will not invent a name for something the language did not
    name. A lambda or plain function stays out of the tally — ``<lambda>`` in a block
    count helps nobody. Wrap it in a ``def`` and decorate it if you want it counted."""
    qc = Circuit(seed=1)
    q = qc.alloc()

    with qsim.within(lambda target: H(target), q):
        X(q)

    assert qc.block_counts() == {}
    assert [op.name for op in qc.history] == ["H", "X", "H"]


def test_stamping_does_not_change_what_a_conjugation_does() -> None:
    """P7 is a bookkeeping change: the state is untouched by it."""

    @qsim.gate
    def basis(x: Qubit) -> None:
        H(x)

    scoped = Circuit(seed=1)
    q = scoped.alloc()
    with qsim.within(basis, q):
        qsim.Z(q)

    by_hand = Circuit(seed=1)
    p = by_hand.alloc()
    H(p)
    qsim.Z(p)
    H(p)

    assert scoped.inspect.state_vector() == pytest.approx(by_hand.inspect.state_vector())


# ---- P8: sampling is deliberately deterministic --------------------------------


def test_the_same_seed_and_the_same_calls_give_identical_samples() -> None:
    """P8: documented rather than merely true. A ``sample()`` in a notebook is a number
    you can write prose about."""

    def build() -> Counter[str]:
        qc = Circuit(seed=7)
        a, b = qc.alloc_many(2)
        H(a)
        CNOT(a, b)
        return qc.inspect.sample(shots=200)

    assert build() == build()


def test_sampling_does_not_disturb_the_stream_that_measurement_draws_from() -> None:
    """P8: ``inspect.sample`` is a question asked *about* a circuit, not an event in it,
    so it draws from its own generator. Adding a sample() line halfway down a seeded
    notebook cannot rewrite the measurements below it."""

    def measure_five(with_sampling: bool) -> list[int]:
        qc = Circuit(seed=11)
        reg = qc.register(5)
        for q in reg:
            H(q)
        if with_sampling:
            qc.inspect.sample(shots=50)
        return [qc.measure(q) for q in reg]

    assert measure_five(with_sampling=True) == measure_five(with_sampling=False)


def test_a_hook_still_sees_nothing_when_a_circuit_is_sampled() -> None:
    """P8, the other half of "sampling leaves no trace": it is not an operation, so it
    never reaches the tape."""
    qc = Circuit(seed=3)
    q = qc.alloc()
    H(q)

    seen: list[Op] = []
    qc.on_op(lambda op, circuit: seen.append(op))
    qc.inspect.sample(shots=10)

    assert seen == []
    assert len(qc.history) == 1
