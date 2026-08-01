"""Measurement: the Born rule, collapse, and why collapse acts on the joint state."""

import numpy as np
import pytest

from qsim import Circuit
from qsim.gates import CNOT, H, X

# ---- basic outcomes -----------------------------------------------------------


def test_measuring_a_definite_state_returns_that_value(qc: Circuit) -> None:
    a = qc.alloc()
    assert qc.measure(a) == 0

    b = qc.alloc()
    X(b)
    assert qc.measure(b) == 1


def test_measuring_a_superposition_gives_both_outcomes_at_the_expected_rate() -> None:
    """The Born rule in action: |amplitude|^2 is a frequency you can count.

    The bound is deliberately loose. 2000 fair coin flips have a standard deviation of
    about 22, so 0.46-0.54 sits roughly 3.5 sigma out on either side: wide enough that
    this passes for any seed, rather than only for the block of seeds it was written
    against.
    """
    trials = 2000
    ones = 0
    for seed in range(trials):
        qc = Circuit(seed=seed)
        a = qc.alloc()
        H(a)
        ones += qc.measure(a)

    assert 0.46 < ones / trials < 0.54


def test_measuring_the_same_qubit_twice_gives_the_same_answer(qc: Circuit) -> None:
    """The first measurement leaves the qubit in the state it reported, so the second
    is no longer random."""
    a = qc.alloc()
    H(a)
    first = qc.measure(a)
    for _ in range(5):
        assert qc.measure(a) == first


def test_measurement_leaves_the_state_normalized(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    qc.measure(a)
    assert qc.inspect.norm() == pytest.approx(1.0)


def test_a_measurement_is_recorded_in_the_history_with_its_outcome(qc: Circuit) -> None:
    a = qc.alloc()
    X(a)
    outcome = qc.measure(a)

    op = qc.history[-1]
    assert op.name == "measure"
    assert op.result == outcome == 1


def test_a_gate_can_still_be_applied_to_a_measured_qubit(qc: Circuit) -> None:
    """Measurement collapses a qubit; it does not retire it."""
    a = qc.alloc()
    H(a)
    outcome = qc.measure(a)
    X(a)
    assert qc.measure(a) == 1 - outcome


# ---- collapse of the joint state ----------------------------------------------


def test_measuring_one_half_of_a_bell_pair_collapses_the_other(bell_pair) -> None:
    """The heart of it: nothing is done to the second qubit, yet it is now definite —
    the amplitudes describing the other outcome are simply gone."""
    qc, a, b = bell_pair
    first = qc.measure(a)

    assert qc.inspect.probabilities()[0b11 if first else 0b00] == pytest.approx(1.0)
    assert qc.measure(b) == first


def test_bell_pair_outcomes_agree_across_many_seeds() -> None:
    for seed in range(50):
        qc = Circuit(seed=seed)
        a, b = qc.alloc_many(2)
        H(a)
        CNOT(a, b)
        assert qc.measure(a) == qc.measure(b)


def test_measuring_the_second_qubit_first_works_the_same_way(bell_pair) -> None:
    qc, a, b = bell_pair
    second = qc.measure(b)
    assert qc.measure(a) == second


# ---- registers ----------------------------------------------------------------


def test_measure_all_reads_the_first_qubit_as_the_most_significant_bit(qc: Circuit) -> None:
    reg = qc.register(2)
    X(reg[0])
    assert qc.measure_all(reg) == 2


def test_measure_all_returns_the_encoded_value(qc: Circuit) -> None:
    reg = qc.register(4)
    reg.encode(13)
    assert qc.measure_all(reg) == 13


def test_measure_all_records_one_measurement_per_qubit(qc: Circuit) -> None:
    reg = qc.register(3)
    qc.measure_all(reg)
    assert qc.gate_counts()["measure"] == 3


def test_a_ghz_register_only_ever_reads_all_zeros_or_all_ones() -> None:
    for seed in range(30):
        qc = Circuit(seed=seed)
        reg = qc.register(3)
        H(reg[0])
        CNOT(reg[0], reg[1])
        CNOT(reg[0], reg[2])
        assert qc.measure_all(reg) in (0b000, 0b111)


# ---- reset --------------------------------------------------------------------


def test_reset_returns_a_qubit_in_the_one_state_to_zero(qc: Circuit) -> None:
    a = qc.alloc()
    X(a)
    qc.reset(a)
    assert qc.inspect.probabilities() == pytest.approx([1.0, 0.0])


def test_reset_leaves_a_qubit_already_in_zero_alone(qc: Circuit) -> None:
    a = qc.alloc()
    qc.reset(a)
    assert qc.inspect.probabilities() == pytest.approx([1.0, 0.0])
    # Measured, found to be 0, and so not flipped: no X was needed.
    assert "X" not in qc.gate_counts()


def test_reset_works_on_a_superposition_whichever_way_it_collapses() -> None:
    for seed in range(20):
        qc = Circuit(seed=seed)
        a = qc.alloc()
        H(a)
        qc.reset(a)
        assert qc.inspect.probabilities() == pytest.approx([1.0, 0.0])


def test_reset_destroys_entanglement_which_is_why_ancilla_scopes_exist(qc: Circuit) -> None:
    a, b = qc.alloc_many(2)
    H(a)
    CNOT(a, b)
    qc.reset(a)

    assert qc.inspect.is_product([a])
    # b was left definite too: measuring a told us what b is.
    assert max(qc.inspect.probabilities()) == pytest.approx(1.0)


# ---- determinism --------------------------------------------------------------


def test_the_same_seed_gives_the_same_measurements() -> None:
    def run() -> list[int]:
        qc = Circuit(seed=99)
        reg = qc.register(4)
        for q in reg:
            H(q)
        return [qc.measure(q) for q in reg]

    assert run() == run()


def test_different_seeds_eventually_disagree() -> None:
    def run(seed: int) -> int:
        qc = Circuit(seed=seed)
        reg = qc.register(8)
        for q in reg:
            H(q)
        return qc.measure_all(reg)

    assert len({run(s) for s in range(20)}) > 1


def test_an_unseeded_circuit_still_measures() -> None:
    qc = Circuit()
    a = qc.alloc()
    H(a)
    assert qc.measure(a) in (0, 1)


def test_measurement_probabilities_match_the_amplitudes_they_came_from() -> None:
    """Statistical check that collapse samples the Born distribution, not something near it."""
    from qsim.gates import Ry

    trials = 2000
    counts = 0
    for seed in range(trials):
        qc = Circuit(seed=seed)
        a = qc.alloc()
        # sin^2(pi/8) ~= 0.146 chance of reading 1.
        Ry(a, theta=np.pi / 4)
        counts += qc.measure(a)

    # sd of the proportion is sqrt(p(1-p)/n) ~= 0.0079, so 0.03 is nearly 4 sigma.
    assert counts / trials == pytest.approx(np.sin(np.pi / 8) ** 2, abs=0.03)
