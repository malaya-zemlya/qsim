"""Acceptance tests T1–T7 from design doc §9 — the specification for Phase 1.

Each test says which physical fact it pins down. Tolerances are part of the spec and
must not be loosened to make a test pass.

One deviation from the design doc's wording: where §9's T2 snippet writes
``entanglement_entropy([0])`` with an axis index, these tests pass qubit *handles*
(``[a]``). Design doc §2.4 and the Phase 1 plan both require handles to be the only
way to name a qubit, precisely so that axis numbers never appear in user code. The
physical content of the test is unchanged.
"""

import copy

import numpy as np
import pytest

from qsim import Circuit
from qsim.errors import DeadQubitError, NoCloningError, QsimError
from qsim.gates import CNOT, CZ, H, Rx, Ry, Rz, S, T, X, Y, Z

# ---- T1 -----------------------------------------------------------------------


def test_t1_hadamard_creates_an_equal_superposition() -> None:
    """T1: H|0> = (|0> + |1>)/sqrt(2) — this is where "both at once" comes from."""
    qc = Circuit(1)
    H(qc.qubits[0])

    expected = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)
    assert qc.inspect.state_vector() == pytest.approx(expected, abs=1e-15)


def test_t1_hadamard_is_its_own_inverse() -> None:
    """T1: H twice is the identity. The second H makes the two paths interfere,
    and they cancel everywhere except |0>."""
    qc = Circuit(1)
    a = qc.qubits[0]
    H(a)
    H(a)

    assert qc.inspect.state_vector() == pytest.approx([1, 0], abs=1e-15)


# ---- T2 -----------------------------------------------------------------------


def test_t2_a_bell_state_is_half_zero_zero_and_half_one_one() -> None:
    """T2: H then CNOT builds (|00> + |11>)/sqrt(2). The |01> and |10> outcomes never
    happen — the two qubits always agree, whatever they turn out to be."""
    qc = Circuit(2)
    a, b = qc.qubits
    H(a)
    CNOT(a, b)

    assert qc.inspect.probabilities() == pytest.approx([0.5, 0, 0, 0.5])


def test_t2_a_bell_state_carries_exactly_one_bit_of_entanglement_entropy() -> None:
    """T2: entropy exactly 1 bit is the acceptance criterion for the partial trace."""
    qc = Circuit(2)
    a, b = qc.qubits
    H(a)
    CNOT(a, b)

    assert abs(qc.inspect.entanglement_entropy([a]) - 1.0) < 1e-12
    assert not qc.inspect.is_product([a])


# ---- T3 -----------------------------------------------------------------------


def test_t3_measuring_one_qubit_of_a_ghz_state_forces_the_other_two() -> None:
    """T3: in (|000> + |111>)/sqrt(2), the first measurement decides all three. Run
    over many seeds so both branches are exercised."""
    for seed in range(50):
        qc = Circuit(3, seed=seed)
        a, b, c = qc.qubits
        H(a)
        CNOT(a, b)
        CNOT(a, c)

        first = qc.measure(a)
        assert qc.measure(b) == first
        assert qc.measure(c) == first


def test_t3_a_ghz_state_produces_both_outcomes_across_seeds() -> None:
    """T3: the correlation is perfect, but which value appears is genuinely random."""
    outcomes = set()
    for seed in range(50):
        qc = Circuit(3, seed=seed)
        a, b, c = qc.qubits
        H(a)
        CNOT(a, b)
        CNOT(a, c)
        outcomes.add(qc.measure(a))

    assert outcomes == {0, 1}


# ---- T4 -----------------------------------------------------------------------


def test_t4_single_qubit_gates_alone_can_never_create_entanglement() -> None:
    """T4: entropy below 1e-12 across every cut. Entanglement needs a gate that acts
    on two qubits at once; no amount of local operations will do it."""
    qc = Circuit(4, seed=7)
    rng = np.random.default_rng(7)
    for q in qc.qubits:
        H(q)
        Rx(q, theta=float(rng.uniform(0, 2 * np.pi)))
        T(q)
        Ry(q, theta=float(rng.uniform(0, 2 * np.pi)))

    qubits = list(qc.qubits)
    for cut in ([qubits[0]], [qubits[2]], qubits[:2], qubits[:3]):
        assert qc.inspect.entanglement_entropy(cut) < 1e-12


# ---- T5 -----------------------------------------------------------------------


def test_t5_two_hundred_random_gates_leave_the_norm_at_one() -> None:
    """T5: every gate is unitary, so total probability is conserved exactly. Drift
    here would mean a broken kernel."""
    qc = Circuit(8, seed=42)
    qubits = list(qc.qubits)
    rng = np.random.default_rng(42)

    one_qubit = [H, X, Y, Z, S, T]
    rotations = [Rx, Ry, Rz]
    two_qubit = [CNOT, CZ]

    for _ in range(200):
        choice = rng.integers(0, 3)
        if choice == 0:
            one_qubit[rng.integers(0, len(one_qubit))](qubits[rng.integers(0, 8)])
        elif choice == 1:
            rotations[rng.integers(0, len(rotations))](
                qubits[rng.integers(0, 8)], theta=float(rng.uniform(0, 2 * np.pi))
            )
        else:
            j, k = rng.choice(8, size=2, replace=False)
            two_qubit[rng.integers(0, len(two_qubit))](qubits[int(j)], qubits[int(k)])

    assert abs(qc.inspect.norm() - 1.0) < 1e-12


# ---- T6 -----------------------------------------------------------------------


def test_t6_copying_a_qubit_handle_raises_and_explains_no_cloning() -> None:
    """T6: the no-cloning theorem, enforced at the API surface. The message is part of
    the specification — the error has to teach."""
    qc = Circuit(1)
    with pytest.raises(NoCloningError, match="no-cloning"):
        copy.copy(qc.qubits[0])


def test_t6_a_qubit_cannot_control_a_gate_on_itself() -> None:
    """T6: CNOT(a, a) would require reading a's value to decide what to do to it,
    which would produce a copy of that value."""
    qc = Circuit(2)
    a, _ = qc.qubits
    with pytest.raises(NoCloningError, match="no-cloning"):
        CNOT(a, a)


def test_t6_a_three_qubit_gate_rejects_a_repeated_qubit() -> None:
    """T6: Toffoli(a, b, a) — same rule, checked across all the arguments."""
    from qsim.gates import Toffoli

    qc = Circuit(2)
    a, b = qc.qubits
    with pytest.raises(NoCloningError, match="no-cloning"):
        Toffoli(a, b, a)


def test_t6_gates_cannot_span_two_circuits() -> None:
    """T6: two circuits are two separate physical systems."""
    first, second = Circuit(1), Circuit(1)
    with pytest.raises(QsimError, match="different circuit"):
        CNOT(first.qubits[0], second.qubits[0])


# ---- T7 -----------------------------------------------------------------------


def test_t7_a_released_qubit_handle_cannot_be_used_for_a_gate() -> None:
    """T7: using a handle after its qubit is gone raises DeadQubitError.

    Phase 2's ancilla scopes are what will release qubits for real; until then the
    liveness flag is set directly here (white-box, one line) so the guard itself is
    tested now rather than in three phases' time.
    """
    qc = Circuit(1)
    a = qc.qubits[0]
    a._live = False

    with pytest.raises(DeadQubitError, match="no longer refers to a qubit"):
        H(a)


def test_t7_a_released_qubit_handle_cannot_be_measured() -> None:
    """T7: the same guard protects measurement, not just gates."""
    qc = Circuit(1)
    a = qc.qubits[0]
    a._live = False

    with pytest.raises(DeadQubitError):
        qc.measure(a)


def test_t7_the_dead_qubit_message_explains_what_went_wrong() -> None:
    """T7: the error explains that the handle names nothing physical, rather than
    silently acting on whichever qubit shifted into its place."""
    qc = Circuit(2)
    a, _ = qc.qubits
    a._live = False

    with pytest.raises(DeadQubitError, match="names nothing physical"):
        qc.inspect.bloch_vector(a)
