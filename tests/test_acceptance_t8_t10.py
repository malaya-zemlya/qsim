"""Acceptance tests T8–T10 from design doc §9 — the combinators.

Tolerances are part of the specification and must not be loosened to make a test pass.

Full 2^n x 2^n matrices appear in this file, built with ``np.kron`` for n ≤ 4, purely to
check the library against an independent construction. The library never builds one.
"""

import numpy as np
import pytest

from qsim import Circuit
from qsim.errors import DirtyAncillaError
from qsim.gates import CNOT, CZ, H, Rx, Ry, Rz, S, T, X, Y, Z

I2 = np.eye(2, dtype=np.complex128)
P0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)  # |0><0|
P1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)  # |1><1|
H_M = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
X_M = np.array([[0, 1], [1, 0]], dtype=np.complex128)


# ---- T8: ancilla cleanup is enforced -----------------------------------------


def test_t8_leaving_an_ancilla_entangled_raises_on_scope_exit() -> None:
    """T8: the deliverable of this phase. Scratch qubits that still carry a record of
    the computation cannot be released, and the simulator checks rather than trusts."""
    qc = Circuit(seed=1)
    a = qc.alloc()
    H(a)

    with pytest.raises(DirtyAncillaError) as raised:
        with qc.ancilla(1) as scratch:
            CNOT(a, scratch[0])  # copies which-branch information into the scratch

    assert "interfere" in str(raised.value)


def test_t8_the_same_block_with_uncomputation_exits_cleanly() -> None:
    """T8: the identical circuit, plus the CNOT that undoes the first one, is fine.

    This is Bennett's trick in miniature: use the scratch, then run the computation
    that dirtied it backwards, and the entanglement that would have destroyed your
    interference is gone.
    """
    qc = Circuit(seed=1)
    a = qc.alloc()
    H(a)
    before = qc.inspect.state_vector()

    with qc.ancilla(1) as scratch:
        CNOT(a, scratch[0])
        CNOT(a, scratch[0])  # uncompute

    assert qc.n_qubits == 1
    assert qc.inspect.state_vector() == pytest.approx(before)


def test_t8_a_dirty_ancilla_destroys_the_interference_it_was_hiding_in() -> None:
    """T8: *why* the check exists. An H-sandwich returns |0⟩ with certainty — unless
    something recorded which path was taken, in which case the two paths can no longer
    cancel and the answer becomes a coin flip. This is the two-slit experiment, and the
    ancilla is playing the part of the environment."""
    clean = Circuit(seed=1)
    a = clean.alloc()
    H(a)
    H(a)
    # Probability this qubit reads 0, read off its own reduced state.
    assert clean.inspect.reduced_density_matrix([a])[0, 0].real == pytest.approx(1.0)

    watched = Circuit(seed=1)
    b = watched.alloc()
    spy = watched.alloc()
    H(b)
    CNOT(b, spy)  # the spy learns which path
    H(b)
    assert watched.inspect.reduced_density_matrix([b])[0, 0].real == pytest.approx(0.5)

    # And the reason: b is no longer in a state of its own, so it has nothing to
    # interfere with. Nobody looked at the spy — its mere existence is enough.
    assert watched.inspect.entanglement_entropy([b]) == pytest.approx(1.0)


# ---- T9: adjoint is a true inverse -------------------------------------------


def random_block(circuit: Circuit, qubits: list, seed: int):
    """Build a fixed, seeded sequence of gates over the given qubits."""
    rng = np.random.default_rng(seed)
    one_qubit = [H, X, Y, Z, S, T]
    rotations = [Rx, Ry, Rz]
    plan: list = []
    for _ in range(40):
        choice = rng.integers(0, 3)
        if choice == 0:
            gate = one_qubit[rng.integers(0, len(one_qubit))]
            plan.append((gate, (int(rng.integers(0, 6)),), None))
        elif choice == 1:
            plan.append(
                (
                    rotations[rng.integers(0, len(rotations))],
                    (int(rng.integers(0, 6)),),
                    float(rng.uniform(0, 2 * np.pi)),
                )
            )
        else:
            j, k = rng.choice(6, size=2, replace=False)
            plan.append(([CNOT, CZ][rng.integers(0, 2)], (int(j), int(k)), None))

    def run() -> None:
        for gate, indices, theta in plan:
            targets = [qubits[i] for i in indices]
            if theta is None:
                gate(*targets)
            else:
                gate(*targets, theta=theta)

    return run


def test_t9_a_block_followed_by_its_adjoint_returns_the_original_state() -> None:
    """T9: fidelity within 1e-13 of 1 after 40 gates on 6 qubits and their inverses.

    Every gate is a rotation, so every gate has an inverse; running the whole program
    backwards is just running each rotation the other way, last one first.
    """
    qc = Circuit(6, seed=9)
    qubits = list(qc.qubits)

    # Start from a scrambled state, so this is not merely a fact about |000000>.
    random_block(qc, qubits, seed=3)()
    original = qc.inspect.state_vector()

    block = random_block(qc, qubits, seed=17)
    block()
    with qc.adjoint():
        block()

    assert abs(qc.inspect.fidelity(original) - 1.0) < 1e-13


def test_t9_the_adjoint_alone_is_not_the_identity() -> None:
    """T9: a guard against the test passing because the block did nothing."""
    qc = Circuit(6, seed=9)
    qubits = list(qc.qubits)
    original = qc.inspect.state_vector()

    random_block(qc, qubits, seed=17)()

    assert qc.inspect.fidelity(original) < 0.99


# ---- T10: control is correct --------------------------------------------------


def test_t10_controlling_x_is_exactly_cnot() -> None:
    """T10: from a random input state, the scope and the built-in gate agree."""
    scoped = Circuit(seed=4)
    a, b = scoped.alloc_many(2)
    H(a)
    Ry(b, theta=0.9)
    T(a)
    with scoped.control(a):
        X(b)

    direct = Circuit(seed=4)
    c, d = direct.alloc_many(2)
    H(c)
    Ry(d, theta=0.9)
    T(c)
    CNOT(c, d)

    assert scoped.inspect.state_vector() == pytest.approx(direct.inspect.state_vector())


def test_t10_a_controlled_block_matches_an_independently_built_matrix() -> None:
    """T10: a two-gate block, lifted by ``control``, against the 8x8 matrix

        |0><0| ⊗ I ⊗ I  +  |1><1| ⊗ U

    assembled here with np.kron. U is the block's own 4x4 unitary, and because gates
    compose left to right while matrices multiply right to left, U = CNOT · (H ⊗ I).
    """
    qc = Circuit(seed=5)
    c, a, b = qc.alloc_many(3)
    # A generic starting state, so the check is not about |000> in particular.
    H(c)
    T(c)
    Ry(a, theta=1.1)
    Rz(b, theta=0.4)
    before = qc.inspect.state_vector()

    with qc.control(c):
        H(a)
        CNOT(a, b)

    cnot_4 = np.kron(P0, I2) + np.kron(P1, X_M)
    block_u = cnot_4 @ np.kron(H_M, I2)
    controlled = np.kron(P0, np.eye(4, dtype=np.complex128)) + np.kron(P1, block_u)

    assert qc.inspect.state_vector() == pytest.approx(controlled @ before)


def test_t10_a_control_in_superposition_leaves_both_branches_alive() -> None:
    """T10: the control is not a switch that picks a branch — both happen at once, and
    the |0> branch is left exactly as it was."""
    qc = Circuit(seed=6)
    c, t = qc.alloc_many(2)
    H(c)
    with qc.control(c):
        X(t)

    assert qc.inspect.probabilities() == pytest.approx([0.5, 0, 0, 0.5])


def test_t10_controlling_with_two_qubits_matches_toffoli() -> None:
    from qsim.gates import Toffoli

    scoped = Circuit(seed=7)
    a, b, t = scoped.alloc_many(3)
    H(a)
    H(b)
    with scoped.control(a, b):
        X(t)

    direct = Circuit(seed=7)
    c, d, u = direct.alloc_many(3)
    H(c)
    H(d)
    Toffoli(c, d, u)

    assert scoped.inspect.state_vector() == pytest.approx(direct.inspect.state_vector())
