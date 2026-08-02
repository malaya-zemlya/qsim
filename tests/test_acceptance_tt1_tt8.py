"""Acceptance tests TT1–TT8 from design doc §9 — conjugation, the tape, and hooks.

Tolerances are part of the specification and must not be loosened to make a test pass.

Full 2^n x 2^n matrices appear in this file, built with ``np.kron`` for n ≤ 4, purely to
check the library against an independent construction. The library never builds one.
"""

import numpy as np
import pytest

import qsim
from qsim import Circuit
from qsim.circuit import Op, Qubit
from qsim.combinators import Block
from qsim.decoherence import dephasing_coupling, pointer_coupling
from qsim.errors import QsimError
from qsim.gates import CNOT, H, Rx, Ry, Rz, S, T, X, Z

I2 = np.eye(2, dtype=np.complex128)
P0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)  # |0><0|
P1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)  # |1><1|
H_M = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
X_M = np.array([[0, 1], [1, 0]], dtype=np.complex128)


def _rz_matrix(theta: float) -> np.ndarray:
    """Rz as a 2x2 matrix, written out here rather than imported from the library."""
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)]).astype(np.complex128)


# ---- TT1: `within` equals the hand-built sandwich -----------------------------


def test_tt1_within_matches_the_sandwich_written_out_by_hand() -> None:
    """TT1: ``with within(H, q):`` around a coupling is H, coupling, H — exactly.

    The physical fact: conjugation is not a new operation. It is three operations in a
    row, and the combinator's only job is to guarantee the third is the inverse of the
    first.
    """
    theta = np.pi / 3

    scoped = Circuit(seed=1)
    q = scoped.alloc("q")
    env = scoped.environment(1)
    Ry(q, theta=0.7)  # a lopsided superposition, so the check is not about |0> alone
    with qsim.within(H, q):
        dephasing_coupling(q, env[0], theta=theta)

    by_hand = Circuit(seed=1)
    q2 = by_hand.alloc("q")
    env2 = by_hand.environment(1)
    Ry(q2, theta=0.7)
    H(q2)
    dephasing_coupling(q2, env2[0], theta=theta)
    H(q2)

    assert scoped.inspect.state_vector() == pytest.approx(by_hand.inspect.state_vector(), abs=1e-13)


def test_tt1_pointer_coupling_in_x_is_that_same_sandwich() -> None:
    """TT1: ``pointer_coupling(basis="x")`` is dephasing conjugated by H, and says so
    in its structure — it is implemented with ``within``, not with a hand-written
    H before and H after that could drift apart."""
    theta = np.pi / 3

    scoped = Circuit(seed=2)
    q = scoped.alloc("q")
    env = scoped.environment(1)
    Ry(q, theta=0.7)
    with qsim.within(H, q):
        dephasing_coupling(q, env[0], theta=theta)

    pointer = Circuit(seed=2)
    q2 = pointer.alloc("q")
    env2 = pointer.environment(1)
    Ry(q2, theta=0.7)
    pointer_coupling(q2, env2[0], theta=theta, basis="x")

    assert scoped.inspect.state_vector() == pytest.approx(pointer.inspect.state_vector(), abs=1e-13)


# ---- TT2: the adjoint of a conjugation inverts only the middle ----------------


@qsim.gate
def _phase_in_x(q: Qubit) -> None:
    """S applied in the X basis: H, then S, then H again."""
    with qsim.within(H, q):
        S(q)


def test_tt2_the_adjoint_of_a_sandwich_inverts_only_its_filling() -> None:
    """TT2: (V U V†)† = V U† V†. Reversing the whole sequence puts V back in front.

    Structurally: the recorded ops of the adjoint are H, S†, H — the basis change
    still happens first and is still undone last, and only the middle changed.
    """
    qc = Circuit(seed=3)
    q = qc.alloc("q")

    _phase_in_x(q)
    forward = [op.name for op in qc.history]

    _phase_in_x.adjoint()(q)
    backward = [op.name for op in qc.history[len(forward) :]]

    assert forward == ["H", "S", "H"]
    assert backward == ["H", "S†", "H"]


def test_tt2_a_sandwich_followed_by_its_adjoint_is_the_identity(random_state) -> None:
    """TT2: fidelity 1 to 1e-13 on a random state — the numerical half of the identity."""
    qc = Circuit(seed=3)
    q = qc.alloc("q")
    # A random one-qubit state, so the round trip is tested on something with no
    # symmetry to hide behind. Poking _psi directly is the only way to install one.
    qc._psi = random_state(1)
    before = qc.inspect.state_tensor().copy()

    _phase_in_x(q)
    _phase_in_x.adjoint()(q)

    assert qc.inspect.fidelity(before) == pytest.approx(1.0, abs=1e-13)


# ---- TT3: control distributes over the sandwich -------------------------------


@qsim.gate
def _turn_about_x(q: Qubit, *, theta: float) -> None:
    """Rz conjugated by H — which is Rx, since H swaps the x and z axes."""
    with qsim.within(H, q):
        Rz(q, theta=theta)


def _prepare_pair(seed: int) -> tuple[Circuit, Qubit, Qubit]:
    """Two qubits in a generic, entangled, deterministic state."""
    qc = Circuit(seed=seed)
    c, q = qc.alloc_many(2)
    H(c)
    T(c)
    Ry(q, theta=1.1)
    CNOT(c, q)
    return qc, c, q


def test_tt3_a_controlled_sandwich_matches_an_independently_built_matrix() -> None:
    """TT3: C(V U V†) against |0><0| ⊗ I + |1><1| ⊗ (H·Rz·H), assembled with np.kron.

    Gates compose left to right while matrices multiply right to left, so the block's
    own unitary is H · Rz · H (here symmetric, since H is its own inverse).
    """
    theta = 0.7
    qc, c, q = _prepare_pair(seed=4)
    before = qc.inspect.state_vector()

    _turn_about_x.controlled(c)(q, theta=theta)

    block_u = H_M @ _rz_matrix(theta) @ H_M
    controlled = np.kron(P0, I2) + np.kron(P1, block_u)

    assert qc.inspect.state_vector() == pytest.approx(controlled @ before, abs=1e-13)


def test_tt3_controlling_every_layer_agrees_with_controlling_only_the_middle() -> None:
    """TT3: V (CU) V† equals C(V U V†) when V avoids the control qubit.

    Control distributes over a product, so lifting all three layers is correct. It is
    also more work than necessary: in the branch where the control is |0⟩ the basis
    change and its undo cancel, so they never needed the control in the first place.
    qsim lifts uniformly — the shortcut is an optimization, not the meaning.
    """
    theta = 0.7

    uniform, c1, q1 = _prepare_pair(seed=5)
    _turn_about_x.controlled(c1)(q1, theta=theta)

    optimized, c2, q2 = _prepare_pair(seed=5)
    H(q2)  # the basis change, deliberately left uncontrolled
    with optimized.control(c2):
        Rz(q2, theta=theta)
    H(q2)

    assert uniform.inspect.state_vector() == pytest.approx(
        optimized.inspect.state_vector(), abs=1e-13
    )


# ---- TT4: rewind is exact, and the tape stays honest --------------------------


def _random_block(qc: Circuit, qubits: tuple[Qubit, ...], rng: np.random.Generator, count: int):
    """Apply ``count`` randomly chosen gates to randomly chosen qubits."""
    one_qubit = [H, X, Z, S, T]
    rotations = [Rx, Ry, Rz]
    for _ in range(count):
        roll = rng.integers(0, 3)
        if roll == 0:
            one_qubit[rng.integers(0, len(one_qubit))](qubits[rng.integers(0, len(qubits))])
        elif roll == 1:
            rotations[rng.integers(0, len(rotations))](
                qubits[rng.integers(0, len(qubits))], theta=float(rng.uniform(0, 2 * np.pi))
            )
        else:
            # Two distinct qubits for the entangler: a qubit cannot control itself.
            a, b = rng.choice(len(qubits), size=2, replace=False)
            CNOT(qubits[int(a)], qubits[int(b)])


def test_tt4_rewind_restores_the_state_and_records_the_undoing() -> None:
    """TT4: 20 random gates on 5 qubits, undone exactly — and visibly.

    Two claims at once. The state returns to fidelity 1: unitaries lose nothing, so
    walking the tape backwards is enough and no saved copy of the state is needed.
    And the history *grows* by 20 ops rather than shrinking by 20: the record says
    what physically happened, including the undoing.
    """
    rng = np.random.default_rng(11)
    qc = Circuit(seed=6)
    qubits = qc.alloc_many(5)
    _random_block(qc, qubits, rng, 8)  # a generic starting state

    mark = qc.checkpoint()
    saved = qc.inspect.state_tensor().copy()
    history_at_mark = len(qc.history)

    _random_block(qc, qubits, rng, 20)
    assert len(qc.history) == history_at_mark + 20
    assert qc.inspect.fidelity(saved) < 0.99  # the block really did move the state

    qc.rewind(mark)

    assert qc.inspect.fidelity(saved) == pytest.approx(1.0, abs=1e-13)
    assert len(qc.history) == history_at_mark + 40


# ---- TT5: measurement severs the tape ------------------------------------------


def test_tt5_rewinding_across_a_measurement_raises_and_explains_why() -> None:
    """TT5: measurement is the one op with no inverse rule, so the tape ends there.

    The error message has to do real teaching work here: it names the reason (nothing
    restores the discarded branches), draws the analogy to a non-differentiable
    operation severing an autograd graph, and points at the alternative — a coherent
    record, which *can* be undone (the quantum eraser, TD3).
    """
    qc = Circuit(seed=7)
    a, b = qc.alloc_many(2)
    mark = qc.checkpoint()
    H(a)
    CNOT(a, b)
    qc.measure(a)

    with pytest.raises(QsimError) as raised:
        qc.rewind(mark)

    message = str(raised.value)
    assert "measurement" in message
    assert "severed" in message
    assert "inverse" in message
    assert "eraser" in message


# ---- TT6: allocation pins the tape ---------------------------------------------


def test_tt6_rewinding_across_an_allocation_raises() -> None:
    """TT6: a new qubit adds an axis, so the recorded ops no longer describe the state
    they were recorded on."""
    qc = Circuit(seed=8)
    a = qc.alloc()
    H(a)
    mark = qc.checkpoint()
    X(a)
    qc.alloc()

    with pytest.raises(QsimError) as raised:
        qc.rewind(mark)

    assert "which qubits exist" in str(raised.value)


def test_tt6_rewinding_across_an_ancilla_release_raises() -> None:
    """TT6: releasing scratch removes an axis and renumbers every axis after it — the
    suffix's ops would name qubits that are no longer there."""
    qc = Circuit(seed=9)
    a = qc.alloc()
    H(a)
    mark = qc.checkpoint()
    with qc.ancilla(1) as scratch:
        CNOT(a, scratch[0])
        CNOT(a, scratch[0])  # uncompute, so the scope exits cleanly

    with pytest.raises(QsimError) as raised:
        qc.rewind(mark)

    assert "which qubits exist" in str(raised.value)


# ---- TT7: hooks see everything and touch nothing --------------------------------


def test_tt7_a_hook_sees_every_gate_and_every_measurement() -> None:
    """TT7: the hook fires for measurements too, which is why the history append had to
    be funnelled through one place."""
    qc = Circuit(seed=10)
    a, b = qc.alloc_many(2)
    seen: list[str] = []
    qc.on_op(lambda op, circuit: seen.append(op.name))

    H(a)
    CNOT(a, b)
    qc.measure(a)
    qc.measure(b)

    assert seen == ["H", "CNOT", "measure", "measure"]
    assert seen == [op.name for op in qc.history]


def test_tt7_a_hook_can_compute_the_entanglement_entropy_after_every_gate() -> None:
    """TT7: an entropy trace is a hook, not a feature — the same numbers you would get
    by asking after each gate by hand. This is how ``viz.entropy_trace`` will work."""
    live: list[float] = []
    hooked = Circuit(seed=11)
    a, b = hooked.alloc_many(2)
    hooked.on_op(lambda op, circuit: live.append(circuit.inspect.entanglement_entropy([a])))
    H(a)
    CNOT(a, b)
    Ry(b, theta=0.4)

    by_hand: list[float] = []
    manual = Circuit(seed=11)
    c, d = manual.alloc_many(2)
    H(c)
    by_hand.append(manual.inspect.entanglement_entropy([c]))
    CNOT(c, d)
    by_hand.append(manual.inspect.entanglement_entropy([c]))
    Ry(d, theta=0.4)
    by_hand.append(manual.inspect.entanglement_entropy([c]))

    assert live == pytest.approx(by_hand, abs=1e-13)


def test_tt7_a_removed_hook_stops_hearing_about_ops() -> None:
    """TT7: ``handle.remove()`` detaches, and nothing arrives afterwards."""
    qc = Circuit(seed=12)
    a = qc.alloc()
    seen: list[str] = []
    handle = qc.on_op(lambda op, circuit: seen.append(op.name))

    H(a)
    handle.remove()
    X(a)
    T(a)

    assert seen == ["H"]


def test_tt7_a_hook_that_applies_a_gate_raises() -> None:
    """TT7: hooks observe and must not emit. An op from inside a hook would land on the
    tape with nothing in the program accounting for it — and would fire the hook again."""
    qc = Circuit(seed=13)
    a, b = qc.alloc_many(2)
    qc.on_op(lambda op, circuit: X(b))

    with pytest.raises(QsimError) as raised:
        H(a)

    assert "hook" in str(raised.value)


# ---- TT8: the block algebra is closed -------------------------------------------


@qsim.gate
def bell(a: Qubit, b: Qubit) -> None:
    """The two-gate block that makes a Bell pair."""
    H(a)
    CNOT(a, b)


def test_tt8_the_adjoint_of_a_block_is_a_block() -> None:
    """TT8: an operation on a block returns a block — named, countable, chainable —
    exactly as an operation on a gate returns a gate."""
    undo = bell.adjoint()

    assert isinstance(undo, Block)
    assert undo.name == "bell†"
    assert repr(undo) == "<Block bell†>"


def test_tt8_a_doubly_adjointed_block_acts_as_the_original(random_state) -> None:
    """TT8: two reversals compose to none."""
    once = Circuit(seed=14)
    a, b = once.alloc_many(2)
    once._psi = random_state(2)  # a generic state; see TT2 on poking _psi
    before = once.inspect.state_tensor().copy()
    bell(a, b)
    plain = once.inspect.state_tensor().copy()

    twice = Circuit(seed=14)
    c, d = twice.alloc_many(2)
    twice._psi = before.copy()
    bell.adjoint().adjoint()(c, d)

    assert twice.inspect.state_vector() == pytest.approx(plain.reshape(-1), abs=1e-13)


def test_tt8_adjoint_and_controlled_chain_and_match_a_built_matrix() -> None:
    """TT8: ``bell.adjoint().controlled(c)`` against the 8x8 matrix built with np.kron.

    bell is CNOT · (H ⊗ I), so bell† is (H ⊗ I) · CNOT — the same two matrices in the
    other order, each replaced by its inverse (both are self-inverse here).
    """
    qc = Circuit(seed=15)
    c, a, b = qc.alloc_many(3)
    H(c)
    T(c)
    Ry(a, theta=1.1)
    Rz(b, theta=0.4)
    before = qc.inspect.state_vector()

    bell.adjoint().controlled(c)(a, b)

    cnot_4 = np.kron(P0, I2) + np.kron(P1, X_M)
    bell_dagger = np.kron(H_M, I2) @ cnot_4
    controlled = np.kron(P0, np.eye(4, dtype=np.complex128)) + np.kron(P1, bell_dagger)

    assert qc.inspect.state_vector() == pytest.approx(controlled @ before, abs=1e-13)


def test_tt8_block_counts_reports_the_derived_names() -> None:
    """TT8: the derived block is what ran, so it is what the count says ran."""
    qc = Circuit(seed=16)
    c, a, b = qc.alloc_many(3)

    bell(a, b)
    bell.adjoint()(a, b)
    bell.controlled(c)(a, b)
    bell.adjoint().controlled(c)(a, b)

    assert qc.block_counts() == {"bell": 1, "bell†": 1, "C-bell": 1, "C-bell†": 1}
    assert {op.block for op in qc.history} == {"bell", "bell†", "C-bell", "C-bell†"}


def test_tt8_every_op_of_a_derived_block_is_a_plain_op() -> None:
    """TT8: "block" is a grouping, not a new kind of instruction — the tape holds only
    elementary gates, each stamped with the block it came from."""
    qc = Circuit(seed=17)
    c, a, b = qc.alloc_many(3)
    bell.adjoint().controlled(c)(a, b)

    assert all(isinstance(op, Op) and op.gate is not None for op in qc.history)
    assert [op.name for op in qc.history] == ["CNOT", "H"]
    # The lifted control goes in front of whatever controls the op already had, so the
    # CNOT is now a two-control op (c and a) and the H a one-control op (c).
    assert [op.controls for op in qc.history] == [(c._id, a._id), (c._id,)]
