"""Unit tests for environment marking, the couplings, and the dephasing widget.

Read these as a usage guide: what `environment()` does and does not do, what each
coupling requires, and what the Inspector's system-versus-environment views mean.
"""

import sys

import numpy as np
import pytest

from qsim import Circuit
from qsim.decoherence import (
    amplitude_damping_coupling,
    damping_angle,
    dephasing_angle,
    dephasing_coupling,
    depolarizing_coupling,
    pointer_coupling,
)
from qsim.errors import QsimError
from qsim.gates import H, X

# ---- environment marking -----------------------------------------------------


def test_environment_qubits_are_ordinary_qubits_that_stay_in_the_state() -> None:
    """Marking traces nothing out: the tensor still has an axis for every qubit.

    This is the single most important thing to understand about `environment()`. It is
    a note about which qubits you intend to stop tracking, not an operation.
    """
    qc = Circuit()
    q = qc.alloc("q")
    env = qc.environment(2)

    assert qc.n_qubits == 3
    assert len(qc.inspect.state_vector()) == 8
    assert list(qc.system_qubits) == [q]
    assert list(qc.environment_qubits) == list(env)


def test_the_global_state_stays_pure_no_matter_how_hard_you_decohere() -> None:
    """Coupling at full strength leaves the whole circuit in a pure state.

    The qubit looks maximally mixed and the global state is perfectly pure, at the same
    time, with no contradiction: they are answers to different questions.
    """
    qc = Circuit()
    q = qc.alloc("q")
    env = qc.environment(1)
    H(q)
    dephasing_coupling(q, env[0], theta=np.pi)

    assert qc.inspect.norm() == pytest.approx(1.0, abs=1e-12)
    assert qc.inspect.entanglement_entropy(list(qc.qubits)) == pytest.approx(0.0, abs=1e-12)
    assert qc.inspect.system_entropy() == pytest.approx(1.0, abs=1e-12)


def test_environment_qubits_can_be_named() -> None:
    """`qc.environment(2, name="bath")` names the register and its qubits."""
    qc = Circuit()
    env = qc.environment(2, name="bath")
    assert env.name == "bath"
    assert [q.name for q in env] == ["bath0", "bath1"]


def test_with_no_environment_marked_the_system_is_everything() -> None:
    """`system_density_matrix()` then describes the whole circuit, and it is pure."""
    qc = Circuit()
    a, b = qc.alloc_many(2)
    H(a)

    assert qc.inspect.system_density_matrix().shape == (4, 4)
    assert qc.inspect.system_entropy() == pytest.approx(0.0, abs=1e-12)


def test_releasing_an_ancilla_also_forgets_that_it_was_an_environment() -> None:
    """A deallocated qubit leaves the environment set behind with it.

    Reaching into `_is_env` directly: there is no public read of the raw id set, and
    the point of the test is that the bookkeeping does not leak.
    """
    qc = Circuit()
    q = qc.alloc("q")
    with qc.ancilla(1) as scratch:
        qc._is_env.add(scratch[0]._id)  # pretend this scratch qubit was an environment
        assert len(qc.environment_qubits) == 1
    assert qc._is_env == set()
    assert list(qc.system_qubits) == [q]


# ---- coherence ---------------------------------------------------------------


def test_coherence_of_a_fresh_superposition_is_one_half() -> None:
    """|+> has the largest coherence a qubit can have."""
    qc = Circuit()
    q = qc.alloc("q")
    H(q)
    assert qc.inspect.coherence(q) == pytest.approx(0.5, abs=1e-12)


def test_coherence_of_a_basis_state_is_zero() -> None:
    """|0> and |1> are not superpositions, so they have nothing to lose."""
    qc = Circuit()
    q = qc.alloc("q")
    assert qc.inspect.coherence(q) == pytest.approx(0.0, abs=1e-12)
    X(q)
    assert qc.inspect.coherence(q) == pytest.approx(0.0, abs=1e-12)


def test_coherence_is_the_equatorial_shadow_of_the_bloch_vector() -> None:
    """|ρ₀₁| = sqrt(x² + y²) / 2, for any state and any amount of decoherence."""
    qc = Circuit()
    q = qc.alloc("q")
    env = qc.environment(1)
    H(q)
    dephasing_coupling(q, env[0], theta=1.1)

    x, y, _ = qc.inspect.bloch_vector(q)
    assert qc.inspect.coherence(q) == pytest.approx(np.hypot(x, y) / 2, abs=1e-12)


# ---- the textbook-parameter converters ---------------------------------------


@pytest.mark.parametrize("gamma", [0.0, 0.25, 0.5, 1.0])
def test_damping_angle_gives_the_requested_decay_probability(gamma: float) -> None:
    """`damping_angle(gamma)` is the theta whose decay probability is gamma."""
    qc = Circuit()
    q = qc.alloc("q")
    env = qc.environment(1)
    X(q)  # start excited, so all of the population can decay
    amplitude_damping_coupling(q, env[0], theta=damping_angle(gamma))

    rho = qc.inspect.system_density_matrix()
    assert rho[0, 0].real == pytest.approx(gamma, abs=1e-12)


@pytest.mark.parametrize("lam", [0.0, 0.3, 1.0])
def test_dephasing_angle_destroys_the_requested_fraction_of_coherence(lam: float) -> None:
    """`dephasing_angle(lam)` is the theta that multiplies coherence by (1 - lam)."""
    qc = Circuit()
    q = qc.alloc("q")
    env = qc.environment(1)
    H(q)
    dephasing_coupling(q, env[0], theta=dephasing_angle(lam))

    assert qc.inspect.coherence(q) == pytest.approx(0.5 * (1 - lam), abs=1e-12)


def test_damping_angle_rejects_a_number_that_is_not_a_probability() -> None:
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        damping_angle(1.5)


def test_dephasing_angle_rejects_a_fraction_outside_the_unit_interval() -> None:
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        dephasing_angle(-0.1)


# ---- coupling requirements ---------------------------------------------------


@pytest.mark.parametrize("size", [1, 3])
def test_depolarizing_coupling_needs_exactly_two_environment_qubits(size: int) -> None:
    """Four possible outcomes need two qubits to record them, and the error says so."""
    qc = Circuit()
    q = qc.alloc("q")
    env = qc.environment(size)
    with pytest.raises(ValueError, match="needs exactly 2 environment qubits"):
        depolarizing_coupling(q, env, p=0.1)


def test_depolarizing_coupling_rejects_a_probability_outside_the_unit_interval() -> None:
    qc = Circuit()
    q = qc.alloc("q")
    env = qc.environment(2)
    with pytest.raises(ValueError, match=r"must lie in \[0, 1\]"):
        depolarizing_coupling(q, env, p=1.4)


def test_pointer_coupling_rejects_an_unknown_basis_and_lists_the_valid_ones() -> None:
    """The message has to teach: the basis is the physically interesting knob here."""
    qc = Circuit()
    q = qc.alloc("q")
    env = qc.environment(1)
    with pytest.raises(QsimError, match="unknown pointer basis 'w'") as excinfo:
        pointer_coupling(q, env[0], theta=0.5, basis="w")

    message = str(excinfo.value)
    assert "'z'" in message and "'x'" in message and "'y'" in message


# ---- couplings as blocks -----------------------------------------------------


def test_a_coupling_is_recorded_as_a_named_block() -> None:
    """Couplings are @qsim.gate blocks, so they show up in block_counts()."""
    qc = Circuit()
    q = qc.alloc("q")
    env = qc.environment(1)
    dephasing_coupling(q, env[0], theta=0.3)
    dephasing_coupling(q, env[0], theta=0.3)

    assert qc.block_counts()["dephasing_coupling"] == 2


def test_pointer_coupling_in_the_y_basis_leaves_its_own_eigenstates_alone() -> None:
    """The y basis is reachable too, via the S†/H conjugation.

    The Y eigenstate (|0> + i|1>)/sqrt(2) is built with H then S, and a y-coupling
    passes it through untouched — while it destroys |0>, which a z-coupling protects.
    """
    from qsim.gates import S

    qc = Circuit()
    q = qc.alloc("q")
    env = qc.environment(1)
    H(q)
    S(q)  # now in the +y eigenstate
    pointer_coupling(q, env[0], theta=np.pi, basis="y")
    assert qc.inspect.system_entropy() == pytest.approx(0.0, abs=1e-12)

    qc = Circuit()
    q = qc.alloc("q")
    env = qc.environment(1)
    pointer_coupling(q, env[0], theta=np.pi, basis="y")  # acting on |0>
    assert qc.inspect.system_entropy() == pytest.approx(1.0, abs=1e-12)


def test_a_coupling_works_against_an_unmarked_qubit_too() -> None:
    """Being an environment is a choice of view, not a property the simulator enforces.

    The qubit decoheres against an ordinary ancilla exactly as it would against a
    marked one — the only difference is that the Inspector still counts that ancilla as
    part of the system, so `system_entropy()` sees the pure two-qubit state.
    """
    qc = Circuit()
    q, bystander = qc.alloc_many(2)
    H(q)
    dephasing_coupling(q, bystander, theta=np.pi)

    assert qc.inspect.coherence(q) == pytest.approx(0.0, abs=1e-12)
    # Nothing is marked, so "the system" is both qubits — and they are jointly pure.
    assert qc.inspect.system_entropy() == pytest.approx(0.0, abs=1e-12)
    # The mixedness appears only when you ask about q alone.
    assert qc.inspect.entanglement_entropy([q]) == pytest.approx(1.0, abs=1e-12)


# ---- the widget --------------------------------------------------------------


def test_dephasing_panels_draws_three_panels() -> None:
    """The drawing half of the widget is an ordinary function, testable headlessly."""
    from qsim import viz

    fig = viz.dephasing_panels(np.pi / 3)
    assert len(fig.axes) == 3


def test_dephasing_panels_shows_a_shorter_bloch_vector_at_stronger_coupling() -> None:
    """The panels really are recomputed per theta rather than drawn from a cache."""
    from qsim import viz

    weak = viz.dephasing_panels(0.2)
    strong = viz.dephasing_panels(3.0)
    assert weak.axes[0].get_title() != strong.axes[0].get_title()


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
def test_interact_dephasing_builds_a_slider() -> None:
    """The widget wiring runs headless: ipywidgets does not need a live kernel.

    The Agg backend cannot show a figure, which is exactly what the suppressed warning
    says; the point here is that the slider is built and the callback runs.
    """
    from qsim import viz

    viz.interact_dephasing()


def test_interact_dephasing_explains_how_to_install_ipywidgets(monkeypatch) -> None:
    """Without the dev dependency the error names it and says what to run.

    Setting the module to None in sys.modules is the standard way to make an import
    fail on demand — there is no other way to reach this branch with it installed.
    """
    from qsim import viz

    monkeypatch.setitem(sys.modules, "ipywidgets", None)
    with pytest.raises(ImportError, match="uv sync"):
        viz.interact_dephasing()
