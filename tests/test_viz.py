"""The plots. Rendered headless under the Agg backend (set in conftest.py)."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from qsim import Circuit, viz
from qsim.gates import CNOT, H, T, X


def test_amplitudes_draws_one_bar_per_basis_state(qc: Circuit) -> None:
    qc.alloc_many(2)
    fig = viz.amplitudes(qc)
    ax = fig.axes[0]

    assert len(ax.patches) == 4
    assert ax.get_ylabel() == "|amplitude|"
    assert [t.get_text() for t in ax.get_xticklabels()] == ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]
    plt.close(fig)


def test_amplitudes_adds_a_phase_colorbar(qc: Circuit) -> None:
    """Bar height is magnitude, color is phase — and the color needs a legend."""
    a = qc.alloc()
    H(a)
    T(a)
    fig = viz.amplitudes(qc)

    assert fig.axes[1].get_ylabel() == "phase of amplitude"
    plt.close(fig)


def test_two_states_with_the_same_heights_can_have_different_colors(qc: Circuit) -> None:
    """The punchline of the whole visualization: identical measurement statistics,
    different phases, different futures."""
    a = qc.alloc()
    H(a)
    plain = [p.get_facecolor() for p in viz.amplitudes(qc).axes[0].patches]

    from qsim.gates import Z

    Z(a)
    flipped = [p.get_facecolor() for p in viz.amplitudes(qc).axes[0].patches]

    assert plain[0] == flipped[0]
    assert plain[1] != flipped[1]
    plt.close("all")


def test_phase_coloring_can_be_switched_off(qc: Circuit) -> None:
    qc.alloc()
    fig = viz.amplitudes(qc, phase_as_hue=False)
    assert len(fig.axes) == 1  # no colorbar
    plt.close(fig)


def test_bitstring_labels_are_rotated_once_they_get_crowded(qc: Circuit) -> None:
    qc.alloc_many(5)
    fig = viz.amplitudes(qc)
    assert fig.axes[0].get_xticklabels()[0].get_rotation() == 90
    plt.close(fig)


def test_amplitudes_refuses_to_draw_an_unreadable_number_of_bars(qc: Circuit) -> None:
    qc.alloc_many(7)
    with pytest.raises(ValueError, match="128 bars"):
        viz.amplitudes(qc)


def test_probabilities_shows_the_most_likely_outcomes_first(bell_pair) -> None:
    qc, _, _ = bell_pair
    fig = viz.probabilities(qc)
    ax = fig.axes[0]

    heights = [p.get_height() for p in ax.patches]
    assert heights == sorted(heights, reverse=True)
    assert ax.get_ylabel() == "probability"
    plt.close(fig)


def test_probabilities_can_be_capped_at_the_top_few(qc: Circuit) -> None:
    reg = qc.register(5)
    for q in reg:
        H(q)
    fig = viz.probabilities(qc, top=4)

    assert len(fig.axes[0].patches) == 4
    plt.close(fig)


def test_the_bloch_plot_reports_the_vector_length_in_its_title(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    fig = viz.bloch(a)

    assert "length 1.00" in fig.axes[0].get_title()
    plt.close(fig)


def test_an_entangled_qubit_plots_as_a_zero_length_vector(bell_pair) -> None:
    qc, a, _ = bell_pair
    fig = viz.bloch(a)
    assert "length 0.00" in fig.axes[0].get_title()
    plt.close(fig)


# ---- rich display in Jupyter --------------------------------------------------


def test_a_circuit_renders_as_an_html_table(bell_pair) -> None:
    qc, _, _ = bell_pair
    html = qc._repr_html_()

    assert "<table" in html
    assert "|00⟩" in html and "|11⟩" in html
    assert "2 qubits" in html


def test_the_html_uses_current_color_so_it_works_in_dark_notebooks(qc: Circuit) -> None:
    qc.alloc()
    assert "currentColor" in qc._repr_html_()


def test_the_html_encodes_phase_as_a_bar_color(qc: Circuit) -> None:
    a = qc.alloc()
    H(a)
    from qsim.gates import Z

    Z(a)
    html = qc._repr_html_()
    # The |1> amplitude is negative, i.e. phase pi, i.e. hue 180.
    assert "hsl(180" in html


def test_the_html_says_how_many_terms_it_left_out(qc: Circuit) -> None:
    reg = qc.register(4)
    for q in reg:
        H(q)
    assert "8 more basis states not shown" in qc._repr_html_()


def test_one_omitted_term_is_described_in_the_singular(qc: Circuit) -> None:
    a, b, c = qc.alloc_many(3)
    H(a)
    H(b)
    CNOT(a, c)
    X(c)
    # Nine amplitudes would be too many; this state has exactly nine significant ones.
    html = viz.circuit_html(qc, max_terms=3)
    assert "1 more basis state not shown" in html


def test_an_unnamed_circuit_omits_the_name_from_its_html() -> None:
    qc = Circuit(1)
    assert "‘" not in qc._repr_html_()


def test_the_html_is_self_contained(bell_pair) -> None:
    """No JavaScript, no images, no external assets — it has to work offline."""
    qc, _, _ = bell_pair
    html = qc._repr_html_()

    assert "<script" not in html
    assert "http" not in html


def test_every_plot_survives_a_single_qubit_circuit(qc: Circuit) -> None:
    """Edge case: one qubit, two bars, and a Bloch sphere."""
    a = qc.alloc()
    for fig in (viz.amplitudes(qc), viz.probabilities(qc), viz.bloch(a)):
        assert fig is not None
        plt.close(fig)


def test_plots_work_on_a_state_with_many_distinct_phases(qc: Circuit) -> None:
    reg = qc.register(3)
    for i, q in enumerate(reg):
        H(q)
        from qsim.gates import Rz

        Rz(q, theta=np.pi / (i + 2))
    fig = viz.amplitudes(qc)
    assert len(fig.axes[0].patches) == 8
    plt.close(fig)
