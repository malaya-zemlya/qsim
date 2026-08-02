"""Acceptance tests TI1–TI4 from `plans/phase-2.25-interferometers.md`.

Tolerances are part of the specification and must not be loosened to make a test pass.
"""

from collections import Counter

import numpy as np
import pytest

from qsim.algorithms.interferometry import (
    BombResult,
    bomb_probabilities,
    bomb_test,
    distinguishability,
    filter_chain,
    fringes,
    mach_zehnder,
    n_path_fringes,
    visibility,
)

# ---- TI1: interference fringes -----------------------------------------------


def test_ti1_the_fringe_pattern_is_the_cosine_squared_of_half_the_phase() -> None:
    """TI1: P(port 0) = cos²(φ/2) to 1e-12.

    Both beam splitters are perfectly fair, so a probability-only account would predict a
    flat 1/2 whatever the phase. It is not flat, because amplitudes add before they are
    squared.
    """
    phases = np.linspace(-2 * np.pi, 2 * np.pi, 41)
    expected = np.cos(phases / 2) ** 2

    assert fringes(phases) == pytest.approx(expected, abs=1e-12)


def test_ti1_one_port_can_be_completely_dark() -> None:
    """TI1: the whole point. With no phase shift the photon *never* leaves by port 1,
    even though each beam splitter sent it both ways."""
    assert mach_zehnder(0.0) == pytest.approx(1.0, abs=1e-12)
    assert mach_zehnder(np.pi) == pytest.approx(0.0, abs=1e-12)


def test_ti1_a_perfect_which_path_detector_flattens_the_fringes() -> None:
    """TI1: at full detector strength every phase gives 1/2 — the classical prediction,
    recovered exactly when the paths become distinguishable."""
    phases = np.linspace(-np.pi, np.pi, 21)
    assert fringes(phases, detector_strength=np.pi) == pytest.approx(0.5, abs=1e-12)


# ---- TI2: complementarity ------------------------------------------------------


def measured_visibility(detector_strength: float) -> float:
    """Fringe visibility read off the pattern itself: (max − min) over a phase sweep."""
    pattern = fringes(np.linspace(0, 2 * np.pi, 2001), detector_strength=detector_strength)
    return float(pattern.max() - pattern.min())


@pytest.mark.parametrize("strength", [0.0, 0.4, np.pi / 4, 1.0, np.pi / 2, 2.5, np.pi])
def test_ti2_visibility_squared_plus_distinguishability_squared_is_one(
    strength: float,
) -> None:
    """TI2: V² + D² = 1 to 1e-12.

    Complementarity as a conservation law rather than a slogan. Fringe sharpness and
    which-path knowledge are not two effects competing — they are one resource, and every
    bit of which-path information is paid for out of the interference.
    """
    v = visibility(strength)
    d = distinguishability(strength)

    assert v**2 + d**2 == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("strength", [0.0, 0.4, np.pi / 4, 1.0, np.pi / 2, 2.5, np.pi])
def test_ti2_the_predicted_visibility_is_the_one_the_fringes_show(strength: float) -> None:
    """TI2: the V in that law is not a definition — it is measurable off the pattern."""
    assert measured_visibility(strength) == pytest.approx(visibility(strength), abs=1e-6)


def test_ti2_partial_information_gives_partial_fringes() -> None:
    """TI2: which-path knowledge is not a switch. Turn the detector part way up and the
    fringes fade part way out."""
    strengths = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    visibilities = [measured_visibility(s) for s in strengths]

    assert visibilities == sorted(visibilities, reverse=True)
    assert visibilities[0] == pytest.approx(1.0, abs=1e-6)
    assert visibilities[-1] == pytest.approx(0.0, abs=1e-6)


# ---- TI3: Elitzur–Vaidman ------------------------------------------------------


def test_ti3_a_live_bomb_is_found_a_quarter_of_the_time_without_exploding() -> None:
    """TI3: exploded 1/2, found 1/4, inconclusive 1/4, each to 1e-12.

    Interaction-free measurement. In the "found" branch no photon was ever absorbed, and
    yet the bomb's presence is certain — because the interference that kept port 1 dark
    required the two arms to remain indistinguishable, and a live bomb ends that.
    """
    probabilities = bomb_probabilities()

    assert probabilities["exploded"] == pytest.approx(0.5, abs=1e-12)
    assert probabilities["found"] == pytest.approx(0.25, abs=1e-12)
    assert probabilities["inconclusive"] == pytest.approx(0.25, abs=1e-12)


def test_ti3_a_dud_bomb_can_never_send_the_photon_to_port_one() -> None:
    """TI3: with nothing recording the path, the interference is intact and port 1 is
    dark. That is what makes a port-1 click *proof* of a live bomb."""
    probabilities = bomb_probabilities(live=False)

    assert probabilities["found"] == pytest.approx(0.0, abs=1e-12)
    assert probabilities["inconclusive"] == pytest.approx(1.0, abs=1e-12)


def test_ti3_running_the_test_reproduces_those_proportions() -> None:
    """TI3: the sampled experiment matches the analytic prediction."""
    outcomes = Counter(bomb_test(seed=s).outcome for s in range(4000))

    assert outcomes["exploded"] / 4000 == pytest.approx(0.5, abs=0.03)
    assert outcomes["found"] / 4000 == pytest.approx(0.25, abs=0.03)
    assert outcomes["inconclusive"] / 4000 == pytest.approx(0.25, abs=0.03)


def test_ti3_a_dud_is_never_reported_as_found() -> None:
    for seed in range(300):
        result = bomb_test(live=False, seed=seed)
        assert result.outcome == "inconclusive"
        assert not result.exploded


def test_ti3_an_exploded_bomb_reports_no_port() -> None:
    exploded = next(bomb_test(seed=s) for s in range(50) if bomb_test(seed=s).exploded)
    assert isinstance(exploded, BombResult)
    assert exploded.outcome == "exploded"
    assert exploded.port == -1


# ---- TI4: more than two paths --------------------------------------------------


def test_ti4_every_path_count_still_peaks_at_one() -> None:
    """TI4: with no phase difference all the amplitudes line up, whatever the path count."""
    for path_qubits in (1, 2, 3, 4):
        assert n_path_fringes(path_qubits, [0.0])[0] == pytest.approx(1.0, abs=1e-12)


def test_ti4_the_peak_narrows_as_paths_are_added() -> None:
    """TI4: two paths give a broad cosine, eight give a spike. This is the mechanism the
    QFT runs on — a quantum algorithm arranges for the wrong answers to cancel."""
    at_fixed_offset = [float(n_path_fringes(n, [0.5])[0]) for n in (1, 2, 3, 4)]

    assert at_fixed_offset == sorted(at_fixed_offset, reverse=True)
    assert at_fixed_offset[0] > 0.9  # two paths: barely off the peak
    assert at_fixed_offset[-1] < 0.1  # sixteen paths: already cancelled


def test_ti4_two_paths_reproduces_the_ordinary_interferometer() -> None:
    """TI4: one path qubit is the Mach-Zehnder of TI1, so the two agree."""
    phases = np.linspace(-np.pi, np.pi, 17)
    assert n_path_fringes(1, phases) == pytest.approx(fringes(phases), abs=1e-12)


# ---- Stern-Gerlach filter chains ------------------------------------------------


def test_a_second_filter_along_the_same_axis_blocks_nothing() -> None:
    """Feynman's Volume III opening. Filtering for spin-up along z twice in a row is no
    more selective than filtering once: the first filter already made it certain."""
    assert filter_chain("zz", 500, seed=1) == [500, 500, 500]


def test_a_filter_along_a_new_axis_blocks_half() -> None:
    """"Definitely up along z" says nothing at all about x."""
    counts = filter_chain("zx", 2000, seed=2)
    assert counts[1] == 2000
    assert counts[2] / 2000 == pytest.approx(0.5, abs=0.05)


def test_a_third_filter_finds_atoms_the_first_one_had_excluded() -> None:
    """The strange one. The third filter is along z — the axis already filtered for — and
    yet half the atoms now fail it. Measuring x did not merely select; it destroyed the z
    information the first filter had established."""
    counts = filter_chain("zxz", 2000, seed=3)

    assert counts[2] / counts[1] == pytest.approx(0.5, abs=0.05)
    assert counts[3] / counts[2] == pytest.approx(0.5, abs=0.05)


def test_the_y_axis_behaves_like_x_against_z() -> None:
    counts = filter_chain("zy", 2000, seed=4)
    assert counts[2] / 2000 == pytest.approx(0.5, abs=0.05)


def test_an_unknown_filter_axis_is_refused() -> None:
    with pytest.raises(ValueError, match="'x', 'y' or 'z'"):
        filter_chain("zq", 10)


def test_a_single_filter_passes_an_unmeasured_beam_untouched() -> None:
    assert filter_chain("z", 100, seed=5) == [100, 100]
