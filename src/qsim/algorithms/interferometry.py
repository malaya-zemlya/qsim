"""Interferometers — the experiments quantum mechanics is actually about.

**The physical fact this module makes concrete:** amplitudes add, probabilities do not.
Send a photon into a Mach–Zehnder interferometer and both of its 50/50 beam splitters
behave perfectly fairly, yet one output port can be *completely dark* — because the two
amplitudes arriving there cancel. There is no way to tell that story with probabilities,
which only ever add up.

The translation that makes all of this run on a qubit simulator:

===========================  ==========================================
apparatus                    qsim
===========================  ==========================================
a 50/50 beam splitter        ``H``
a phase shifter in one arm   ``Rz(path, theta=phi)``
which path the photon took   the state of one qubit, ``|0⟩`` or ``|1⟩``
a which-path detector        ``CNOT(path, detector)``
a *partial* detector         ``with qc.control(path): Ry(detector, theta)``
the two output ports         the two outcomes of measuring the path qubit
===========================  ==========================================

So a Mach–Zehnder interferometer is ``H``, then a phase, then ``H`` — which means the
H-sandwich from notebook 04 was one all along. Nothing in this module needed any new
machinery; the whole file is Phase 1 and Phase 2 gates wearing optical labels.

Every function here builds its own circuit, following the pattern of ``chsh.py``: these
are experiments to run and read, not parts to compose.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from qsim.circuit import Circuit
from qsim.gates import CNOT, H, Ry, Rz


def mach_zehnder(phase: float, *, detector_strength: float = 0.0) -> float:
    """Probability that the photon leaves a Mach–Zehnder interferometer by port 0.

    With no detector this is cos²(φ/2): at φ=0 the photon *always* takes port 0, and at
    φ=π it always takes port 1 — even though each beam splitter sends it both ways.

    ``detector_strength`` is an angle. At 0 the detector learns nothing and the fringes
    are perfect; at π it learns which arm with certainty and the fringes vanish
    completely, leaving a flat 1/2. In between it learns *something*, and the fringes
    fade in proportion — which-path information is not an on/off switch.
    """
    qc = Circuit(name="mach-zehnder")
    path = qc.alloc("path")

    H(path)  # first beam splitter: one photon, both arms
    Rz(path, theta=phase)  # a phase shifter in one arm

    if detector_strength:
        detector = qc.alloc("detector")
        # The detector rotates only in the branch where the photon took arm 1, so its
        # final state is correlated with the path. Nobody has to *look* at it.
        with qc.control(path):
            Ry(detector, theta=detector_strength)

    H(path)  # second beam splitter: the two arms meet and interfere

    # The probability the path qubit reads 0, with the detector (if any) ignored — which
    # is what a detector at port 0 would actually count.
    return float(qc.inspect.reduced_density_matrix([path])[0, 0].real)


def fringes(
    phases: Sequence[float] | np.ndarray, *, detector_strength: float = 0.0
) -> np.ndarray:
    """``mach_zehnder`` across a sweep of phases — the interference pattern."""
    return np.array([mach_zehnder(p, detector_strength=detector_strength) for p in phases])


def visibility(detector_strength: float) -> float:
    """How sharp the fringes are, from 1 (perfect) to 0 (gone).

    Equal to the overlap between the two detector states, cos(θ/2): the fringes survive
    exactly to the extent that the detector *failed* to distinguish the arms.
    """
    return float(abs(np.cos(detector_strength / 2)))


def distinguishability(detector_strength: float) -> float:
    """How well the detector's final state reveals which arm the photon took, 0 to 1.

    Together with :func:`visibility` this obeys **V² + D² = 1** for a pure state — the
    quantitative form of complementarity. Fringe sharpness and which-path knowledge are
    not two separate effects competing for attention; they are one resource, split.
    """
    return float(abs(np.sin(detector_strength / 2)))


@dataclass(frozen=True)
class BombResult:
    """One run of :func:`bomb_test`."""

    #: ``"exploded"``, ``"found"`` (a live bomb detected without setting it off), or
    #: ``"inconclusive"`` (the run tells you nothing; try again with another photon).
    outcome: str
    exploded: bool
    #: Which output port the photon left by, if it survived.
    port: int


def bomb_test(*, live: bool = True, seed: int | None = None) -> BombResult:
    """The Elitzur–Vaidman bomb tester: find a live bomb without setting it off.

    The setup: a crate of bombs, each triggered by absorbing a single photon. Some are
    duds — their trigger is broken, so a photon passes straight through. You want a bomb
    you *know* is live. Classically this is hopeless: the only way to learn that a
    trigger works is to trigger it.

    Put the bomb in one arm of a Mach–Zehnder interferometer. A live bomb absorbs any
    photon in its arm, which means it registers which path was taken — so it destroys the
    interference, exactly as an ordinary which-path detector does. A dud registers
    nothing, the interference survives, and port 1 stays dark.

    Therefore **a click at port 1 can only happen if the bomb is live** — and in that run
    the photon went the other way and the bomb never absorbed it. That happens 1/4 of the
    time. Half the time the bomb explodes; a quarter of the time you learn nothing and can
    try again with the next photon.

    Nothing in the found-it branch "touched" the bomb. What changed was not the photon's
    path but what was *possible*: the interference that kept port 1 dark required the two
    arms to stay indistinguishable, and the bomb's mere presence ended that.
    """
    qc = Circuit(name="bomb-test", seed=seed)
    path = qc.alloc("path")
    bomb = qc.alloc("bomb")

    H(path)
    if live:
        # A live bomb learns whether the photon came down its arm. That is a CNOT: the
        # same operation as any other which-path detector.
        CNOT(path, bomb)
    H(path)

    # Read the bomb first: did it absorb the photon?
    exploded = qc.measure(bomb) == 1
    if exploded:
        return BombResult(outcome="exploded", exploded=True, port=-1)

    port = qc.measure(path)
    # Port 1 is unreachable when the interference is intact, so a click there proves the
    # bomb is live.
    outcome = "found" if port == 1 else "inconclusive"
    return BombResult(outcome=outcome, exploded=False, port=port)


def bomb_probabilities(*, live: bool = True) -> dict[str, float]:
    """The exact outcome distribution behind :func:`bomb_test`, without sampling.

    For a live bomb: exploded 1/2, found 1/4, inconclusive 1/4. For a dud: inconclusive
    with certainty, because with the interference intact port 1 is dark.
    """
    qc = Circuit(name="bomb-probabilities")
    path = qc.alloc("path")
    bomb = qc.alloc("bomb")

    H(path)
    if live:
        CNOT(path, bomb)
    H(path)

    # psi[path, bomb], so index [p, 1] is "the bomb absorbed the photon".
    probabilities = np.abs(qc.inspect.state_tensor()) ** 2
    return {
        "exploded": float(probabilities[0, 1] + probabilities[1, 1]),
        "found": float(probabilities[1, 0]),
        "inconclusive": float(probabilities[0, 0]),
    }


def n_path_fringes(path_qubits: int, phases: Sequence[float] | np.ndarray) -> np.ndarray:
    """Interference across 2**``path_qubits`` paths instead of two.

    Each extra qubit doubles the number of routes the photon can take. Give path *k* a
    phase of k·φ and the amplitudes at the output add like a geometric series: they all
    line up at φ=0, and cancel more and more sharply as φ moves away. Two paths give a
    broad cosine; eight give a narrow spike.

    This is the mechanism the quantum Fourier transform runs on (Phase 3). A quantum
    algorithm is largely the art of choosing phases so that the wrong answers cancel and
    the right one survives — the same cancellation, aimed.
    """
    results = []
    for phase in phases:
        qc = Circuit(name="n-path")
        paths = qc.register(path_qubits, name="p")
        for q in paths:
            H(q)
        # Qubit i of the register carries place value 2**(n-1-i), since reg[0] is the
        # most significant bit. Giving each its own phase makes path k pick up k*phase.
        for i, q in enumerate(paths):
            Rz(q, theta=phase * 2 ** (path_qubits - 1 - i))
        for q in paths:
            H(q)
        # The photon "arrives" if every path qubit reads 0 again.
        results.append(float(qc.inspect.probabilities()[0]))
    return np.array(results)


def filter_chain(axes: str, shots: int, *, seed: int | None = None) -> list[int]:
    """Stern–Gerlach filters in series: how many atoms survive each stage.

    Feynman opens Volume III with this. A Stern–Gerlach apparatus splits a beam by spin
    along some axis; block one output and you have a *filter*, and what emerges is
    definitely spin-up along that axis. Chain them:

    - ``"zz"`` — filter along z, then z again. Everything survives; the second filter
      learns nothing new.
    - ``"zx"`` — filter along z, then x. Half survive: "definitely up along z" says
      nothing about x.
    - ``"zxz"`` — and now the strange one. The third filter is along z again, the axis we
      already filtered for, and yet half the atoms fail it. The x-filter did not just
      *select*; it destroyed the z information that the first filter had established.

    Returns the count surviving each stage, starting from ``shots`` atoms. Implemented by
    actually measuring and discarding, because a filter *is* "measure, then throw away
    what came out the wrong side" — and watching atoms get thrown away is the lesson.
    """
    if any(axis not in "xyz" for axis in axes):
        raise ValueError(f"filter axes must each be 'x', 'y' or 'z'; got {axes!r}")

    # Rotating the state and then measuring z is the same experiment as measuring along a
    # tilted axis — the same trick chsh.py uses, and the same reason.
    angles = {"z": 0.0, "x": np.pi / 2, "y": -np.pi / 2}

    passed_stage = [0] * len(axes)
    rng = np.random.default_rng(seed)

    # One atom, one circuit, walked through the whole chain — each atom is its own
    # experiment, and an atom blocked by a filter simply stops travelling.
    for _ in range(shots):
        qc = Circuit(seed=int(rng.integers(0, 2**32)))
        atom = qc.alloc("atom")  # leaves the oven spin-up along z
        for stage, axis in enumerate(axes):
            # Turn the apparatus, not the atom: rotating by -angle and then measuring z
            # is a measurement along the tilted axis (the same trick as in chsh.py).
            Ry(atom, theta=-angles[axis])
            if qc.measure(atom) == 1:
                break  # came out the blocked port
            passed_stage[stage] += 1
            # It passed, so it is now spin-up along this axis. Rotate back so the state
            # is expressed in the lab frame for the next filter.
            Ry(atom, theta=angles[axis])

    return [shots, *passed_stage]
