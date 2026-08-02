# Phase 2.25 — Interferometers: Feynman's experiments, run

**Read first:** `plans/master-plan.md` (Conventions), `qsim-design.md` §0 (orientation) and
§4.4 (where this leads). Requires Phase 2 complete. Uses **no new library machinery** —
that was verified before this plan was written.

**Goal:** the canonical interference experiments — the ones Feynman opens Volume III with
— as runnable code. This phase exists because the project is for learning quantum
mechanics, and interference is the thing quantum mechanics is *about*. Everything here is
already expressible with Phase 1–2 gates; what was missing was the framing.

**The identification that makes it work:** a Hadamard **is** a 50/50 beam splitter. `Rz`
is a phase shifter in one arm. A CNOT onto a spare qubit is a which-path detector, and a
*controlled rotation* onto that qubit is a detector you can turn down. So `H · Rz(φ) · H`
is a Mach–Zehnder interferometer, and the H-sandwich notebook 04 already built was one all
along without saying so.

**Files created:** `src/qsim/algorithms/interferometry.py`;
`tests/test_acceptance_ti1_ti4.py`; `notebooks/05-interferometers.ipynb`.

**Renumbering:** decoherence moves to notebook 06, QFT to 07, Shor to 08, Grover/DJ to 09.
Notebook 04's closing pointer is updated to match.

---

## 1. `algorithms/interferometry.py`

Self-contained demo functions that build their own circuits, following the precedent set
by `chsh.py` at the Phase 1.5 interface review — these are experiments to run and read,
not building blocks to compose.

```python
def mach_zehnder(phase: float, *, detector_strength: float = 0.0) -> float
    # P(the photon leaves by port 0). Beam splitter, phase shift in one arm,
    # optional which-path detector, second beam splitter.

def fringes(phases: Sequence[float], *, detector_strength: float = 0.0) -> np.ndarray

def visibility(detector_strength: float) -> float          # cos(theta/2)
def distinguishability(detector_strength: float) -> float  # sin(theta/2)

def bomb_test(*, live: bool = True, seed: int | None = None) -> BombResult
@dataclass(frozen=True)
class BombResult:
    outcome: str        # "exploded" | "found" | "inconclusive"
    exploded: bool
    port: int

def n_path_fringes(path_qubits: int, phases: Sequence[float]) -> np.ndarray
    # 2**path_qubits paths; the fringe sharpens as paths are added.

def filter_chain(axes: str, shots: int, *, seed: int | None = None) -> list[int]
    # Stern-Gerlach: a chain of filters along "z", "x", ... Returns how many atoms
    # survive each stage, starting from `shots`.
```

Implementation notes:

- **Measuring along a tilted axis** is the same trick CHSH used (`Ry(q, theta=-angle)`
  then measure Z); cross-reference `chsh.py` rather than re-explaining it.
- **`detector_strength`** is an angle θ. The detector is
  `with qc.control(path): Ry(detector, theta=theta)` — at θ=0 it learns nothing, at θ=π it
  learns everything, and in between it learns *some*. This partial coupling is the whole
  point: which-path information is not binary.
- **`filter_chain`** must actually measure and post-select, not compute a formula. A
  Stern–Gerlach filter *is* "measure, then throw away the atoms that came out the other
  port", and seeing the discarding happen is the lesson.
- **`bomb_test`** measures the bomb qubit first (did it explode?), then the output port.
  The docstring must be careful: nothing "interacts" with the bomb in the branch where it
  is found — the bomb's *presence* removes the interference that would have kept port 1
  dark, and that is what the detector click reports.

## 2. Acceptance tests — `tests/test_acceptance_ti1_ti4.py`

- **TI1 — fringes.** `mach_zehnder(phi)` equals cos²(φ/2) to 1e-12 across a sweep. At
  φ=0 the photon always leaves by port 0; at φ=π always by port 1. Both ports are dark
  half the time even though each beam splitter is 50/50 — that is interference.
- **TI2 — complementarity.** V² + D² = 1 to 1e-12 for detector strengths across [0, π],
  with V measured from the fringe (max − min over a fine phase sweep) and D computed from
  the detector states' overlap. The docstring states the physics: fringe visibility and
  which-path knowledge are not two effects, they are one resource split two ways.
- **TI3 — Elitzur–Vaidman.** With a live bomb: P(explodes) = 1/2, P(found safely) = 1/4,
  P(inconclusive) = 1/4, each to 1e-12. With a dud: port 1 *never* fires. So a port-1 click
  proves a live bomb that no photon touched.
- **TI4 — N paths.** Fringes from 1, 2 and 3 path qubits (2, 4, 8 paths); assert the peak
  stays at 1 and the peak *narrows* monotonically as paths are added — the mechanism the
  QFT (Phase 3) runs on.
- Plus unit tests: `filter_chain("zz")` passes everything through the second filter,
  `filter_chain("zx")` roughly halves it, and `filter_chain("zxz")` shows the "restored"
  beam — a third filter along z returns atoms the first one had already excluded.

## 3. Notebook — `05-interferometers.ipynb` ("Amplitudes, not probabilities")

1. What you will learn. Feynman's claim that the double slit contains "the only mystery",
   and that we can now run it.
2. **The beam splitter is a Hadamard.** Build a Mach–Zehnder by hand; sweep the phase;
   plot the fringes. Note that both detectors are dark half the time despite every beam
   splitter being 50/50 — amplitudes add, probabilities do not.
3. **Which path?** Add the detector; watch the fringes flatten. Sweep `detector_strength`
   and plot V and D on one axis, then V² + D² as a flat line at 1. Complementarity as a
   *conservation law*, not a slogan.
4. **The bomb.** Elitzur–Vaidman, run for many seeds, with a results table. Then the
   honest caveat: nothing touched the bomb, and yet its presence changed what the photon
   could do — the interference that kept port 1 dark needed both paths to stay
   indistinguishable.
5. **More paths.** 2 and 3 path qubits; watch the fringe sharpen. One line forward: this
   is what the QFT does, with the phases chosen so that one outcome survives.
6. **Stern–Gerlach filters.** Feynman's Volume III opening. `filter_chain("zz")` vs
   `"zx"` vs `"zxz"`: the third filter "restores" atoms the second one had no business
   letting through. Connect to notebook 03's rotated-basis measurements.
7. **Delayed choice.** Decide whether to erase the which-path record *after* the photon
   has passed the beam splitter, using `with qc.adjoint():` on the detector coupling.
   Fringes return. State plainly what this does and does not show: no retrocausality — the
   record simply never became a *record* until something read it. Point forward to
   notebook 06, where the eraser is done properly with an environment.
8. What you now know / next: decoherence, which is this same story with an environment
   too large to uncompute.

## Definition of done

- TI1–TI4 and the unit tests pass; all earlier tests pass; **100% coverage maintained**.
- pyright/ruff clean; notebooks 01–05 all execute.
- No new library machinery outside `algorithms/` — if this phase needs a new gate or
  kernel, something has been misunderstood.
- Notebook 04's forward pointer updated; master plan's phase index updated.
- Report "Decisions made".

## Interface decisions — resolved

The demo-module shape — self-contained functions building their own circuits — was
settled at the Phase 1.5 review and applied unchanged here, so this phase needed no fresh
interface review. The owner chose the experiment list directly: Mach–Zehnder and
complementarity as the core, plus all four optional demos (bomb tester, N-path,
Stern–Gerlach chains, delayed choice), and placed the material as its own phase before
decoherence rather than folding it into notebook 06.

## Deviations from this plan (as built)

- **`bomb_probabilities()` was added**, returning the exact outcome distribution without
  sampling. TI3's specified tolerance of 1e-12 is unreachable from `bomb_test`, which
  measures; the analytic function makes the acceptance test meaningful, and the notebook
  uses it to show 1/2, 1/4, 1/4 free of sampling noise. `bomb_test` is still tested
  statistically against it.
- **`filter_chain` was rewritten during development.** The first version re-ran the
  earlier filters on a fresh circuit at each stage, which double-filtered and gave
  `zxz` → 43 survivors out of 400 where ~105 was correct. It now walks each atom through
  the whole chain once, in one circuit, which is both right and a truer picture: an atom
  blocked by a filter simply stops travelling.
- **`fringes` and `n_path_fringes` accept `np.ndarray` as well as `Sequence[float]`**,
  since every caller passes `np.linspace`.
- **Notebook 04's closing section needed more than a filename change.** The renumbering
  first updated only the `05-decoherence.ipynb` references, leaving a heading that
  promised notebook 05 would be about the environment. It now carries a bridge paragraph
  introducing interferometers, then hands off to notebook 06 — caught by the notebook
  agent, not by the renumbering pass.
- Worth recording as a capability: **`with qc.adjoint():` wrapping `with qc.control(c):`
  works**, which is what makes the delayed-choice demo possible without defining a block.
  Nothing else in the notebooks or tests exercises that particular nesting.

## What this phase deliberately cannot do

Worth recording, because these look like they should be in scope and are not:

- **Hong–Ou–Mandel / two-photon interference.** Needs bosonic Fock space — a mode
  occupied by 0, 1 or 2 photons — which is not a qubit. A qubit simulator cannot express
  it without a different backend.
- **The literal double slit with wave packets in space.** Continuous variables, not
  qubits. The two-path version here captures the *logic* of the experiment (amplitudes
  add, which-path knowledge destroys the pattern) but not its spatial physics.
- **A genuine path integral over all paths.** N-path interference gestures at it; summing
  over continuously many paths is a different computation.
