# Phase 6 — Grover, Deutsch–Jozsa, and the remaining visualization surface

**Read first:** `plans/master-plan.md` (Conventions), `qsim-design.md` §8.5, §10 (circuit diagrams, entropy trace, widgets), T20–T21 in §9. Requires Phase 5 complete.

**Goal:** the two remaining algorithms, plus everything of §10 not built in earlier phases: circuit diagrams from history, the entropy trace, and the interactive widgets. Closes out Phases 0–6; afterwards the master plan's Phase 7/8 planning begins.

**Files created:** `src/qsim/algorithms/grover.py`, `src/qsim/algorithms/deutsch_jozsa.py`; `viz.circuit`, `viz.entropy_trace`, `viz.interact_grover`, `viz.interact_qft_comb` in `viz.py`; `tests/test_grover.py`, `test_deutsch_jozsa.py`, `test_viz_diagrams.py`, `tests/test_acceptance_t20_t21.py`; `notebooks/08-grover-deutsch-jozsa.ipynb`.

---

## 1. `algorithms/grover.py`

```python
def phase_oracle(marked: int, reg: Register) -> None
    # Flips the sign of exactly one basis state |marked>: X-conjugate the qubits
    # where marked's bit is 0, then a multi-controlled Z, then undo the X's.
    # Docstring: an "oracle" is just a subroutine that recognizes the answer —
    # here by tagging it with a minus sign no measurement could see directly.
def diffusion(reg: Register) -> None
    # H^n, phase-flip about |0...0>, H^n = reflection about the average amplitude.
def grover(n: int, marked: int, *, iterations: int | None = None,
           seed=None) -> GroverResult
    # iterations=None -> floor(pi/4 * sqrt(2^n)); GroverResult carries the
    # success probability read from inspect (exact), the iteration count, and
    # the measured outcome.
def success_probability(n: int, k: int) -> float
    # The prediction sin^2((2k+1)*theta), sin(theta)=1/sqrt(2^n) — pure math,
    # used by tests and the notebook to overlay theory on simulation.
```

Multi-controlled Z: `with control(*reg[:-1]): Z(reg[-1])` — Phase 2's arbitrary-arity control does this natively; comment that hardware would decompose it.

The rotation picture goes in the module docstring, drawn in words for the newcomer: the state lives in a 2D plane spanned by |answer⟩ and |everything else⟩; oracle = reflection about one axis, diffusion = reflection about the average; two reflections = a rotation by 2θ; after ~(π/4)√N steps you've rotated onto the answer — and *overshooting rotates past it* (T20 tests the decrease; the widget shows it).

## 2. `algorithms/deutsch_jozsa.py`

```python
def constant_oracle(value: int, x: Register, y: Qubit) -> None
def balanced_oracle(mask: int, x: Register, y: Qubit) -> None   # f(x) = parity(x & mask), mask != 0
def deutsch_jozsa(oracle, n: int, *, seed=None) -> bool          # True = constant
```

Docstring frames it honestly: a toy problem nobody needs solved, kept because it is the smallest complete example of the quantum trick — one query decides constant-vs-balanced where classical needs 2^(n-1)+1 in the worst case; introduce the |−⟩ phase-kickback trick (link back to phase estimation's kickback — same mechanism, simplest costume).

## 3. `viz.circuit(qc)` — text diagram from history

Render `qc.history` as a text diagram (returns `str`; also printable via `print(viz.circuit(qc))`):

```
q0: ─H─●──────M─
q1: ───X─●────M─
q2: ─────X──H───
```

Rules: one row per qubit (label from register names where known); time flows left; `●` controls, boxed letters for gates (`H`, `X`, `Rz(π/4)` with angle), `M` for measurement, `X` targets shown as `X` (`⊕` optional — ASCII fallback must exist); multi-qubit gates draw a vertical connector on the same column; blocks (from `@qsim.gate`) render collapsed as a named box spanning their qubits by default, with `expand_blocks=True` to inline. Column layout can reuse `depth()`'s greedy layering. Keep it plain-text (no matplotlib) — it must work in a terminal too; cap width with wrapping at ~120 columns (design doc allows text or matplotlib; text is the deliverable, matplotlib version optional and only if cheap).

Dead/deallocated qubits: rows end at deallocation (ancilla scopes visibly open and close — a nice teaching artifact; show `┤0⟩?` style verification marker at scope exit if easy, else omit).

## 4. `viz.entropy_trace(qc)`

Design doc §10: replay the recorded history from scratch on a fresh state, sampling `entanglement_entropy` after each gate for a chosen cut (default: each qubit against the rest, plotted as one line per qubit; accept a `cut=` argument for a specific bipartition). Slow is fine (says the design doc). Implementation: build a fresh `Circuit`, re-execute ops one at a time via the Phase 2 `_execute` funnel, inspect between ops. Measurement ops in history replay as *recorded outcomes* (deterministic replay — use the outcome stored in history, projecting accordingly; comment why replaying the coin-flip would desynchronize the trace).

## 5. Widgets (§10.2 remaining)

- `viz.interact_grover(n)` — iteration slider k; bar chart of success probability per basis state plus the theory curve `success_probability(n, k)` with the current k marked; the overshoot visible past the optimum.
- `viz.interact_qft_comb(N, a)` — slider over phase-register width t (or over "before/after modexp+QFT" steps); shows the comb sharpening at multiples of 2^t/r. Reuses Phase 5's period-finding circuit up to the pre-measurement state. This one is slow per step (~seconds); memoize per-t results in the closure.

Same conventions as `interact_dephasing` (Phase 2.5): lazy ipywidgets import, inner draw function testable headlessly, `# pragma: no cover` only on the `interact` wiring line.

## 6. Tests

**Acceptance (`test_acceptance_t20_t21.py`):**
- **T20:** n=6, one marked item, 6 iterations → success probability > 0.99 (read exactly from `inspect`, no sampling); and for k = 0..10 assert simulated probability matches `success_probability(6, k)` to 1e-9 — including the values *past* k=6 that go back down. Docstring: amplitude amplification is a rotation, not a ratchet; more is not better.
- **T21:** constant oracles (both values) → all-zeros with probability 1 (to 1e-12); balanced oracles (several masks) → all-zeros with probability 0 (< 1e-12). One classical-contrast sentence in the docstring (2^(n-1)+1 queries).

**Unit tests, as documentation:** `phase_oracle` flips exactly one sign (compare state vectors); `diffusion` maps the uniform state to itself (up to global phase — comment what "global phase is unobservable" means); `grover` with `iterations=0` = uniform distribution; `deutsch_jozsa` rejects `mask=0` as balanced-oracle input with a teaching message; `viz.circuit` golden tests — small circuits with exact expected multiline strings (Bell circuit, a controlled block, an ancilla scope; these goldens document the diagram language); `entropy_trace` on the Bell circuit returns [0, 1] after the two gates; widgets' draw functions run headless.

## 7. Notebook — `08-grover-deutsch-jozsa.ipynb` ("Amplifying the right answer")

1. What you will learn. Search framing: find 1 marked item among N with ~√N looks — and *why that's the best possible* is a real theorem (state without proof).
2. The oracle demystified: a minus sign nobody can see (`viz.amplitudes` with phase hue — bar heights identical, one bar's color flipped). Why a phase, not a flag: phases are what interference works on (echo notebook 01 §5).
3. One Grover iteration by hand on n=3: amplitudes after oracle, after diffusion — bar charts each step; the "reflection about the average" arithmetic done numerically in markdown.
4. The rotation picture; run to the optimum, then *past* it (`interact_grover`); T20's theory-vs-simulation overlay plot.
5. Deutsch–Jozsa as a curio: the smallest quantum speedup; phase kickback reprise; one-cell run of both oracle types.
6. Circuit diagrams: `viz.circuit` on this notebook's circuits and on Shor-15 (collapsed blocks — the algorithm's structure at a glance); `entropy_trace` of the Bell circuit and of period-finding (entanglement rising through modexp — T19's plot, now available to every user).
7. Series wrap-up: what you now know (the full arc: amplitudes → entanglement → decoherence → QFT → Shor → Grover); pointers to what Phase 7 would add (error correction, the emergence of the classical).

## Definition of done

- T20, T21 + unit tests pass; the **entire suite T1–T25/TB/TD is green**; **100% coverage**; pyright/ruff clean; notebooks 01–08 all execute (full `uv run jupyter execute notebooks/*.ipynb` — final integration check of the whole series).
- `viz.circuit` goldens committed; diagrams render sanely for every circuit in the notebooks.
- Report "Decisions made".

## Interface decisions to review with the owner (before building)

1. `viz.circuit` output sample (the golden strings above, mocked up) — does the diagram language read well? Unicode box-drawing vs pure ASCII default?
2. `GroverResult` fields; `grover()` returning a result object vs the measured int — recommend the result object (consistent with `ShorResult`); confirm.
3. `entropy_trace` default cut (per-qubit lines vs single chosen cut) — show both plots, pick one default.
4. Block rendering default in diagrams (collapsed vs expanded) — show Shor-15 both ways (expanded is thousands of columns; collapsed recommended).
