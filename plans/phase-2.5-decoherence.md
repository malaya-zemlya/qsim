# Phase 2.5 — Decoherence and mixed states by dilation

**Read first:** `plans/master-plan.md` (Conventions), `qsim-design.md` §4.4 (all of it — it is the spec), TD1–TD7 in §9, §10.2 (the dephasing widget). Requires Phase 2 complete.

**Goal:** noise and decoherence with zero new simulator machinery — just extra qubits marked "environment" and small unitary couplings. The design doc's central sentence, which goes in the module docstring and the notebook: *decoherence is not something that happens to a state; it is what a subsystem looks like when you decline to track the rest of the world.*

**Files created:** `src/qsim/decoherence.py`, additions to `circuit.py` (environment marking) and `inspector.py` (system-vs-environment views), `viz.interact_dephasing` in `viz.py`; `tests/test_decoherence.py`, `tests/test_acceptance_td1_td7.py`; `notebooks/05-decoherence.ipynb`.

---

## 1. Environment marking (`circuit.py`, `inspector.py`)

```python
env = qc.environment(2, name="E")   # Register; qubits flagged as environment
```

Critical design point (design doc §4.4, verbatim constraint): **`environment()` traces nothing out.** The environment qubits stay in the tensor, entangled, forever; the global state remains pure. The flag only tells the Inspector which qubits count as "the system" so diagnostics default to the reduced view. Implementation: a `_is_env: set[int]` of qubit ids on the circuit.

Inspector additions:

```python
qc.inspect.system_density_matrix() -> np.ndarray   # reduced_density_matrix over non-env qubits
qc.inspect.system_entropy() -> float               # entanglement entropy of that cut
qc.inspect.coherence(q) -> float                   # |rho_01| of q's reduced 1-qubit state
```

`coherence` docstring introduces the word: the off-diagonal element of a qubit's density matrix measures how much superposition survives; 0.5 for a fresh |+⟩, 0 for a coin flip. Bloch-vector connection: |ρ₀₁| = √(x²+y²)/2 — the shadow of the Bloch vector on the equatorial plane.

## 2. `decoherence.py` — the couplings

Module docstring: the "decline to track" sentence above, plus Stinespring in plain words — every noise process can be written as (a) a perfectly reversible interaction with extra qubits you then (b) refuse to look at; the library implements (a) literally and makes (b) a *choice of view*, which is why the eraser (§4) works. **Each coupling's docstring must state the Kraus operators it realizes** (design doc requirement) — write them as small explicit matrices in the docstring, with one sentence: "Kraus operators are the standard textbook way to write noise; TD6 checks that tracing out our environment gives exactly this channel."

```python
def dephasing_coupling(q: Qubit, env: Qubit, theta: float) -> None
def amplitude_damping_coupling(q: Qubit, env: Qubit, theta: float) -> None
def depolarizing_coupling(q: Qubit, env: Register, p: float) -> None   # len(env) == 2
def pointer_coupling(q: Qubit, env: Qubit, theta: float, basis: str = "z") -> None
```

All are `@qsim.gate` blocks (so they can be adjointed — the eraser depends on it) built from Phase 1/2 gates only. Constructions, from the design doc with implementation detail added:

- **Dephasing:** `with control(q): Ry(env, theta=theta)`. Result: |1⟩'s branch tags the environment by cos(θ/2)|0⟩+sin(θ/2)|1⟩ while |0⟩'s leaves it at |0⟩. Comment the traced-out effect: populations untouched; ρ₀₁ shrinks by cos(θ/2); at θ=π the environment has recorded which-path info perfectly and coherence is zero. Kraus: K₀ = diag(1, cos(θ/2)), K₁ = diag(0, sin(θ/2)).
- **Amplitude damping:** excitation transfer: `with control(q): Ry(env, theta=theta)` then `CNOT(env, q)`. Verify in a comment the mapping |1,0_E⟩ → cos(θ/2)|1,0_E⟩ + sin(θ/2)|0,1_E⟩, |0,0_E⟩ untouched — the qubit decays toward |0⟩, energy "leaking" into the environment. γ = sin²(θ/2); Kraus: K₀ = diag(1, √(1−γ)), K₁ = [[0, √γ],[0,0]].
- **Depolarizing:** prepare the 2-qubit env in √(1−p)|00⟩ + √(p/3)(|01⟩+|10⟩+|11⟩) (two `Ry` + one controlled-`Ry` with angles derived in a comment — solve the three amplitudes explicitly), then apply env-controlled X (env=01), Y (env=10), Z (env=11). 0-controls via X-conjugation of the control qubit (X before and after). Kraus: √(1−p)·I, √(p/3)·{X, Y, Z}.
- **`pointer_coupling`:** dephasing conjugated into a chosen basis: for `basis="x"`, `H(q)`, dephase, `H(q)`; for `"y"` use the S/H conjugation; `"z"` is plain dephasing. Docstring carries the einselection point (design doc): *which states survive decoherence is decided by how the environment couples, not by anything intrinsic to the state* — coupling through Z makes {|0⟩,|1⟩} the robust "pointer" states; through X makes {|+⟩,|−⟩} robust. This is the beginning of the answer to "why does the world look classical" (notebook 05 §6, and Phase 7's Darwinism demo later).

## 3. `viz.interact_dephasing` (§10.2)

```python
def interact_dephasing() -> None   # ipywidgets slider over theta in [0, pi]
```

Builds a fresh |+⟩ + environment circuit per slider value; draws side-by-side: the Bloch vector of the system qubit (shrinking along x toward the origin) and the two-slit visibility curve cos(θ/2) with a marker at the current θ. Import `ipywidgets` lazily with a clear `ImportError` message ("dev dependency — run uv sync"). Keep it ~30 lines; it is a demo, not a framework. (Coverage: the figure-drawing inner function is testable headlessly by calling it directly with a θ value; the `interact` wiring line may need a `# pragma: no cover` — justify it in the report.)

## 4. Acceptance tests — `tests/test_acceptance_td1_td7.py`

Implement TD1–TD7 exactly per design doc §9. Implementation notes:

- **TD1:** |+⟩, couple at θ, assert Bloch x = cos(θ/2) to 1e-12 for θ ∈ {0, π/4, π/2, 3π/4, π}; at π the whole vector is at the origin (maximally mixed).
- **TD2 (the two-slit test):** H → couple → H; visibility = P(0) − P(1) follows cos(θ/2). Docstring: this *is* the double-slit experiment — the H's are the beam splitter, the environment is anything that could tell which path was taken; knowing the path kills the fringes.
- **TD3 (eraser):** couple at π, assert coherence ≈ 0 and system entropy ≈ 1 bit; `with adjoint(): dephasing_coupling(q, e, theta=pi)` — wait: adjoint of a *block call* — use `dephasing_coupling.adjoint()(q, e, theta=pi)` or the scope form; both must work (this cross-checks Phase 2's param capture). Assert coherence back to 0.5 within 1e-12, entropy < 1e-12, final H gives |0⟩ deterministically. Docstring: **decoherence is reversible if you kept the environment** — and cross-reference T18: a dirty ancilla is an environment; uncomputation is erasure.
- **TD4:** dephasing leaves ρ diagonal unchanged to 1e-12 while |ρ₀₁| decays; amplitude damping changes both (assert ρ₁₁ decreased).
- **TD5 (einselection):** with basis="z": computational populations preserved, |+⟩/|−⟩-basis coherence destroyed; basis="x": exactly reversed. Assert both by computing the reduced ρ in both bases (conjugate with H in the test).
- **TD6 (the most important):** for each coupling and several parameter values, prepare a seeded random 1-qubit state, run the dilation, get `system_density_matrix()`; independently compute Σ K ρ Kᵀ* in the test file with the explicit 2×2 Kraus matrices from the docstrings; `np.allclose(..., atol=1e-12)`. Docstring: this test *proves* the dilations are the channels they claim to be — the whole phase rests on it.
- **TD7:** for 20 seeded random couplings/states, S(system) == S(environment) to 1e-10. Docstring: for a globally pure state the two halves of any cut are exactly equally mixed — a surprising symmetry and a free audit of the partial trace.

Unit tests (`test_decoherence.py`) for coverage-as-documentation: `environment()` qubits excluded from `system_density_matrix`; env qubits still count in `n_qubits` (nothing was traced away); `depolarizing_coupling` rejects `len(env) != 2`; `pointer_coupling` rejects unknown basis with a message listing valid ones; coherence of |0⟩ is 0 and of |+⟩ is 0.5; `interact_dephasing`'s draw function runs headless.

## 5. Notebook — `05-decoherence.ipynb` ("Why the world looks classical")

The conceptual heart of the whole library; take the space it needs.

1. What you will learn. The puzzle stated honestly: if superposition is real, why do we never see a coffee cup in two places?
2. A qubit meets a bystander: |+⟩, one environment qubit, `dephasing_coupling` at small θ. Show the global state's `ket()` (still pure! four terms) vs the system qubit's Bloch vector (shrunk). Two views of one state.
3. Turn the knob: the θ-sweep plot (coherence vs θ), then `viz.interact_dephasing()` — play with it.
4. The two-slit experiment in three gates (TD2's circuit narrated cell by cell): fringes at θ=0, gone at θ=π. Markdown aside: this is *the* experiment — Feynman's "only mystery" — and the environment qubit is the "detector at the slits."
5. Information, not disturbance: emphasize the population rows of ρ never changed (TD4 live) — the environment didn't *kick* the qubit; it *learned* about it. Decoherence = leaked information.
6. Einselection: pointer_coupling in z vs x; the interaction chooses which basis survives. "Position looks classical because interactions couple through position."
7. The quantum eraser (TD3 live): uncompute the coupling, coherence returns exactly. Punchline in markdown: the mixedness was never *in* the qubit — it was in our refusal to look at the whole. Cross-ref notebook 04's dirty ancilla: same phenomenon, and in Phase 5 it will decide whether Shor's algorithm works.
8. What you now know / next (the QFT — the wave-analysis tool the big algorithms are built on).

## Definition of done

- TD1–TD7 + unit tests pass; all earlier tests pass; **100% coverage maintained**; pyright/ruff clean; notebook 05 executes headless (widget cell included).
- Every coupling docstring states its Kraus operators; TD6 checks them independently.
- Report "Decisions made" (including any `# pragma: no cover` with justification).

## Interface decisions to review with the owner (before building)

1. Coupling parameter style: raw `theta` everywhere vs. the channel-native parameter (e.g. damping probability `gamma` for amplitude damping, with θ derived) — show both call styles; recommend θ for uniformity with a `gamma=`-style note in docstrings.
2. `interact_dephasing()` layout mock (two panels) — worth a quick look before it's built.
3. Whether `qc.inspect.bloch_vector(q)` should *automatically* use the reduced state when environments exist — it already does by construction (reduced ρ of one qubit); confirm the owner understands the default-view semantics of `environment()` with a two-line example.
