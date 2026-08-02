# `qsim` — Demo Notebooks Plan

Companion to `qsim-design.md`. Each entry below is one Jupyter notebook to be built once the relevant library phase exists. Notebooks are the pedagogical payload of the project; the library exists so these can exist.

**Conventions for the implementer (Claude Code):**

- Pseudocode below uses the design-doc API (`Circuit`, `Register`, `inspect.*`, blocks, combinators, `environment()`). Adapt to the real signatures as they settle; the *shape* of each demo is the requirement, not the exact calls.
- Every notebook: seeded RNG, runs top-to-bottom in < 30 s, ends with an **"Assertions"** cell re-checking its central claim numerically (so notebooks double as slow acceptance tests; run them in CI with `nbclient`).
- Markdown cells carry the physics narrative. Write them for a reader who knows linear algebra but is meeting the phenomenon for the first time. State the punchline of each notebook in its first cell.
- Plots: amplitude bars with phase-as-hue (`viz.amplitudes`), probability bars, Bloch spheres, entropy traces. Every notebook should have at least one plot; the plot usually *is* the argument.
- Prefer few qubits. Nothing here needs more than ~12.

---

## Track A — Seeing the state (after Phase 1)

### A1 — `one_qubit_playground.ipynb`
Single qubit under H, X, Z, Rx, Ry, Rz. Bloch vector after each gate; animate a continuous `Rx(theta)` sweep. Show that Z does nothing visible to a $|0\rangle$ but everything to a $|+\rangle$ — phases are invisible until interference.

```
qc = Circuit(1); q = qc.alloc()
for theta in linspace(0, 2π, 60):
    fresh circuit; H(q); Rz(q, theta); H(q)
    record P(0)   # plot P(0) = cos²(θ/2) — interference makes phase visible
```

### A2 — `entanglement_and_marginals.ipynb`
Bell state. Punchline: **the joint state is pure, the parts are maximally mixed.**

```
H(a); CNOT(a, b)
inspect.reduced_density_matrix([a])     # == I/2
inspect.bloch_vector(a)                 # == (0,0,0) — no local state at all
inspect.entanglement_entropy([a])       # == 1 bit
```
Contrast with the product state `H(a); H(b)`: same single-qubit marginals? No — Bloch vector of `a` is (1,0,0) there. Then GHZ: measure qubit 0, show conditional state of the rest collapses. End: sweep `Ry(a, θ); CNOT(a,b)` and plot entropy vs θ — entanglement as a continuous dial.

### A3 — `no_cloning_and_dead_qubits.ipynb`
The error-message tour. Deliberately trigger `NoCloningError` (`CNOT(a,a)`, `copy.copy(q)`), `DeadQubitError`, `DirtyAncillaError`; display each message. Narrative: each exception is a theorem wearing a stack trace.

### A4 — `quaternions_and_spin.ipynb` *(owner request, added after batch AB)*
Single-qubit rotations **are** unit quaternions — same group, half-angles included. The half-angle is *derived*, not asserted, via mirrors: a reflection is already a sandwich (v → n v n), two mirrors at angle α make a rotation by 2α, so the rotation quaternion is the Hamilton product of its two mirror normals and cos(θ/2) is the angle between the mirrors. Quantum coda: X/Y/Z/H are the Bloch sphere's π-rotation "mirrors" (H X H = Z is "reflect x onto z"), and composing two of them at axis-angle φ rotates by exactly 2φ — shown live with `within`. Hand-rolled numpy quaternions (Hamilton product commented); the exact dictionary $U = a\,I - i(b\,\sigma_x + c\,\sigma_y + d\,\sigma_z) \leftrightarrow q = a + bi + cj + dk$ verified numerically on qsim's own gate matrices (products match products); Bloch-vector rotation via `inspect.bloch_vector` matches quaternion conjugation $qvq^{-1}$. Then the double cover: sweep θ to 4π — the Bloch vector is 2π-periodic while the quaternion needs 4π, hitting −1 at 2π. Finale: the −1 is *physical* — `with control(c): Rx(q, 2π)` flips the control's phase and an H-sandwich measures it deterministically. Punchline: the sign quaternions carry silently, a qubit can cash out; cross-link A1's closing cell.

---

## Track B — Decoherence (after Phase 2.5)

### B1 — `decoherence_dial.ipynb`
TD1/TD2 as a story. Prepare $|+\rangle$, couple to environment at strength θ, plot Bloch-x and interference visibility vs θ; overlay the predicted $\cos(θ/2)$.

```
for theta in sweep:
    H(q); dephasing_coupling(q, env, theta); H(q)
    visibility[theta] = 2*abs(P(0) - 0.5)
```

### B2 — `quantum_eraser.ipynb`
TD3 as narrative: decohere fully (θ=π), show entropy = 1 bit and interference gone; then `with adjoint(): dephasing_coupling(...)` and show it all come back. Punchline cell: **decoherence is a fact about your bookkeeping, not about the state — if you kept the environment, nothing was lost.**

### B3 — `einselection.ipynb`
`pointer_coupling` through σz vs σx. For each, prepare states in both bases, plot which survives. Punchline: the classical-looking basis is *chosen by the interaction*, and this is why the everyday world has positions rather than superpositions of positions.

### B4 — `wigners_friend.ipynb`
Measurement modeled as a friend (record qubit) inside the box.

```
# friend "measures" q
CNOT(q, friend)
# 1. friend's marginal reproduces Born-rule collapse exactly
inspect.reduced_density_matrix([q])       # diagonal — looks collapsed from inside
# 2. Wigner uncomputes the friend
with adjoint(): CNOT(q, friend)
H(q)                                       # interference restored
```
Narrative: deferred measurement, von Neumann's movable cut, and why "when did the collapse happen?" has no operational answer. Cross-reference B2 — the friend is a dephasing environment with a name.

### B5 — `horizon.ipynb` *(needs the `escaped` marking — small addition to §4.4)*
Decoherence made *structurally* irreversible: mark an environment qubit as escaped (photon past the cosmological horizon); any later gate on it raises.

```
dephasing_coupling(q, env, π)
qc.escape(env)                 # point of no return
with adjoint(): dephasing_coupling(q, env, π)   # -> raises EscapedQubitError
```
Narrative: FAPP-irreversibility vs in-principle (Bell), horizons as the one non-movable cut (Bousso–Susskind), and the honest caveats (observer-dependence of horizons; Page-curve results). The eraser of B2 fails here not with a wrong answer but with an exception — the API distinction *is* the philosophical distinction.

---

## Track C — Interference at work (after Phase 3)

### C1 — `qft_gallery.ipynb`
QFT of basis states, of periodic states, of a Gaussian. The money plot: prepare $\sum_j |jr\rangle$ (period r), apply QFT, watch the flat input become a comb with peaks at multiples of $2^t/r$. Also: bit-reversal shown explicitly (`swap=False` vs `swap=True`), and QFT-vs-`np.fft` agreement.

### C2 — `phase_estimation_precision.ipynb`
φ exactly representable (3/8) vs not (1/3): sharp peak vs leakage sidelobes. Sweep register size t and plot estimate error vs t — the 2n+1 rule made visual. Then approximate QFT: fidelity vs truncation m, and where Shor-style estimation starts failing.

### C3 — `semiclassical_qft.ipynb`
Coherent vs Griffiths–Niu measured-feedback phase estimation, same seed structure, overlaid outcome histograms + TVD. Narrative: deferred measurement run in reverse — trading purity for qubits.

### C4 — `precision_and_conditioning.ipynb`
The float32/float64 demo (T17). Same period-finding state through QFT in both dtypes; scatter |amp64| vs |amp32| on log axes. Peaks agree to ~7 digits, valleys are relative-error garbage — and it doesn't matter. Narrative cells: unitarity ⇒ κ=1 ⇒ additive error; butterfly = pairwise summation; badly-conditioned outputs are exactly the ones with no probability weight.

---

## Track D — Shor (after Phase 5)

### D1 — `reversible_arithmetic.ipynb`
Cuccaro adder walked through on small registers, with a circuit diagram and an exhaustive truth-table check. Then the ancilla context manager: run the adder *without* the uncompute step and catch `DirtyAncillaError`. Show `gate_counts()` — the honest Toffoli price of `a^x mod N`.

### D2 — `period_finding_anatomy.ipynb`
Shor on N=15 opened up, one stage per section:
```
1. Hadamards on exponent register        -> flat superposition (plot)
2. modexp                                -> entropy(exponent : work) jumps to ~log2(r)  (entropy trace plot)
3. QFT on exponent register              -> comb at multiples of 2^t/r (plot)
4. measure; continued fractions          -> show convergents table, recovered r, factors
```
This is the library's centerpiece notebook.

### D3 — `uncomputation_or_death.ipynb`
T18 as narrative: period-finding with and without ancilla uncomputation (test-only escape hatch), side-by-side output distributions, peak-to-background ratio. Cross-reference B1: the dirty ancillas are an environment; the lost peak is lost visibility; this is the two-slit experiment inside an algorithm.

### D4 — `shor_end_to_end.ipynb`
`shor(15)`, `shor(21)`, full `ShorResult` trace including failures and retries (bad `a`, odd r, trivial gcd). Qubit and gate counts vs N. Closing section: what a permutation-matrix "modexp" would hide, and why T24 exists.

---

## Track E — Foundations (after Phase 5; library-complete material, except E0)

### E0 — `l1_vs_l2.ipynb` *(owner-specced in full — the spec is `plans/demo-l1-vs-l2-spec.md` and governs)*
Classical probability and QM side by side: the same design — norm-1 states, norm-preserving linear maps, tensor composition — on the 1-norm vs the 2-norm. A ~60-line notebook-local `csim` mirrors qsim's tensor mechanics (diff the simulators, find only the matrix constraints); simplex rigidity vs √NOT; the Fisher–Rao orthant as a statics-only theory; addition vs cancellation; shared coin vs Bell state with CHSH (purity is the difference); and the bridge back — |U|² is doubly stochastic and full dephasing after every gate *is* the classical simulator. **Buildable after Phase 2.5** (shipped), unlike the rest of Track E.

### E1 — `deferred_measurement.ipynb`
Every mid-circuit measurement in a small circuit replaced by CNOT-to-record; final statistics identical (TVD over seeded shots). Then the one thing deferral *loses*: conditional states. Narrative ties to B4 and C3 — the same principle in three costumes.

### E2 — `born_from_typicality.ipynb`
The single-sample question, honestly framed. Long sequence of measurements on identical qubits with P(1)=p; branch-count measure vs Born measure:

```
# within-branch frequency under Born weighting concentrates at p  (finite-frequency theorem)
# under uniform branch counting it concentrates at 1/2 — empirically wrong
plot both distributions of within-branch frequency for n = 4, 8, 12
```
Markdown must state the circularity plainly: "with high Born-measure" presupposes the measure. The notebook demonstrates the theorem *and* its philosophical limit. No pretending this is resolved.

### E3 — `gleason_teaser.ipynb`
Not a proof — an exhibit. For a qutrit (simulate a 3-level system as 2 qubits with one amplitude pinned, or just use raw numpy here): try to hand-construct a non-Born frame function (additive over orthonormal bases, noncontextual) and watch every attempt fail / collapse to tr(ρP). Numerically fit: random frame functions constrained to additivity are always quadratic forms. Companion: the p-norm cell — isometries of the p-norm sphere for p≠2 are finite (permutations+signs); only p=2 admits continuous reversible dynamics. Punchline: probabilities-on-subspaces + continuous reversibility ⇒ Born, by theorem, twice over.

### E4 — `entanglement_temperature.ipynb`
Restriction manufactures thermality.

```
H = transverse_field_ising(n=10, g=1.05)      # raw numpy, eigh
psi = ground_state(H)
rho = partial_trace(psi, keep=first_half)
K   = -logm(rho)                               # entanglement Hamiltonian
plot spectrum of K; compare rho to e^{-K}/Z (trivially exact) and to a fitted Gibbs state of the *physical* half-chain H at best-fit β
```
Then the thermofield double: $\sum_n e^{-βE_n/2}|n\rangle|n\rangle$, trace either side → exact Gibbs state. Sweep β, plot entropy of one side vs temperature. Narrative: Unruh in caricature — what lives behind a cut is thermal, temperature is entanglement with what you can't see, and the Wick-rotation pun $it \leftrightarrow β$ has structure behind it (KMS, modular flow), pointed to but not developed.

### E5 — `error_correction_teaser.ipynb` *(after Phase 7)*
3-qubit repetition code under stochastic X noise (trajectories): encode, apply noise with probability p per qubit, syndrome-extract onto ancillas, correct, decode. Plot logical error vs physical p; show the threshold-like crossover where encoding starts helping. Then show the code failing under Z noise — motivation for Shor-9/Steane, and for the whole field.

---

## Suggested build order

A1–A3 immediately after Phase 1 (they'll shake out the inspect API), B1–B4 after 2.5, C1–C4 after 3, D1–D4 after 5, then E. B5 and E5 need small library additions (`escape` marking; trajectories) — treat those as feature requests from the notebook, which is the right direction of pressure: demos drive the API.

## Out of scope (deliberately)

- Anything requiring >12 qubits or >30 s runtime
- VQE / QAOA / variational anything — different project
- Real-hardware comparisons
- A notebook "explaining interpretations" without a computation at its center. Every foundations claim here earns its place by being demoable; the ones that aren't (Born-rule derivations, one true interpretation) are flagged as open, not narrated as settled.
