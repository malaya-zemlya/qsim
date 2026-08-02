# Notebook spec: `l1_vs_l2.ipynb` — Classical Probability and Quantum Mechanics, Side by Side

*(Owner-authored spec, recorded verbatim 2026-08-02; registered as demo E0 in `qsim-demo-notebooks.md`. The spec's pseudocode uses design-doc-era API — the real library's signatures win; the shape is the requirement.)*

**Build outcome (shipped 2026-08-02).** 72 cells, 3.0 s runtime, all assertions green. Substantive deviations, recorded so this spec doesn't mislead the next reader:

1. **Part 1(b) as written is false**: `COIN @ COIN == COIN` (an absorbing fixed point), so `S² = COIN` has the exact solution `S = COIN` and the residual is 0, not > 0.05. The notebook keeps the search, reports the true answer, and moves the bounded-away-from-zero assertion onto `S² = FLIP` (residual 1.0) — which is the √NOT contrast the surrounding text actually makes.
2. Part 1(a): uniform Dirichlet sampling finds *zero* stochastically-invertible matrices (measure-zero set); half the sample is drawn near permutations across 12 decades so the assertion has teeth.
3. Part 4(c): both parties rotated by the same θ leave the Bell correlator at 1 (rotational invariance); Alice uses −θ, Bob +θ to get the cos 2θ curve.
4. Part 4(d): the spec's settings are polarizer angles ⇒ `Ry(theta=-2·angle)`, and the CHSH minus sign lands on E(a,b′) with those four settings; the exhaustive classical enumeration uses the identical combination, so the bound comparison is fair.
5. `csim` is kernel *functions* (mirroring `qsim.state`), not a `CState` class — that is what makes the Part 0 empty-diff demonstration possible; the diff covers the 1-qubit kernel, with the controlled variant mirrored in shape and commented rather than diffed.
6. Part 1(c) is a static two-panel plot, not an animation (animations don't survive `jupyter execute`).
7. Bonus finding used to close the loop: `|Rx(0.9)|²` is exactly `LAZY(0.189)` from Part 1(c).

Track E, foundations. Buildable after Phase 2.5 (needs blocks, decoherence couplings, `inspect`; does not need QFT or arithmetic).

**Punchline (state in the first markdown cell):** Classical probability theory and quantum mechanics are the same design — states as norm-1 vectors, dynamics as norm-preserving linear maps, composition by tensor product — instantiated on the 1-norm and the 2-norm respectively. Every quantum phenomenon in this notebook is a consequence of that single substitution, and classical probability reappears inside QM as the fixed point of decoherence.

**Runtime budget:** < 30 s. Nothing above 4 qubits / 4 classical bits except the CHSH sampling. Seeded RNG throughout.

---

## Part 0 — A 60-line classical simulator: `csim`

First code cell defines (or imports from `qsim.contrib.csim` if promoted to the library later) a deliberately minimal mirror of the `qsim` API:

```
class CState:
    # probability vector over {0,1}^n, shape (2,)*n, real, >=0, sums to 1
    def __init__(n): p = zeros((2,)*n); p[0,...,0] = 1.0

def apply_stochastic(p, S, wire):
    # S: 2x2 column-stochastic matrix; same tensordot-on-axis mechanics as qsim
    # (deliberately identical code shape to qsim's gate application — put them
    #  side by side in adjacent cells and point at the one different line: none.
    #  The difference is only WHICH matrices are allowed in.)

def marginal(p, keep): ...          # sum over discarded axes  (mirror of partial trace)
def sample(p, shots, rng): ...

# classical "gates"
FLIP        = [[0,1],[1,0]]                      # permutation: the NOT gate
COIN        = [[.5,.5],[.5,.5]]                  # randomizer: "classical H"
LAZY(eps)   = [[1-eps, eps],[eps, 1-eps]]        # binary symmetric channel
CCOPY       = copy bit a onto bit b              # the classical CNOT (on the joint array)
```

Markdown: the state spaces. Classical n=1: the segment p ∈ [0,1] (draw it). Quantum n=1: the Bloch ball (draw it). A probabilistic bit is a 1-dimensional convex body; a qubit is a 3-dimensional one. Note for later: the segment's interior points decompose uniquely into endpoints; the ball's center decomposes along any diameter.

**Assertions:** norm preservation under 1000 random stochastic maps (L1) and unitaries (L2) respectively, to 1e-12.

---

## Part 1 — Reversibility: the simplex is rigid, the sphere is round

**Claim:** the only stochastic matrices with stochastic inverses are permutations; reversible classical dynamics is deterministic relabeling. The sphere admits a continuous group of reversible maps.

```
# (a) numerical search: sample 20,000 random 3x3 and 4x4 stochastic matrices
#     (Dirichlet columns), keep those whose inverse exists and is entrywise >= -1e-9
#     with columns summing to 1. Assert: every survivor is within 1e-6 of a
#     permutation matrix.  Plot: histogram of distance-to-nearest-permutation
#     for survivors vs all samples.

# (b) attempt "half a coin flip": find stochastic S with S @ S == COIN.
#     Solve numerically (least squares over stochastic matrices); show the
#     residual is bounded away from 0.  Contrast: half a NOT exists in qsim —
#     SX gate, SX@SX == X — and half of ANYTHING exists: U**0.5 via eigh.
#     Demonstrate: apply Rx(pi/2) twice, get X, to 1e-12.

# (c) the sweep: animate/plot the classical state under repeated LAZY(0.1)
#     (marches monotonically to the center, never returns) vs the Bloch vector
#     under repeated Rx(0.1) (orbits forever).
```

Markdown: this is the p-norm rigidity theorem in action — isometries of the 1-norm ball are permutations-and-signs; only p=2 gives a continuous isometry group. "There is no square root of a coin flip" is the slogan; the existence of √NOT is the first strictly-quantum fact in the notebook. Cross-reference: this is the same theorem that closed the "why squared amplitudes" form question (E3).

**Assertions:** (a) as stated; (b) residual > 0.05; (c) L1 distance to uniform is monotone nonincreasing under LAZY; Bloch norm constant to 1e-12 under Rx.

---

## Part 2 — √p: the Fisher–Rao sphere, and why it has no dynamics

**Claim:** the square-root embedding makes classical probability *geometrically* quantum — states on the positive orthant of the unit sphere, Bhattacharyya overlap as inner product — but no interesting dynamics is linear in √p, so the orthant is a statics-only theory.

```
# (a) embed: r_i = sqrt(p_i). Verify ||r||_2 == 1. For pairs (p, q):
#     inner(r_p, r_q) == sum sqrt(p_i q_i)  (Bhattacharyya / classical fidelity).
#     Disjoint-support distributions -> orthogonal vectors. Show 3-4 examples.

# (b) for the 3-outcome simplex, scatter-plot the embedded states: the positive
#     octant of S^2. (This IS the Bloch-sphere octant; draw both.)

# (c) the failure: for random stochastic S and random p, compare
#     sqrt(S @ p)   vs   any-linear-map @ sqrt(p).
#     Fit the best linear map M minimizing ||sqrt(S p) - M sqrt(p)|| over a
#     sample of p's; show residual is large unless S is a permutation.
#     Assert: residual < 1e-10 iff S is (numerically) a permutation.
```

Markdown: to make dynamics linear on amplitudes you must let amplitudes leave the positive orthant — signs, then complex phases. State the hierarchy explicitly: nonnegative amplitudes = classical statics; real signed amplitudes = real QM (Stueckelberg; experimentally falsified as a description of nature in Bell-type tests, arXiv 2101.10873 lineage — cite, don't develop); complex = QM. Interference requires leaving the orthant; that is Part 3.

**Assertions:** (a) to 1e-12; (c) as stated.

---

## Part 3 — Two paths: addition vs cancellation

**Claim:** stochastic paths always add; amplitudes can cancel. This is the resource behind every quantum algorithm.

Build the minimal interferometer in both theories, in parallel cells:

```
# classical:  COIN -> (phase has no meaning; insert nothing) -> COIN
#   P(0) = 0.5 always. There is no classical operation on one bit that makes
#   the two COINs undo each other while remaining non-deterministic.
#   (Permutations undo each other, but they're deterministic — Part 1.)

# quantum:    H -> Rz(phi) -> H
#   P(0) = cos^2(phi/2).  Sweep phi; plot both curves on one figure:
#   classical flat at 0.5, quantum swinging 0..1.
#   At phi=0: H then H == identity — a "coin flip" that un-flips. The
#   probability of the 1-outcome had two contributing paths with amplitudes
#   +1/2 and -1/2. Print the two path amplitudes explicitly by inspecting
#   the state between the gates.
```

Markdown: connect to Part 1's √NOT — interference is *why* continuous reversibility is possible: paths can destructively combine, so motion on the sphere never leaks norm. Optional half-cell: Elitzur–Vaidman bomb tester as a 3-line payoff of cancellation (flag as optional for the implementer).

**Assertions:** classical P(0) == 0.5 for all phi to 1e-12; quantum curve matches cos²(φ/2) to 1e-12; the two intermediate path amplitudes are ±1/2.

---

## Part 4 — Correlation vs entanglement: purity is the difference

**Claim:** the shared coin and the Bell state have *identical* joint statistics in the computational basis. They differ in (a) what pure means, (b) what happens when you rotate the question.

```
# (a) build both:
#     classical: COIN on bit a, CCOPY a->b.   Joint dist: [.5, 0, 0, .5]
#     quantum:   H(a); CNOT(a,b).             |amps|^2:   [.5, 0, 0, .5]
#     Assert elementwise equality. Local marginals both maximally mixed.

# (b) purity: the classical joint state is MIXED — exhibit its unique
#     decomposition [.5,0,0,.5] = .5*(00) + .5*(11): correlation as ignorance
#     of a definite fact. The quantum joint state is PURE (it's a vector);
#     inspect.entanglement_entropy([a]) == 1 bit while the JOINT entropy == 0.
#     Classical analogue impossible: joint Shannon entropy (=1) can never be
#     less than a marginal's (=1).  Assert the inequality violation:
#     S(joint)=0 < S(marginal)=1 in the quantum case.
#     This is Break #3 and the single most important cell of the notebook.

# (c) rotate the question: measure both systems in a tilted basis.
#     Quantum: apply Ry(theta) to each qubit before measuring — correlations
#     persist, E(a,b) = cos(2*theta)-like curve.
#     Classical: rotating the "measurement basis" is not even expressible —
#     the only available operations are stochastic pre-processing of definite
#     bits. Model the best classical strategy: local stochastic response to
#     the shared coin (this is exactly a local-hidden-variable model).

# (d) CHSH: quantum settings (0, pi/4) x (pi/8, 3pi/8) via Ry rotations +
#     measurement, sampled with shots=20000, seeded.
#     S_quantum ≈ 2*sqrt(2) (assert > 2.6).
#     Classical: exhaustively enumerate ALL deterministic local strategies
#     (16 of them) — max S = 2 exactly; stochastic strategies are convex
#     mixtures, so 2 bounds them too. Assert quantum > classical bound.
```

Markdown: classical correlation is always ignorance of a common cause (unique simplex decomposition, Part 0's segment vs ball); entanglement is correlation in a state of *maximal knowledge*, and (d) shows no ignorance-model reproduces it. Note honestly: CHSH here demonstrates the statistics; loopholes and spacelike separation are physics beyond a simulator's reach.

**Assertions:** as inlined above. (d) is the notebook's longest cell; keep shots moderate.

---

## Part 5 — The bridge: decoherence squares your unitaries

**Claim:** entrywise-squaring a unitary, S_ij = |U_ij|^2, yields a doubly stochastic matrix — and this is not a formal pun: it is *the dynamics you actually get* when dephasing follows every gate. Classical probability is the fixed point of decoherence.

```
# (a) formal check: for 500 random U in SU(2) and SU(4) (Haar via QR),
#     S = |U|^2 elementwise is doubly stochastic to 1e-12.

# (b) operational: single qubit, repeat K times:
#         apply U;  dephasing_coupling(q, fresh_env_qubit, theta)
#     For theta = pi (full dephasing): assert the qubit's diagonal populations
#     after k steps equal  S^k @ p0  from csim, to 1e-10, for k = 1..6.
#     The quantum simulator, run through a fully decohering environment,
#     IS the classical simulator with the squared matrix.

# (c) the dial: sweep theta from 0 to pi. Plot trajectory of the Bloch vector:
#     theta=0 great-circle orbit (Part 1c) deforming continuously into the
#     theta=pi monotone crawl to the center (Part 1c again, classical side).
#     One figure containing the whole notebook.
```

Markdown, closing: the two columns of the dictionary — L1/stochastic/correlation vs L2/unitary/entanglement — are not rival theories. The classical column is the quantum column as seen by an environment that records everything (B-track callback: the environment's records square the amplitudes, by the Born rule the notebook has been using all along). End with the honest asymmetry: the embedding of classical *into* quantum is exact (Part 5); the attempted embedding of quantum into classical fails at Parts 1, 3, and 4 — rigidity, cancellation, purity. Those three failures have names: no continuous reversibility, no interference, no entanglement. They are the subject of the rest of the library.

**Assertions:** (a), (b) as stated; (c) final Bloch norm at theta=pi below 0.02 after K steps, constant at 1.0 to 1e-12 at theta=0.

---

## Implementation notes for Claude Code

- `csim` mirrors `qsim`'s tensor mechanics on purpose. Where possible make the two `apply` functions textually identical; the pedagogy of Part 0 is "diff the simulators, find nothing but the matrix constraints."
- Part 4(d): enumerate deterministic strategies as functions {settings}->{outputs}; do not Monte-Carlo the classical bound — exact enumeration is the point (the bound is a theorem, not an estimate).
- Part 1(a)'s random-inverse search: use pseudo-inverse + feasibility check, not exact inversion, to dodge conditioning noise; document the tolerances.
- Haar sampling in 5(a): QR of complex Gaussian with phase correction (standard Mezzadri recipe), seeded.
- Every "Assert" line above goes in the final Assertions cell as well as inline, so the notebook fails loudly under `nbclient` if any claim rots.
- Figures: Parts 1c, 3, 4d, 5c are the four required plots; 5c is the cover image.
- Cross-references to add in other docs: E-track index in `qsim-demo-notebooks.md`; B1 (dial), A2 (marginals), E3 (p-norm rigidity).
