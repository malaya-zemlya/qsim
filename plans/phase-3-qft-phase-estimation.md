# Phase 3 — QFT and phase estimation

**Read first:** `plans/master-plan.md` (Conventions), `qsim-design.md` §8.1, §8.2, T11–T15 in §9. Requires Phase 2.5 complete. (T17 is deliberately deferred to Phase 5 — it needs the period-finding circuit; the master plan records this deviation.)

**Goal:** the Quantum Fourier Transform as a circuit of H and controlled-phase gates, exact and approximate, plus both flavors of phase estimation. This is the measuring instrument Shor's algorithm reads its answer through.

**Files created:** `src/qsim/algorithms/qft.py`, `src/qsim/algorithms/phase_estimation.py`; `tests/test_qft.py`, `tests/test_acceptance_t11_t15.py`; `notebooks/06-qft-phase-estimation.ipynb`.

---

## 1. `algorithms/qft.py`

```python
@qsim.gate
def qft(reg: Register, *, swap: bool = True, approx: int | None = None) -> None
@qsim.gate
def iqft(reg: Register, *, swap: bool = True, approx: int | None = None) -> None
```

(If Phase 2's `@qsim.gate` doesn't yet support keyword-only classical params cleanly, extend it — the capture machinery already stores params.)

Module docstring for the non-expert, in this order: (1) the DFT in one paragraph — "decompose a list of numbers into the frequencies it contains"; (2) the QFT is the same linear map applied to the 2^n amplitudes of a register; (3) the miracle and the catch — it runs in ~n² gates instead of ~2^n·n operations, *but* the frequencies end up encoded in amplitudes you cannot read out directly; algorithms must be designed so one frequency dominates (which is exactly what Shor's does). Include the defining formula from design doc §8.1.

Circuit (standard, textbook):

```python
n = len(reg)
for j in range(n):
    H(reg[j])
    for k in range(j + 1, n):
        m = k - j + 1                       # controlled rotation by 2*pi / 2^m
        if approx is not None and m > approx:
            continue                        # drop tiny rotations (see below)
        CPhase(reg[k], reg[j], theta=2 * np.pi / 2**m)
if swap:
    for i in range(n // 2):                 # undo the bit reversal (see below)
        SWAP(reg[i], reg[n - 1 - i])
```

Required explanatory comments (write them for a reader meeting this for the first time):
- Why H-then-phases works: after processing qubit j, its state is (|0⟩ + e^{2πi·0.j_j j_{j+1}…j_{n-1}}|1⟩)/√2 — each qubit ends up carrying one binary digit's worth of the input read as a binary *fraction*. Show the binary-fraction notation 0.b₁b₂… = b₁/2 + b₂/4 + … in the docstring.
- **Bit reversal:** the circuit naturally produces the output with qubit order reversed; the SWAP network fixes it. Keep `swap=True` the default, and the docstring notes you can pass `swap=False` to *see* the reversal (T11's test file demonstrates handling it by hand — design doc requirement).
- `iqft`: implement as `qft` structure with negated angles and reversed loop order (or, equivalently and preferably for teaching, `qft.adjoint()` internally — but then approx/swap params must round-trip through the adjoint machinery; choose whichever is cleaner in practice and document the equivalence; T12 pins correctness either way).

Document the two precision facts from design doc §8.1 (phase-register size for Shor; only O(log(t/ε)) distinct angles needed, error ~ t²·2^{−m}) in the docstring, each with a plain-language gloss.

## 2. `algorithms/phase_estimation.py`

```python
def phase_estimation(unitary: Block, target: Register, out: Register) -> None
def semiclassical_phase_estimation(unitary: Block, target: Register, t: int,
                                   *, circuit: Circuit) -> int
```

Docstring pitch (no QM assumed): a unitary's eigenvalues all have absolute value 1, so each is e^{2πiφ} — "a phase". Phase estimation reads φ to t binary digits. It is the universal quantum measuring instrument: Shor's is phase estimation of a multiplication map.

- **Coherent version:** H on every `out` qubit; then for each j, `unitary.controlled(out[j])` applied 2^j times (a loop — honest repeated application, no shortcut powers unless the Block provides them); then `iqft(out)`; caller measures. Comment why controlled-U^{2^j} writes the j-th binary digit of φ into the j-th qubit's phase (phase kickback — explain the term: the control qubit, not the target, picks up the eigenvalue's phase, because the target is an eigenstate and is left unchanged).
- **Semiclassical (Griffiths–Niu):** one reusable phase qubit instead of t. Loop from the least-significant digit up: H, apply controlled-U^{2^j}, apply the *classically conditioned* correction rotation `Rz`/`Phase` by −2π·(0.0 b_{j+1} b_{j+2} …) built from digits already measured, H, measure, `reset`, record digit. The classical feedback is ordinary Python — point at the parallel with teleportation's corrections (notebook 03). Comment the deferred-measurement principle: measuring early + feeding back classically is *provably equivalent* to keeping everything coherent and measuring at the end — T15 confirms it empirically.

## 3. Tests

**Acceptance (`test_acceptance_t11_t15.py`), per design doc §9:**

- **T11 (QFT vs FFT):** random seeded 5-qubit state; `qft(reg)` output flat vector vs `np.fft.ifft(psi_flat) * np.sqrt(2**n)` to 1e-12. Comment block required by the design doc: spell out the two conventions being reconciled — numpy's `ifft` carries the e^{+2πijk/N} kernel and a 1/N normalization vs. the QFT's 1/√N, and the qubit-0-is-MSB indexing means the flat vector is already in standard integer order. Also run the `swap=False` variant and undo the reversal in the test by reshaping to `(2,)*n`, `np.transpose` with reversed axes, and flattening — with a comment that transpose-with-reversed-axes *is* bit reversal. This test documents the whole project's bit convention by example.
- **T12:** `qft` then `iqft` = identity to 1e-13 (random 6-qubit state, both swap settings).
- **T13:** for t = 12 and m ∈ {2,…,10}: fidelity between approx-QFT and exact-QFT outputs; assert monotone non-decreasing in m and fidelity(m=8) > 0.999.
- **T14:** U = `Phase(theta=2π·3/8)` as a 1-qubit Block; eigenstate |1⟩; 3-qubit register → outcome `0b011` with probability > 0.999 (read the probability from `inspect`, don't sample).
- **T15:** φ = 0.3 (not exactly representable — the interesting case); coherent version's outcome distribution from `inspect.probabilities()` vs semiclassical outcomes over 500 seeded runs; total-variation distance < 0.05. Comment: TVD = ½ Σ|p−q|, and why 500 shots justifies the 0.05 bound (≈ 1/√500 scale).

**Unit tests (`test_qft.py`), tests-as-documentation:** qft on |0…0⟩ gives the uniform superposition (every amplitude 1/√2^n — the "all frequencies of nothing" case, comment it); qft on a 1-qubit register is exactly H; `approx=1` keeps only the H's; gate_counts of qft(n) has n(n−1)/2 CPhase gates and n H's (structure test); `semiclassical_phase_estimation` uses exactly one phase qubit (n_qubits check); empty register raises cleanly.

## 4. Notebook — `06-qft-phase-estimation.ipynb` ("Reading frequencies off a quantum state")

1. What you will learn. Classical warm-up: a sampled cosine and `np.fft` finding its frequency (3 cells; the owner knows basic numpy — explain fft output layout briefly).
2. Amplitudes as a signal: prepare a small register whose amplitudes trace a cosine (test-style state prep is fine here — say so honestly in the markdown: "we're cheating with inspect-level state prep to make a clean picture; Shor's will earn its periodic state").
3. Run `qft`, `viz.amplitudes` before/after — the comb appears. Play with the frequency.
4. How the circuit does it: walk the 3-qubit QFT gate by gate with `ket()` shown after each gate; binary fractions; why bit-reversal happens (`swap=False` shown, then fixed).
5. Approximate QFT: plot fidelity vs m (T13's data as a picture); the point — most of those tiny rotations were never doing much work.
6. Phase estimation: the eigenvalue-as-angle picture (draw the unit circle); phase kickback narrated; T14's exact case run live, then a non-representable φ showing the peaked-but-spread distribution.
7. Semiclassical version: same answer, 1 qubit instead of t; measure-early = classical feedback; connect back to teleportation.
8. What you now know / next (building a reversible adding machine — the last ingredient Shor's needs).

## Definition of done

- T11–T15 + unit tests pass; all earlier tests pass; **100% coverage maintained** (both `swap` branches, `approx` branch, error paths); pyright/ruff clean; notebook 06 executes.
- T11 contains the convention-documenting comment block; module docstrings meet the pedagogy bar.
- Report "Decisions made".

## Interface decisions to review with the owner (before building)

1. `phase_estimation(unitary, target, out)` argument order and names (design doc says `eigenstate` for the target register — propose `target` with the docstring noting it should hold an eigenstate; confirm which name the owner finds clearer).
2. Semiclassical signature: it needs a circuit and returns an int — show the call in context (it allocates its own phase qubit inside `circuit`); confirm ergonomics.
3. `iqft` as separate function vs `qft.adjoint()` only — recommend keeping both (`iqft` reads better in Shor's), confirm.
