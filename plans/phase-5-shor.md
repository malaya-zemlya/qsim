# Phase 5 — Shor's algorithm end-to-end

**Read first:** `plans/master-plan.md` (Conventions), `qsim-design.md` §8.4, and T17–T19, T22–T25 in §9 (T18 is the most important test in the suite; T17 lands here per the master-plan deviation note). Requires Phase 4 complete.

**Goal:** factor 15 and 21 with a fully honest circuit, expose the complete classical+quantum trace in a `ShorResult`, and demonstrate empirically that skipping uncomputation destroys the algorithm (T18).

**Files created:** `src/qsim/algorithms/shor.py`; `tests/test_shor_classical.py`, `tests/test_acceptance_t17_t19.py`, `tests/test_acceptance_t22_t25.py`; `notebooks/08-shor.ipynb`. (Notebook numbering follows the master-plan index.)

---

## 1. Classical scaffolding (`shor.py`, pure-Python helpers, all public and unit-tested)

```python
def continued_fraction_convergents(num: int, den: int) -> list[tuple[int, int]]
    # All convergents p/q of num/den. Docstring teaches the algorithm with the
    # worked example the notebook reuses: 1365/2048 -> ... -> 2/3 -> ...
def candidate_period(measured: int, t: int, N: int) -> int | None
    # Best convergent denominator q < N for measured/2^t; None if none works.
def _classical_checks(N: int) -> ...   # reject even N, prime powers, primes —
    # with messages saying WHY Shor's is not needed for those cases.
```

## 2. Period finding and the driver

```python
@dataclass
class ShorResult:
    N: int; a: int
    t: int                       # phase-register width
    n_qubits: int
    measured: int                # raw phase-register readout
    convergents: list[tuple[int, int]]
    period: int | None
    period_verified: bool        # pow(a, r, N) == 1, checked classically
    factors: tuple[int, int] | None
    failure: str | None          # human-readable reason when factors is None
    gate_counts: dict[str, int]

def find_period(a: int, N: int, *, seed=None, semiclassical=False,
                _skip_uncompute=False) -> ShorResult    # one quantum run
def shor(N: int, *, a: int | None = None, seed=None, semiclassical=False,
         max_attempts: int = 10) -> ShorResult          # full driver with retries
```

`failure` values are teaching text, not codes: "measured 0 — the phase register gave no information this run; rerun", "period 3 is odd — a^(r/2) needs r even; try another a", "a^(r/2) ≡ −1 (mod N) — the gcd trick degenerates; try another a". Design doc: return the failure reason rather than looping silently — `shor` retries over `a` (seeded rng choosing a coprime to N) up to `max_attempts`, collecting attempts; the returned `ShorResult` is the successful one (or the last, with `failure` set).

**Coherent period-finding circuit** (per design doc sizes: N=15 → 11 qubits with t=2n=8? — no: 2n+3 = 11 total for n=4 means t=2n=8 phase qubits + n+1=5 work + ... follow Beauregard's accounting exactly and assert the totals in T22; derive the register widths in comments, don't hand-wave):
1. phase register `x` (t = 2n bits), work register `out` (n+1 bits) encoded to |1⟩, plus Beauregard's ancilla.
2. H on every phase qubit; `modexp(a, x, out, N, anc)`; `iqft(x)`; `measure_all(x)`.
3. Classical post-processing via the §1 helpers; factors = `gcd(a^(r/2) ± 1, N)`.

**Semiclassical variant** (`semiclassical=True`): the Griffiths–Niu loop from Phase 3 driving the controlled modular multiplications with one reusable phase qubit — total ≈ n+4 qubits; assert the count in T25. Reuse `semiclassical_phase_estimation` if its Block interface fits controlled-modexp-steps; otherwise inline the loop here and note it.

**The T18 escape hatch:** `_skip_uncompute=True` routes the multiplier's uncompute step (Beauregard's swap-trick step 3, Phase 4 plan §4) into a no-op and passes `_unsafe_skip_check=True` to the ancilla machinery — garbage stays. Keyword is underscore-private and its docstring says it exists only for T18/notebook 08.

## 3. Tests

**`test_shor_classical.py`** — the classical helpers, exhaustively and as documentation: convergents of 1365/2048 land near 2/3... (use the true worked example: for N=15, a=7, t=8, a typical measurement is 64, 128, or 192; 192/256 → 3/4 → period 4); `candidate_period` on all t=8 measurements for r=4 recovering 4 where theory says it should; `_classical_checks` messages; `ShorResult.failure` strings for the odd-period and a^(r/2)≡−1 cases (drive with hand-picked a: for N=15, a=14 gives r=2 with 14^1 ≡ −1 — the degenerate case; a good doc-test).

**`test_acceptance_t17_t19.py`:**
- **T17 (precision):** run find_period(7, 15, seed=fixed) under complex64 and complex128 (`qsim.set_dtype`); compare the phase-register probability distributions just before measurement: peak bins agree to ~1e-7 relative; valley bins (near-zero) differ wildly in relative terms — assert both, per design doc, with the design-doc comment about badly-conditioned outputs being exactly the ones that don't matter. Restore dtype in a fixture finalizer.
- **T18 (the demonstration test — most important in the suite):** N=15, a=7, seeded. Run once normally, once with `_skip_uncompute=True`. Metric: peak-to-background ratio of the phase-register distribution (max of the 4 expected peak bins over the mean of all other bins). Assert clean-run ratio ≥ 10× the dirty-run ratio. Docstring (required by design doc): this is TD2 in different clothing — the dirty ancillas are an environment that recorded which-path information; the lost peak is lost visibility; see §4.4 and notebook 06.
- **T19 (entanglement across modexp):** N=15, a=7; entropy between exponent register and the rest: ≈0 before modexp, ≈log2(4)=2 bits after (accept 1.9–2.0 — it saturates near log2 r), and state the design-doc point in the docstring: the algorithm *works by* entangling exponent with work register, then reading the periodicity that entanglement imprints. Sample the during-modexp points with an `on_op` hook (Phase 2.75) rather than hand-splicing the circuit — the hook is exactly the instrument for "watch a quantity evolve as a program runs".

**`test_acceptance_t22_t25.py`:**
- **T22:** `shor(15, a=7, seed=...)` → factors {3,5}, period 4, `n_qubits == 11`, deterministic under the seed.
- **T23:** `shor(21, seed=...)` → {3,7} within 10 attempts (seeded so it's reproducible; find and hard-code a good seed, comment the attempt count observed).
- **T24 (honesty check):** on the T22 run's `gate_counts`: ≥100 three-qubit gates (Toffoli + doubly-controlled-phase — count both, per the Phase 4 architecture decision), and no op in history touches >3 qubits. Comment: a permutation-matrix cheat would show ~0 such gates; the loose floor only needs to separate honest from fake (master-plan risk note / design doc).
- **T25:** `shor(15, semiclassical=True, seed=...)` succeeds; assert its qubit count (expected ≈ n+4 = 8; pin the exact number once implemented and comment the accounting).

Runtime guardrail: T22/T18 runs are ~11 qubits × tens of thousands of gates — target < 30s each; T23 (13 qubits) < 2 min. If exceeded, profile before weakening anything; likely culprits are needless `psi.copy()` in kernels.

## 4. Notebook — `08-shor.ipynb` ("Factoring numbers with interference")

The capstone. Structure:
1. What you will learn; why factoring matters (RSA in two sentences); the honest-circuit promise (quote the §8.3 constraint — and that *this* notebook's circuit keeps it).
2. Period finding = factoring: the classical reduction with small-number arithmetic the reader can verify by hand (7^x mod 15 table; spot the period; gcd punchline). All classical, no quantum yet.
3. The quantum part in one picture: superpose all exponents → compute 7^x mod 15 *once, in superposition* → the state is periodic → QFT reads the period. Show T19's entropy jump live ("the registers now share their fate").
4. Watch the comb: `viz.probabilities` of the phase register before measurement — flat, then 4 peaks at multiples of 256/4 = 64. **The design doc calls this the moment Shor's stops being symbol manipulation; give it space.**
5. From peak to period: measure, continued fractions step by step (reuse the helper's trace), verify r, compute the factors.
6. When it fails: run the failure modes (measured 0; odd r via another a; a=14's degenerate case) and read the `failure` strings. Probabilistic ≠ broken — retry logic shown.
7. **The uncomputation demonstration (T18 live):** clean vs `_skip_uncompute=True` distributions side by side; markdown callback to notebooks 04 and 06 — the garbage register is an environment; this is decoherence self-inflicted. The single most important cell pair in the whole notebook series.
8. `shor(21)`; qubit/gate-count table (`gate_counts` for both N); semiclassical variant and its qubit savings.
9. What you now know / next (Grover — a completely different way to use interference).

## Definition of done

- All acceptance + unit tests pass; every earlier test passes; **100% coverage maintained** (failure branches, `semiclassical`, `_skip_uncompute`, dtype switch all exercised); pyright/ruff clean; notebook 08 executes in < ~5 min.
- `ShorResult` exposes the full trace; failure strings teach.
- Report "Decisions made" (semiclassical qubit count, T23 seed/attempts, observed tolerances/timings).

## Interface decisions to review with the owner (before building)

1. `ShorResult` field list above — walk through it with a filled-in example for N=15; anything missing the owner would want to see in the notebook trace?
2. `shor()` retry semantics (return-with-failure vs raise after max_attempts) — recommend return-with-failure; confirm.
3. The T18 escape-hatch naming (`_skip_uncompute`) and its appearance in a teaching notebook — comfortable showing a private flag, or should it be a public, loudly-documented `uncompute=False` parameter *because* the demonstration is a first-class feature? (Recommend the latter, actually — the design doc calls the demo first-class; present both.)
4. Runtime budgets above — acceptable?
