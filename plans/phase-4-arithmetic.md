# Phase 4 — Reversible arithmetic (the honest requirement)

**Read first:** `plans/master-plan.md` (Conventions, and Known risks #3–4), `qsim-design.md` §8.3 (the heart of the project) and T16 in §9. Requires Phase 3 complete. Expect this to be the largest and most finicky phase (the design doc says so too).

**Goal:** modular exponentiation compiled honestly from reversible arithmetic — never a precomputed permutation, never a construction that requires knowing the answer. The design doc's hard constraint goes verbatim in a comment at the top of `arithmetic.py`, with the explanation of why it exists (most "Shor's in 30 lines" demos cheat here and thereby contain none of the algorithm's content).

**Files created:** `src/qsim/algorithms/arithmetic.py`; `tests/test_arithmetic.py` (unit), `tests/test_acceptance_t16.py`; no notebook (Shor's notebook in Phase 5 covers the arithmetic story).

---

## 0. The architecture decision (present at interface review FIRST)

The design doc asks for two things that come from different papers: Cuccaro ripple-carry adders (Toffoli-based, "school addition") *and* Beauregard's 2n+3-qubit Shor construction — but Beauregard is built on Draper's *Fourier-space* adder, not Cuccaro's. Resolution, to present to the owner in plain terms:

> There are two ways to build a reversible adding machine. **Ripple-carry (Cuccaro)** is binary school addition: walk bit by bit, carry in hand — completely transparent, every step inspectable, but it needs extra qubits and Shor-on-15 lands around 20 qubits (runs in maybe a minute, not milliseconds). **Fourier-space (Draper)** first runs the register through the QFT, where adding a number becomes just phase rotations — elegant, directly reuses what notebook 06 taught, and gets Shor-on-15 down to the design doc's 11 qubits — but "addition as rotation" is harder to eyeball. Recommendation: **build both.** Cuccaro is the teaching adder and the T16 workhorse; Draper is what modexp/Shor's run on (matching the design doc's qubit counts). T24's honesty check then counts 3-qubit gates of either kind (Toffoli or doubly-controlled phase).

Everything below assumes that recommendation is accepted; if the owner chooses otherwise, stop and revise this plan.

## 1. Layer 0 — Cuccaro ripple-carry adder (the teaching adder)

`cuccaro_adder(a: Register, b: Register, carry: Qubit) -> None` — computes b += a in place (b keeps the sum, a is unchanged, `carry` is one clean ancilla-style qubit holding the final carry-out). Reference: Cuccaro, Draper, Kutin, Moulton, arXiv:quant-ph/0410184.

Give the subagent the exact gate sequence. Define the two 3-qubit blocks as `@qsim.gate`s with docstrings:

```
MAJ(c, b, a):   CNOT(a, b); CNOT(a, c); Toffoli(c, b, a)
  # "majority": after this, a holds MAJ(a,b,c) = the carry OUT of this bit
  # position, while b and c hold XORs the UMA step will unwind.
UMA(c, b, a):   Toffoli(c, b, a); CNOT(a, c); CNOT(c, b)
  # "un-majority and add": undoes MAJ's scrambling and leaves b = a XOR b XOR c
  # = the sum bit. MAJ then UMA with nothing in between is the identity —
  # the sum appears only because the carry chain visited in between.
```

Structure: MAJ up the bit chain (LSB first), CNOT the top carry into `carry`, UMA back down in reverse order. **Fix and document the register endianness here**: registers pass numbers with `reg[0]` = most significant bit (matching `measure_all` and `encode`); the adder walks from index n−1 (least significant) upward. Get this into one prominent comment — endianness confusion is the expected bug of this phase.

Module docstring must also make the *reversibility* point for the newcomer: classical AND/OR gates destroy information and therefore can't be quantum gates; Toffoli (controlled-controlled-NOT) is how classical logic is done reversibly — it computes AND into a target while keeping its inputs. One paragraph, before any circuit detail.

## 2. Layer 1 — Draper Fourier adder with classical addend

Beauregard's key simplification: in Shor's, one addend is a *classical known number* (N, or a·2^j mod N). Adding classical constant c to a register in Fourier space needs only single-qubit phase gates:

```python
@qsim.gate
def phi_add(reg: Register, c: int) -> None
    # PRECONDITION (state in docstring): reg is in Fourier space, i.e. qft(reg,
    # swap=False) has been applied. In that representation, adding c is a phase
    # rotation on each qubit: qubit j advances by angle 2*pi * c / 2^(n-j).
    # No carries, no ancillas — the QFT turned addition into rotation.
    for j, q in enumerate(reg):
        Phase(q, theta=2 * np.pi * c / 2 ** (len(reg) - j))
```

(Verify the exponent against the project's MSB-first convention and the `swap=False` layout when implementing — derive it in a comment from the QFT definition rather than trusting this sketch.) Subtraction = `phi_add.adjoint()` (angles negate — Phase 2 machinery). Controlled and doubly-controlled versions come free from `with control(...)`.

## 3. Layer 2 — modular adder (Beauregard, arXiv:quant-ph/0205095 §2.2)

`modular_adder(c: int, reg: Register, N: int, anc: Qubit, ctrl: tuple[Qubit, ...]) -> None` — computes reg = (reg + c) mod N, controlled on `ctrl`, using one ancilla and one extra high bit on `reg` (register is n+1 bits for an n-bit N so the intermediate reg+c never overflows). Implement Beauregard's sequence faithfully; write it as a numbered comment block:

```
1. (ctrl-)phi_add(reg, c)                # reg = reg + c        (Fourier space)
2. phi_add(reg, N).adjoint()             # reg = reg + c - N    (maybe negative!)
3. iqft(reg, swap=False)                 # leave Fourier space to read the sign
4. CNOT(reg[0], anc)                     # top bit = 1 iff we went negative;
                                         # copy that fact into the ancilla
5. qft(reg, swap=False)                  # back to Fourier space
6. with control(anc): phi_add(reg, N)    # add N back only if we underflowed
   # reg now holds (reg+c) mod N, but anc still KNOWS whether we underflowed —
   # it is entangled garbage. Steps 7-10 uncompute it (Bennett's trick, §4.3):
7. (ctrl-)phi_add(reg, c).adjoint()      # temporarily subtract c again
8. iqft ... X(reg[0]); CNOT(reg[0], anc); X(reg[0]) ... qft
   # the top bit now flags "did NOT underflow"; the X-conjugated CNOT flips
   # anc back to |0> in exactly the branches where it was 1
9. (ctrl-)phi_add(reg, c)                # restore reg = (reg+c) mod N
```

The uncompute of `anc` is the pedagogical jewel of this phase — the comment must say this is the same dirty-ancilla physics as notebook 04/05, *inside* a production circuit. Wrap the whole thing's ancilla use in `with qc.ancilla(1)` where the call pattern allows, so the Phase 2 verifier actually audits Beauregard's uncomputation at runtime. (If scope plumbing through nested blocks fights the combinator design, allocate `anc` explicitly and `inspect.assert_zero` it after — but report that compromise.)

## 4. Layer 3 — controlled modular multiplier and modexp

Per Beauregard §2.3–2.4, with the design doc's signatures:

```python
def controlled_modular_multiplier(c: Qubit, x: Register, out: Register,
                                  a: int, N: int, anc) -> None
    # out += a * x mod N, controlled on c:
    # for each bit x_j of x (MSB-first register, mind the weights!):
    #   doubly-controlled (c and x[j]) modular_adder of (a * 2^weight) mod N into out
    # The addend per bit is CLASSICALLY precomputed — that is legitimate
    # (it's repeated squaring/doubling, not foreknowledge of the answer).

def controlled_modular_multiply_inplace(c: Qubit, x: Register, a: int, N: int, anc) -> None
    # |x> -> |a*x mod N> controlled on c, via Beauregard's swap trick:
    #   1. controlled_modular_multiplier: |x>|0> -> |x>|ax mod N>
    #   2. controlled SWAP of the two registers
    #   3. adjoint of controlled_modular_multiplier with a^{-1} mod N
    #      (pow(a, -1, N) — python computes modular inverses natively)
    # Step 3 is uncomputation again: it erases the leftover copy of x.
    # Requires gcd(a, N) == 1 (else no inverse exists) — raise ValueError
    # with a teaching message if not.

def modexp(a: int, x: Register, out: Register, N: int, anc) -> None
    # |x>|1> -> |x>|a^x mod N>. out starts at |1> (caller encodes it).
    # For each bit j of the exponent register x:
    #   controlled_modular_multiply_inplace(x[j], out, a^(2^weight) mod N, N, anc)
    # Classical prep: the powers a^(2^k) mod N by repeated squaring — comment
    # that this classical loop is the "fast exponentiation" idea, and the
    # quantum circuit runs all exponents in superposition through it.
```

Top-of-file comment (design doc §8.3 hard constraint, verbatim quote plus the "why"): modexp must be compiled from reversible arithmetic; no permutation matrices, no lookup tables, nothing that requires knowing the answer.

Also expose small internal helpers as testable functions (`_qft_nosap` wrappers, addend tables) rather than burying them in closures — 100% coverage plus tests-as-documentation want them visible.

## 5. Tests

**Acceptance (`test_acceptance_t16.py`) — T16, exhaustive and exact:**

Arithmetic circuits map basis states to basis states, so no sampling: `encode` the inputs, run, check via `inspect.amplitude` that all weight sits on the expected output (amplitude within 1e-9 of 1 — thousands of float rotations; report the observed worst case).

- `cuccaro_adder`: all (a, b) pairs for 3-bit and 4-bit registers — sums and final carry both checked (2×(64+256) cases; runs fast at ≤9 qubits).
- `modular_adder`: for N ∈ {5, 7, 13} (register n+1 bits): all (c, b) with b < N; check (b+c) mod N; also check the ancilla came back clean (that's the point).
- `controlled_modular_multiply_inplace`: N=15, a ∈ {2, 7, 8, 13}: all x < N with control on and off (off ⇒ identity).
- `modexp`: N=15, a=7: all x ∈ [0, 15]; expect 7^x mod 15. One superposition test too: H on all exponent qubits, then modexp, then assert via `inspect` that the joint state is Σ|x⟩|7^x mod 15⟩/4 — the "compute all values at once" state, with a comment that *this* is what a permutation-matrix shortcut would fake.

**Unit (`test_arithmetic.py`), tests-as-documentation:** MAJ/UMA round-trip is identity; `phi_add` then its adjoint is identity; `phi_add(reg, 0)` is identity; adding 1 repeatedly walks the register through all values (a for-loop odometer — nice doc test); `pow(a,-1,N)` precondition raise for gcd≠1 with message checked; endianness pin: `encode(6)`, add 1, `measure_all` == 7 on a 4-bit register; every public function rejects overlapping registers (x and out sharing a qubit → `NoCloningError`).

**Structure preview (not yet T24, but cheap):** assert `gate_counts()` after a modexp contains no gate acting on >3 qubits.

## 6. Performance guardrail

T16's exhaustive loops re-simulate thousands of small circuits; keep each case on a fresh minimal circuit (≤ 11 qubits) and the whole file under ~60s. If it creeps past that, reduce to N ∈ {5, 13} for `modular_adder` and note it — do not sacrifice exhaustiveness over inputs for a given N.

## Definition of done

- T16 + unit tests pass; all earlier tests pass; **100% coverage maintained**; pyright/ruff clean. (No notebook this phase.)
- `arithmetic.py` opens with the honesty-constraint comment; the modular-adder uncompute sequence carries its numbered teaching comments; endianness documented in one place and pinned by test.
- Report "Decisions made" — expected entries: the exact `phi_add` angle convention derived, how ancilla verification was plumbed, observed T16 amplitude tolerance.

## Interface decisions to review with the owner (before building)

1. **The two-adder architecture decision** (§0 above) — the one that matters. Present the plain-language tradeoff; recommend "build both."
2. Signature ergonomics: `modexp(a, x, out, N, anc)` argument order (design doc's) vs `modexp(x, out, *, a=..., N=...)` — show both as call sites.
3. Whether `anc` parameters should be explicit registers (design doc style) or internally allocated via `with qc.ancilla(...)` — recommend internal allocation (the verifier then runs automatically); confirm.
4. T16 runtime budget (~60s) acceptable, or should the exhaustive matrices shrink?
