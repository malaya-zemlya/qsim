# Phase 1.5 — Entanglement demos: CHSH, teleportation, superdense coding

**Read first:** `plans/master-plan.md` (Conventions), `qsim-design.md` §8.6 and the TB1–TB3 test specs in §9. Requires Phase 1 complete.

**Goal:** three short algorithm modules that prove, with runnable code, that entanglement is not classical correlation (CHSH), that quantum states can be moved but not copied (teleportation), and that one entangled qubit can carry two classical bits (superdense coding). This phase needs *nothing* beyond Phase 1 machinery: mid-circuit measurement, Python `if` on measurement results, and `inspect.expectation`.

**Files created:** `src/qsim/algorithms/__init__.py`, `src/qsim/algorithms/chsh.py`, `src/qsim/algorithms/teleportation.py`; `tests/test_acceptance_tb1_tb3.py`; `notebooks/03-bell-tests-teleportation.ipynb`.

---

## 1. `algorithms/chsh.py`

The CHSH game, presented in the code as a *game* (that framing is the accessible one):

> Alice and Bob each receive a random question bit (x for Alice, y for Bob) and must each answer ±1 without communicating. They win if the product of their answers equals +1, except when both questions are 1, where they need −1. Classical strategies win at most 75% of rounds. Sharing a Bell pair wins ≈85.4%.

API:

```python
def chsh_expectation(angle_a: float, angle_b: float) -> float
    # Build a fresh Bell pair; rotate Alice's measurement basis by angle_a and
    # Bob's by angle_b (see below); return <A B> via inspect.expectation("ZZ").

def chsh_S(settings: tuple[float, float, float, float] = OPTIMAL_SETTINGS) -> float
    # S = E(a,b) + E(a,b') + E(a',b) - E(a',b') for angles (a, a', b, b')

def chsh_sampled(shots: int, *, seed: int | None = None,
                 settings: tuple[float, float, float, float] = OPTIMAL_SETTINGS) -> float
    # Same S but each E estimated from `shots` actual measurements per setting.

def classical_bound() -> float
    # Brute-force every deterministic local strategy; return the max S (== 2.0).

OPTIMAL_SETTINGS: tuple[float, float, float, float]  # (0, pi/2, pi/4, -pi/4)
```

Implementation notes:

- "Measure in a rotated basis" must be explained in the docstring, because it's the one new physics idea in this phase: measuring observable cos θ·Z + sin θ·X is the same as applying `Ry(q, theta=-θ)` and then measuring Z. In code: rotate, then `expectation("ZZ")` (analytic) or rotate-then-`measure` (sampled), mapping outcome 0 → +1 and 1 → −1.
- With the optimal settings, each of the three "+" correlators is cos(π/4)=1/√2 and the "−" one is −1/√2, so S = 4/√2 = 2√2 ≈ 2.828. State this arithmetic in a comment so the expected value isn't magic.
- `classical_bound`: a deterministic local strategy is 4 fixed answers (A, A′, B, B′ ∈ {±1}); loop over all 16 with `itertools.product`, compute S = A·B + A·B′ + A′·B − A′·B′, return the max. Comment the one-line algebra showing why it's 2: S = A(B+B′) + A′(B−B′), and one of (B+B′), (B−B′) is always 0 while the other is ±2.
- Each call builds its own private `Circuit` (these are self-contained demos, not gates on a user's circuit) — note this in the module docstring since it differs from the gate-style API.

## 2. `algorithms/teleportation.py`

```python
def teleport(state_prep: Callable[[Qubit], None], *, seed: int | None = None) -> TeleportResult

@dataclass
class TeleportResult:
    m1: int; m2: int              # Alice's two measurement outcomes
    fidelity: float               # |<target state | prepared state>|^2, from inspect
    corrections: str              # e.g. "X then Z" — which fixups Bob applied
```

Protocol, spelled out step-by-step in the module docstring for a reader who has just met Bell pairs (notebook 03 mirrors this):

1. Three qubits: `msg` (prepared by `state_prep` — an unknown state), and a Bell pair `a` (Alice's) + `b` (Bob's).
2. Alice: `CNOT(msg, a)` then `H(msg)`, then measure both of her qubits → two classical bits.
3. Bob applies `X(b)` if the second bit is 1, `Z(b)` if the first is 1 (plain Python `if` — classical feedback is just control flow in an eager simulator; say so in a comment, it demystifies "classical communication").
4. `b` is now exactly the message state; `msg` and `a` are left in computational basis states — the original is *destroyed*, which is the no-cloning theorem's fingerprint on the protocol. Compute fidelity by preparing the same state on a fresh 1-qubit circuit and comparing Bloch vectors / state vectors of the reduced state of `b`.

Also in this file:

```python
def superdense_send(bits: tuple[int, int], *, seed: int | None = None) -> tuple[int, int]
    # Encode two classical bits into one half of a Bell pair (apply Z^b1 X^b2 to
    # Alice's qubit), "send" it, decode with CNOT + H + measure. Returns decoded bits.
```

Docstring: superdense coding is teleportation run backwards — teleportation spends one Bell pair + 2 classical bits to move 1 qubit; superdense spends one Bell pair + 1 qubit to move 2 classical bits.

## 3. Tests — `tests/test_acceptance_tb1_tb3.py`

- **TB1:** `chsh_S()` == 2√2 to 1e-12. `chsh_sampled(shots=100_000, seed=...)` > 2.7. `classical_bound()` == 2.0 exactly. Docstring: "no local hidden-variable model reaches S > 2; quantum mechanics does. This is the experimentally confirmed (Nobel 2022) sense in which entanglement is not classical correlation."
- **TB2:** teleport random states (build `state_prep` from seeded random `Ry`/`Rz` angles); assert fidelity 1 to 1e-12; run enough seeds to hit all four (m1,m2) branches and assert all four seen; assert the source qubit's reduced state is a computational basis state (entropy < 1e-12 and z-component of Bloch vector ±1).
- **TB3:** all four messages round-trip exactly, any seed.

## 4. Notebook — `03-bell-tests-teleportation.ipynb` ("Entanglement is not secret agreement")

1. What you will learn. Recap Bell pair from notebook 02. Pose the skeptic's question honestly: *"maybe the two qubits just secretly agreed on their answers in advance — like a pair of gloves separated into two boxes."* This local-hidden-variable idea is the thing CHSH kills; take it seriously first.
2. The CHSH game, rules in plain language, with a table of the win condition.
3. Best classical strategy: run `classical_bound()`, show the 16-row strategy table as a small dataframe-style printout, max S = 2 (win rate 75%).
4. The quantum strategy: measurement angles drawn on a circle diagram (matplotlib, simple); run `chsh_S()` → 2.828…; `chsh_sampled` → ≈2.82. Sentence on the 2022 Nobel Prize for the real-world versions of this experiment.
5. Sweep: plot S vs Bob's angle offset — the cos curve peaking at 2√2 above a horizontal classical-bound line at 2. (One plot; label the classical bound.)
6. Teleportation: the problem (you can't copy, you can't measure-and-resend without destroying the state — connect to no-cloning from notebook 02). Walk the protocol with `ket()` printed at each step for one concrete run. Show all four correction branches by reseeding.
7. Superdense coding briefly, as the mirror image.
8. What you now know / next (making blocks of gates controllable and reversible — the tools all later algorithms need).

## Definition of done

- TB1–TB3 pass; all earlier tests still pass with **100% coverage of `src/qsim/`** maintained (cover: non-optimal `settings` arguments, every correction branch in `teleport`, all four `superdense_send` messages — the TB specs already force most of this); pyright and ruff clean; notebook 03 executes.
- Test names read as documentation (e.g. `test_no_classical_strategy_beats_S_of_2`, `test_teleported_state_arrives_intact_while_original_is_destroyed`).
- CHSH docstrings contain the game framing and the classical-bound algebra.
- Report "Decisions made".

## Interface decisions — resolved

Presented as usage examples before any code was written. **Phase 1.5 shipped; this is the
record, not a to-do list.**

1. **Self-contained functions**, as proposed — each demo builds its own `Circuit` and runs
   the whole experiment. *Why:* these are experiments to run and read, not parts to
   compose, and nothing later in the build depends on them, so a block-style API would be
   flexibility nobody spends. The cost is that the protocol becomes something you call
   rather than something you write, so notebook 03 walks teleportation gate by gate on its
   own circuit first and only then shows the packaged `teleport()`.
2. **`TeleportResult` keeps the plan's four fields plus `source_bits`** — the no-cloning
   fingerprint, showing Alice's qubits left holding two random classical bits and no trace
   of the message. This sets the shape for `ShorResult` in Phase 5: a frozen dataclass of
   named results, not a tuple.
3. **Notebook 03's skeptic framing approved as sampled** — the pair-of-gloves image, giving
   the local hidden-variable position a fair hearing before Bell's test refutes it. Written
   verbatim into the notebook's opening.

## Deviations from this plan (as built)

- **`TeleportResult` has a sixth field, `source_bloch_z`.** TB2 requires numeric proof that
  Alice's qubits end in a computational basis state, and the result object is the test's
  only channel to the circuit. `source_bits` is derived from it (sign of z), so the claim
  "the original is destroyed" is checked against the state rather than inferred from the
  measurement outcomes.
- **`classical_strategies()` added** alongside `classical_bound()`, returning all 16 rows as
  `((A, A′, B, B′), S)`. Notebook 03 prints the exhaustive search rather than asserting its
  result, which §8.6 of the design doc explicitly asks for.
- **`chsh_sampled` samples via `inspect.sample()`** instead of rebuilding a circuit per shot.
  The draws are statistically identical to repeated runs — `sample()` is non-collapsing and
  draws independently from the same distribution — and it avoids building 400,000 circuits
  for the 100k-shot acceptance test. The docstring says plainly that a real lab would have
  to rebuild the pair for every shot.
- **`chsh_sampled` derives its four per-setting seeds from one generator**, not `seed`,
  `seed+1`, `seed+2`, `seed+3`. Consecutive seeds give correlated PCG64 streams; see
  `tests/CLAUDE.md`. Caught in review of the first implementation, which had the bug.
- **`CLASSICAL_LIMIT` and `QUANTUM_LIMIT` constants exported** so notebooks and tests stop
  spelling `2.0` and `2*np.sqrt(2)` inline.
- **`superdense_send` validates its input**, raising `ValueError` on anything that is not two
  bits.
- **Notebook 03 gained a section the plan did not have** — §5, "But what actually
  travelled?" — added at the owner's request. It demonstrates no-signalling numerically
  (Alice's reduced state is `I/2` whatever Bob does, and conditioning on his outcome *does*
  change it while averaging over outcomes restores it), states Bell's **three** assumptions
  rather than one, and treats the observer-centred / relative-state reading seriously: it
  gives up "definite single outcomes", not locality, and survives Bell only if the
  observer's post-measurement state is genuinely superposed rather than merely unknown. It
  points forward to notebook 05, where decoherence builds exactly that mechanism.
- **Notebook 02's forward pointer was wrong** — it named `03-chsh-and-friends.ipynb`, which
  never existed. Corrected to the real filename.
- Sign convention worth recording: measuring the observable cos θ·Z + sin θ·X is
  `Ry(q, theta=-θ)` followed by a z-measurement. The minus sign is because rotating the
  state by −θ is the same experiment as rotating the apparatus by +θ.
