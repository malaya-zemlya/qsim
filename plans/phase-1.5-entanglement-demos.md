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

## Interface decisions to review with the owner (before building)

1. The game-style API above (self-contained functions building their own circuits) vs. gate-style building blocks on a user circuit — show both call styles, confirm the former.
2. `TeleportResult` fields — is this the trace-style result object the owner wants (it previews `ShorResult` in Phase 5)?
3. Notebook 03's skeptic framing — one-paragraph sample for tone check.
