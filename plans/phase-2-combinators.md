# Phase 2 — Combinators: record mode, `control`, `adjoint`, ancilla scopes, `@qsim.gate`

**Read first:** `plans/master-plan.md` (Conventions), `qsim-design.md` §4.1–§4.3, §2.4 (axis lifecycle — this phase stress-tests it), T8–T10 specs in §9. Requires Phase 1.5 complete.

**Goal:** blocks of gates become first-class: they can be controlled, inverted, and safely use scratch qubits whose cleanliness is *verified*. The `DirtyAncillaError` check is the deliverable that matters — it is how the library teaches uncomputation.

**Files created:** `src/qsim/combinators.py`, additions to `circuit.py` and `gates.py`; `tests/test_combinators.py`, `tests/test_acceptance_t8_t10.py`; `notebooks/04-combinators.ipynb`.

---

## 1. Record mode (the mechanism everything else here uses)

Design doc §4.1–4.2. Execution stays eager *except* inside combinator scopes, where gates are recorded instead of executed, transformed on scope exit, then executed.

On `Circuit`, add a stack: `_record_stack: list[list[Op]]`. Gate application path becomes:

```python
def _emit(self, op: Op) -> None:
    if self._record_stack:
        self._record_stack[-1].append(op)   # inside a combinator: record, don't run
    else:
        self._execute(op)                    # normal eager path: run + append to history
```

`Op` gains what execution needs to be replayable (it mostly has it from Phase 1): gate name, qubit *ids* (never axes — resolution happens at execution time via the id→axis table), params, controls tuple. `_execute(op)` is the single funnel: resolves axes, dispatches to the right `state.py` kernel, appends to `history`. Refactor Phase 1's gate path through `_emit`/`_execute` if it isn't already shaped that way — this refactor is expected and safe because the Phase 1 test suite pins behavior.

Scopes must nest (design doc: "Scopes nest"): `with control(c): with adjoint(): block()` works, and transformations compose in the right order (see §4 below for the required semantics test).

## 2. `control` and `adjoint` context managers

```python
with qsim.control(c):        # single control
with qsim.control(c1, c2):   # multiple controls
with qsim.adjoint():
```

Both are implemented in `combinators.py`. They find the active circuit from the first gate recorded — **no**: they need the circuit at `__enter__` to push the record buffer. Resolve it from the control qubits' circuit (`control`), and for `adjoint()` (which takes no qubits) from a module-level "current circuit" — avoid that global. Instead: `adjoint()` pushes a *circuit-agnostic* pending scope onto a module-level stack, and the first recorded gate binds it to that gate's circuit; raise if a second circuit's gate appears in the same scope. Simpler alternative (choose this if the above feels fragile in implementation): both context managers live on the circuit too — `qc.control(c)` / `qc.adjoint()` — with the module-level `qsim.control(c)` resolving the circuit from `c` and `qsim.adjoint()` requiring at least one gate to bind. Whichever is built, `qsim.control(...)`/`qsim.adjoint()` from the design doc must work; present the binding question at the interface review.

Transformations on scope exit:

- **control(controls):** for each recorded op, append the control ids to `op.controls`. Execution then routes through `apply_controlled` with all controls — this is why Phase 1 built `apply_controlled` for arbitrarily many controls. There is no decomposition into elementary gates: at simulator level, "conditioned on all controls being |1⟩" is just more sliced axes. The docstring must say this explicitly *and* note the contrast: on real hardware a multiply-controlled gate must be decomposed into 1- and 2-qubit gates; the simulator's slice is the mathematical meaning those decompositions implement. Validation at exit: a control qubit that also appears inside the body's ops → `NoCloningError` (a qubit can't control an operation on itself).
- **adjoint():** reverse the buffer; invert each op. Each gate declared its inverse in Phase 1 (`H⁻¹=H`, `S⁻¹=S†` new named entry, `Rz(θ)⁻¹=Rz(−θ)`, `T⁻¹=T†`, …). Add the dagger gates (`Sdg`, `Tdg`, `SXdg`) as real gates in `gates.py` (public — users will see them in history and diagrams). Measurement ops inside an adjoint scope → raise `QsimError` at record time with a teaching message: measurement is irreversible, the only non-unitary operation in the library; there is nothing to replay backwards.

Reusable-block forms `block.controlled(c)(...)` / `block.adjoint()(...)` come from the decorator (§3 below).

## 3. `@qsim.gate` — reusable blocks

```python
@qsim.gate
def bell(a: Qubit, b: Qubit) -> None:
    H(a); CNOT(a, b)

bell(a, b)                    # runs eagerly, but history records it as one "bell" unit
bell.adjoint()(a, b)
bell.controlled(c)(a, b)
```

Implementation: the decorator wraps the function in a `Block`. Calling a `Block` runs the body inside a fresh record scope, then executes the buffer — capturing classical args at record time (design doc §4.2: this is what lets the same block be inverted/controlled later). `Block.adjoint()` and `Block.controlled(*controls)` return new callables that apply the corresponding buffer transformation before execution. History: record a structured entry (block name + nested ops) so `gate_counts()` can count both the block as a unit and its constituent gates — `gate_counts()` counts elementary gates; add `block_counts()` if trivial, else defer and note it. Enforce the design-doc rule "decorated blocks must be composed only of other gates and blocks — no direct state manipulation": blocks have no access to `psi` anyway, so the enforcement is that the body receives only handles; document it.

## 4. Ancilla scopes — the deliverable that matters

Design doc §4.3, on `Circuit`:

```python
with qc.ancilla(3) as anc:        # anc: Register of 3 fresh qubits in |000>
    ...
# __exit__: verify anc is |000> and unentangled; then deallocate; handles die
```

The exit check, spelled out (state the reasoning as a comment — it is a small theorem):

```python
# If the probability that the ancillas read all-zero is exactly 1, then every
# amplitude with any ancilla bit = 1 vanishes, so the state factorizes as
# |0...0>_anc ⊗ |rest> — the ancillas are BOTH zero AND unentangled. One number
# checks both conditions. Compute it as 1 - (norm of the all-zeros ancilla slice)^2.
```

Use `inspect.assert_zero(subset, tol=1e-10)` from Phase 1. On failure raise `DirtyAncillaError` with the full teaching message (design doc §4.3): leftover garbage stays entangled with the answer register and destroys the interference the algorithm depends on; real hardware could not even check this; cross-reference §4.4/notebook 05 ("a dirty ancilla is an environment that has recorded which-path information"). Include the offending probability in the message.

Deallocation (the §2.4 stress test): take the all-zeros slice of the ancilla axes (this drops those axes), rebuild the id→axis table for all surviving qubits, mark the ancilla handles `_live = False`. After this, T7's promise becomes real: using a dead handle raises `DeadQubitError` whose message explains ancilla scoping. Also implement the design-doc docstring note: dropping a handle without a scope does *not* free the qubit (no quantum garbage collection — physics, not a missing feature).

Test-only escape hatch for Phase 5's T18: `qc.ancilla(n, _unsafe_skip_check=False)` — keyword-only, underscore-prefixed, documented as "exists solely so test T18 can demonstrate what goes wrong; never use it". It skips the check but still deallocates **by tracing** — no. Careful: if the ancillas are dirty, slicing to all-zeros would change the state's norm (that *is* the corruption T18 wants to exhibit — a non-unitary drop of amplitude). Decide and document: with `_unsafe_skip_check=True`, the qubits are **not** deallocated (they remain as silent garbage axes, handles die) — this models "walking away from your garbage," which is exactly the physics T18 needs. Flag this at the interface review.

## 5. Acceptance tests — `tests/test_acceptance_t8_t10.py`

- **T8:** entangle ancillas and exit → `DirtyAncillaError` (assert the message mentions interference); identical block with uncomputation exits cleanly. Use a CNOT into the ancilla as the "dirt", and a second CNOT as the uncompute.
- **T9:** random 6-qubit block (seeded gate sequence); apply `U` then `with adjoint(): U`; fidelity with the initial random state within 1e-13 of 1.
- **T10:** `with control(c): X(t)` equals `CNOT(c, t)` on a random 2-qubit + control state (state vectors allclose). Multi-gate block: compare against an independently built controlled matrix in the test file with `np.kron` (n ≤ 4; comment the kron construction — this is the only place full matrices are allowed).
- **Composition semantics (extra, required):** `control` ∘ `adjoint` == `adjoint` ∘ `control` (controlled-inverse equals inverse-of-controlled — assert on states); nested `control(c1)` inside `control(c2)` equals `control(c1, c2)`.

Unit tests (`test_combinators.py`) for full coverage, written as documentation: measurement inside `adjoint` raises with the "measurement is irreversible" message; control qubit reused inside body raises; nested ancilla scopes; ancilla scope exited via an exception still cleans up handles; `_unsafe_skip_check` leaves garbage axes (assert `n_qubits` unchanged); `bell.adjoint()` on a Bell state returns |00⟩; a `Block` with a classical parameter round-trips through `adjoint` (angle negated).

## 6. Notebook — `04-combinators.ipynb` ("Programs made of gates")

1. What you will learn. Why blocks: algorithms are built from named reusable pieces (preview: the QFT and Shor's arithmetic are blocks all the way down).
2. `@qsim.gate`, define `bell`, call it, show `gate_counts()`/history treating it as a unit.
3. `adjoint`: running a program backwards. Every gate has an inverse because every gate is a rotation (unitary = no information lost — contrast with classical AND, which destroys a bit). Demo: scramble with a random block, unscramble with its adjoint. Then: *what can't be undone* — measurement; show the raise and read its message aloud in markdown.
4. `control`: doing something "only if" a qubit says so — while the qubit is in superposition. Show `with control(c): bell(a,b)` with c in |+⟩; inspect the resulting state (half the amplitude made a Bell pair, half did nothing — a superposition *of circuits having run or not*).
5. Ancillas: scratch space. The trap: compute something using scratch, "throw the scratch away", and your answer breaks. Build the failure live: put a superposed query through a block that copies which-path info into an ancilla, show interference visibility die (mini two-slit with an H-sandwich, previewing notebook 05); then uncompute and show it restored. Read the `DirtyAncillaError` message in a `try/except` cell.
6. Bennett's trick in words: compute → copy out the answer → uncompute the scratch. Why reversible computing forces this discipline.
7. What you now know / next (decoherence: what "the environment measured you" does to a qubit — same phenomenon as the dirty ancilla, at cosmic scale).

## Definition of done

- T8–T10 + composition tests + unit tests pass; all earlier tests pass; **100% coverage maintained** (the escape hatch, every raise path, and both context-manager exit routes — normal and exception — included).
- pyright/ruff clean; notebook 04 executes.
- The Phase 1 refactor to `_emit`/`_execute` introduced no behavior change (Phase 1 suite green without edits — any needed edit must be reported).
- Report "Decisions made".

## Interface decisions to review with the owner (before building)

1. `qsim.control(c)` / `qsim.adjoint()` module-level vs `qc.control(c)` / `qc.adjoint()` methods — show both in a usage example; how should `adjoint()` find its circuit?
2. The `_unsafe_skip_check=True` semantics for T18 (garbage axes stay, handles die) — walk the owner through why in one paragraph.
3. `block.controlled(c)(x, y)` call-shape — comfortable, or prefer `block.controlled(c, x, y)`?
4. Whether `gate_counts()` should count inside blocks (recommended: yes, elementary gates) — confirm with an example output.
