# Phase 2.75 — The tape and the transform layer: `within`, closed block algebra, checkpoint/rewind, hooks

**Read first:** `plans/master-plan.md` (Conventions), `qsim-design.md` §4.5–§4.6 (the spec for this phase, new), §2.1 (tape methods on `Circuit`), TT1–TT8 in §9, and `src/qsim/CLAUDE.md`. Requires Phase 2.5 complete (it is; this phase retrofits shipped code).

**Goal:** complete the PyTorch-shaped design the library already half-has. The history becomes a user-visible tape (checkpoints, rewind, hooks); conjugation becomes a combinator (`within`); and the block algebra closes (`Block.adjoint()` returns a `Block`). Plus fixes for three deviations found by auditing the shipped code against this design.

**Files touched:** `src/qsim/combinators.py` (within, Block closure), `circuit.py` (checkpoint/rewind/on_op, funnel fix), `decoherence.py` (pointer_coupling refactor), `viz.py` (no change yet — entropy_trace lands in Phase 6 as a hook client); `tests/test_tape.py`, `tests/test_acceptance_tt1_tt8.py`; notebook 04 extended (new sections), notebook 06 gets one cross-reference cell.

---

## 0. Deviations found in shipped code (fix all three here)

- **D1 — `Block.adjoint()` / `Block.controlled()` return bare closures** ([combinators.py:188](src/qsim/combinators.py#L188), [:198](src/qsim/combinators.py#L198)), while gates return real gates ("both return another gate — so they compose", [gates.py:154](src/qsim/gates.py#L154)). The algebra must close for blocks exactly as it does for gates. Fix in §2 below.
- **D2 — Two paths append to the history.** Gates append inside `Circuit._execute` ([circuit.py:605](src/qsim/circuit.py#L605)); measurements go through `Circuit._record` ([measure.py](src/qsim/measure.py), [circuit.py:562](src/qsim/circuit.py#L562)). Hooks must see *every* op, so unify: `_execute` ends with `self._record(op)`, and `_record` becomes the single place history grows (and where hooks fire). Behavior-neutral refactor; the full existing suite pins it.
- **D3 — `pointer_coupling` hand-rolls its conjugation** ([decoherence.py:277-287](src/qsim/decoherence.py#L277-L287)): H before, dephase, H after, with comments narrating the sandwich. Reimplement via `within` once it exists — the code should *say* "this is dephasing conjugated into another basis" in its structure, not only in its comments. TD1–TD7 must pass unchanged (they pin behavior); TT1 additionally pins the equivalence.

## 1. `qsim.within` (`combinators.py`)

```python
class WithinScope:
    """with qsim.within(V, *args, **kwargs): body — V now, body eagerly, V† on exit."""
```

Module-level `within(V, *args, **kwargs)` returning the scope; export from `qsim` `__init__`. Implementation on `__enter__`:

1. Resolve the circuit from `args` via the existing `_circuit_of` (reuse it — same rule as blocks: at least one `Qubit`/`Register` argument; otherwise raise `QsimError` with a message telling the user to pass the qubits V acts on as arguments, and why: the scope must know which circuit's tape to capture on).
2. Capture V by running it under a record push: `circuit._push_record(); V(*args, **kwargs); v_ops = circuit._pop_record()`.
3. Validate: any op in `v_ops` with `gate=None` (measurement) → raise `QsimError` — the teaching message: the basis change must be undone on exit, and measurement is the one operation that cannot be; nothing irreversible can be part of a conjugation's wrapper.
4. Emit the captured ops (`circuit._emit(op)` each) — V runs *now*, or lands in an enclosing scope's buffer, which is exactly right: an enclosing `control`/`adjoint`/block sees V, body, V† as ordinary ops and transforms them uniformly (correctness argued in design doc §4.5).

On `__exit__`: if an exception is passing through, emit nothing (consistent with `_Scope`: never run half a construct on the way out of an error — comment this). Otherwise emit `op.gate.adjoint_op(op)` for `reversed(v_ops)`.

Note what is deliberately absent: the body is **not** recorded. `within` is not a record-mode scope; the body executes eagerly and the state stays inspectable between its statements. Put that in the class docstring — it is the design's point, and the contrast with `control` (counterfactual ⇒ must record) belongs there too, in one sentence each.

Docstring also carries the two identities from design doc §4.5 — $(VUV^\dagger)^\dagger = VU^\dagger V^\dagger$ and the control-distribution fact — in the plain-language register the master plan requires. Do **not** implement the "uncontrolled basis change" optimization in this phase; TT3 asserts the uniform lifting matches the optimized *math*, not that the code takes the shortcut.

## 2. Closing the block algebra (`combinators.py`)

Replace the closure-returning `adjoint`/`controlled` with `Block`-returning versions. Approach: give `Block` an internal constructor parameter for an op-transformation pipeline, e.g. `Block(fn, name=..., transform=...)` where `transform: Callable[[list[Op]], list[Op]] | None` is applied to the recorded ops after `_record`:

```python
def adjoint(self) -> Block:
    # bell.adjoint() is a Block named "bell†" whose recording is bell's, reversed
    # and inverted. adjoint of adjoint composes two reversals — i.e. cancels.
def controlled(self, *controls: Qubit) -> Block:
    # named "C-bell" ("CC-bell" for two controls); prepends control ids to each op,
    # reusing ControlScope's transform (extract it to a shared function so the
    # self-control validation and the teaching message live in ONE place).
```

Requirements:

- Derived blocks are `Block` instances: chainable (`bell.adjoint().controlled(c)`), repr-able, and their names appear in `op.block` stamps and `block_counts()` under the derived name (`bell†`). Use `†` and `C-` prefixes; keep `full_name`-style spelled-out forms consistent with gates (`bell.adjoint().name == "bell†"`, check how gates spell theirs and match — see `src/qsim/CLAUDE.md` on the two-name convention).
- `bell.adjoint().adjoint()` must act identically to `bell` (TT8). Composing transforms is fine; detecting-and-cancelling is not required.
- `controlled` must run the same self-control validation `ControlScope` runs (control id appearing in the body's ops), producing the same `NoCloningError` message — via the shared extracted function, not a copy.
- Keep the current closure behavior for *call-time argument capture*: a derived block still records freshly at each call (classical args consumed then), so `bell.adjoint()(a, b)` and a stored `undo = bell.adjoint()` used later both work.

## 3. Checkpoint and rewind (`circuit.py`)

```python
@dataclass(frozen=True)
class Checkpoint:
    """A position on a circuit's tape, plus the allocation fingerprint valid there."""
    _history_len: int
    _next_id: int
    _n_qubits: int

def checkpoint(self) -> Checkpoint
def rewind(self, mark: Checkpoint) -> None
```

`rewind` walks `self._history[mark._history_len:]` newest-first and executes each op's inverse; the inverse ops are appended to the history like any other executed op (**the tape stays honest** — design doc §4.6; the docstring must carry the editor-undo analogy: the state returns, the record shows how). After the walk, the state equals the checkpoint state to float precision; nothing is truncated.

Raise `QsimError` (all with teaching messages) when:

- a combinator scope is open (`self._record_stack`): the tape and the state disagree mid-scope;
- `mark` came from a different circuit or `mark._history_len > len(self._history)` (a mark from a "future" that a prior rewind already unwound past — say so plainly);
- allocation changed: `self._next_id != mark._next_id or self.n_qubits != mark._n_qubits` — the suffix's ops refer to axes that must still exist and mean the same thing; releasing or allocating qubits re-numbers the world under the tape;
- any suffix op has `gate is None` (measurement): **the flagship message of the phase.** Measurement is the one operation with no inverse rule — it severed the tape exactly the way a non-differentiable op severs an autograd graph. Point forward: the eraser (notebook 06) rewinds *through* recorded information precisely because the record was kept coherent instead of measured.

Implementation note: reuse `op.gate.adjoint_op(op)` + `_execute` — no new kernel code. Check the messages before touching the state so a failed rewind changes nothing (validate the whole suffix first, then execute).

## 4. Hooks (`circuit.py`)

```python
class HookHandle:
    def remove(self) -> None: ...
def on_op(self, fn: Callable[[Op, Circuit], None]) -> HookHandle
```

- Hooks fire in `_record`, after the op is appended — so they see gates *and* measurements (D2 makes this true), in execution order, with the state already updated.
- Reentrancy guard: set a `self._in_hook` flag around hook invocation; `_emit` raises `QsimError` while it is set — hooks observe the tape, they do not write it (message: a hook that applied gates would put ops on the tape that no program line accounts for; if you want to transform a program, use blocks/scopes).
- `remove()` detaches; removing twice is a no-op. Iterate over a snapshot of the hook list so a hook removing itself mid-fire is safe.
- Keep it minimal: no priorities, no filtering — a hook that only wants measurements checks `op.name` itself.

## 5. Tests

**Acceptance — `tests/test_acceptance_tt1_tt8.py`:** implement TT1–TT8 exactly as specified in design doc §9 (they were written for this phase; tolerances and structural asserts included). Notes: TT1's third leg (pointer_coupling equivalence) doubles as the D3 regression test; TT3's kron comparison follows the T10 pattern (test-file-only, n ≤ 4); TT4 uses a seeded random block via the existing conftest helpers.

**Unit — `tests/test_tape.py`,** tests-as-documentation, 100% coverage of every new line and message:

- `within` with a `Register` argument; `within` with no qubit argument raises (message checked); `within` whose V measures raises; body exception → V† not emitted (assert history); `within` nested in `control` and in a block; `within(H, q)` twice nested (VW U W†V†).
- every `rewind` error path (scope open, foreign mark, stale mark after a deeper rewind, alloc since mark, ancilla release since mark, measurement in suffix — message substrings checked, "severed"/"inverse" wording present);
- `rewind` to a mark taken at history position 0; `rewind` twice to the same mark; checkpoint-rewind inside a notebook-style loop;
- hooks: fire-order matches history order; measurement ops delivered with `result` set; self-removal during fire; two hooks; the reentrancy raise;
- closed algebra: derived names in `block_counts()`; `adjoint` of a parametrized block negates angles (assert on recorded params); double-controlled naming.

**Regression:** the entire existing suite (T1–T10, TB, TD + unit tests) passes **unchanged** — D2 and D3 are behavior-neutral; any test edit must be reported as a red flag, not committed silently.

## 6. Notebook updates

**Notebook 04 (combinators) — three new sections at the end** (keep the existing narrative untouched; these continue it):

1. *"Conjugation: do, act, undo"* — the sandwich pattern with `within`; show the X-basis dephasing example; the $(VUV^\dagger)^\dagger$ identity demonstrated by printing `history` op names of an adjointed sandwich block.
2. *"The tape"* — the PyTorch analogy stated for a reader who may not know PyTorch (one sentence: "deep-learning libraries run operations immediately but keep a record for later transformations; so do we"); `checkpoint`/`rewind` demo: entangle, watch coherence die, rewind, watch it return; then *measure* and show `rewind` refusing, reading the error aloud — segue sentence to notebook 06's eraser: "keep the record coherent instead of measuring, and you can rewind through it".
3. *"Watching a program run"* — `on_op` hook collecting per-gate entanglement entropy into a list, plotted with matplotlib; note this is how `viz.entropy_trace` (Phase 6) works under the hood.

**Notebook 06 (decoherence) — one added markdown cell** in the eraser section, cross-referencing the new notebook-04 tape section: the eraser *is* a rewind that survived a coupling because nothing was measured. No code changes.

## 7. Definition of done

- TT1–TT8 + `test_tape.py` pass; **entire prior suite passes unchanged**; 100% coverage maintained; pyright/ruff clean; notebooks 01–06 all execute.
- D1–D3 fixed; `_record` is the single history-append site (grep proves it); `pointer_coupling` contains a `within` and no hand-rolled H-sandwich.
- Design doc §4.5/§4.6 docstring content present in the code (identities, tape-honesty, severed-tape message).
- Report "Decisions made", including exact derived-block naming and any message wording choices.

## Interface decisions to review with the owner (before building)

1. **Rewind tape semantics** — honest-append (recommended, specced above) vs truncate-to-mark. Present the tradeoff in one paragraph: honest-append keeps the tape a true record (diagrams, counts, entropy traces never lie) at the cost of `gate_counts()` including undo gates; truncation gives clean counts but a tape that pretends.
2. **Naming** — `checkpoint`/`rewind` (recommended; editor/VCS resonance) vs `save`/`restore` vs `mark`/`undo_to`.
3. **`within` argument form confirmation** with the final call examples from design doc §4.5 (this was provisionally settled in conversation; confirm against the real signatures).
4. **Hook signature** — `fn(op, circuit)` (recommended) vs `fn(op)` with the circuit closed over; show both in a live-entropy example.
