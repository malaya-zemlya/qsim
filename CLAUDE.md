# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`qsim` is a NumPy state-vector quantum simulator built for **learning quantum mechanics and quantum computing** — conceptual transparency over speed, always. It is two products at once: an importable library (`src/qsim/`) and a Jupyter-notebook learning environment (`notebooks/`). The owner is learning QM through this project: they know linear algebra and basic NumPy but **no quantum mechanics** — every explanation in code, notebooks, and conversation must be pitched accordingly, and non-obvious NumPy operations must be explained in comments.

## Authoritative documents (read in this order)

1. `plans/master-plan.md` — build workflow, **binding conventions** (pedagogy rules, testing rules, code style), risk register. Its Conventions section governs all code written here.
2. `plans/phase-*.md` — detailed per-phase build plans (Phase 0 scaffolding through Phase 6). Each is self-contained for a subagent and ends with "Interface decisions to review with the owner". Where a phase plan and the design doc conflict, **the phase plan wins**.
3. `qsim-design.md` — the design document: full API surface, algorithm specs, and the acceptance tests T1–T25 / TB1–TB3 / TD1–TD7 (§9), which are the specification.

## Build workflow

Work proceeds phase by phase (0 → 1 → 1.5 → 2 → 2.5 → 3 → 4 → 5 → 6; phases 7/8 unplanned until 6 ships). Before building any phase's public API, **present its interface to the owner as short usage examples and get approval** — this review step is mandatory, per the master plan. One commit per phase.

## Commands

Valid once Phase 0 (scaffolding) has been executed — if `src/` doesn't exist yet, Phase 0 hasn't run:

```bash
uv sync                          # install deps (Python 3.14)
uv run pytest -v                 # full suite; coverage of src/qsim/ enforced at 100%
uv run pytest tests/test_state.py -v                  # one file
uv run pytest tests/test_state.py::test_name -v       # one test
uv run pyright                   # must report 0 errors
uv run ruff check .
uv run jupyter execute notebooks/*.ipynb              # notebooks must run top-to-bottom
uv run jupyter lab               # interactive use
```

## Non-negotiable constraints

- **Never construct a 2^n × 2^n matrix inside `src/qsim/`.** Gates apply via tensor contractions (`tensordot`/`moveaxis`), diagonal broadcasting, or axis slicing on the `(2,)*n` state tensor. Tests may use `np.kron` for n ≤ 4, in test files only.
- **Bit convention:** `psi[b0, b1, ..., b_{n-1}]` is the amplitude of |b0 b1 … b_{n-1}⟩; **qubit 0 is the most significant bit**. Everywhere, no exceptions.
- **Qubit handles store stable ids, never axis numbers**; the `Circuit` owns the id→axis table (design doc §2.4). The Circuit is the qubit pool — no second owner of the state tensor.
- **`modexp` must be honestly compiled from reversible arithmetic** — no precomputed permutations, no lookup tables, nothing requiring foreknowledge of the answer (design doc §8.3). T24 structurally enforces this.
- **Error messages and comments teach.** This inverts the usual comment discipline: explanatory teaching comments are the product. Exceptions (`NoCloningError`, `DirtyAncillaError`, `DeadQubitError`) explain the physics, not just the misuse.
- **Never weaken an acceptance test** (tolerance, assertion, or scope) to make it pass.
- 100% line+branch coverage of `src/qsim/` is enforced; tests double as documentation (descriptive behavior-stating names, readable as a usage guide).
- No union return types in public APIs (e.g. `alloc()`/`alloc_many(n)`, never `Qubit | tuple`).
- Runtime deps are `numpy` and `matplotlib` only; matplotlib imported lazily inside `viz`/display hooks.

## Architecture (big picture)

- The n-qubit state is one `np.ndarray` of shape `(2,)*n`; **the axes are the tensor factors of the Hilbert space** — this identification is the central pedagogical object. `state.py` holds pure kernel functions over arrays; `circuit.py` owns the state, allocation table, and history; gates are module-level callables in `gates.py` that resolve their circuit from qubit handles.
- Execution is **eager** (gates run immediately) except inside combinator scopes (`control`, `adjoint`, `@qsim.gate` blocks), where ops are recorded, transformed, then executed (`combinators.py`). Controlled gates are implemented by *slicing the control axes*, not by bigger matrices.
- Everything impossible on real hardware lives behind `qc.inspect` (module `inspector.py` — named that to avoid shadowing stdlib `inspect`). The namespace boundary is itself pedagogy: everything inside it is cheating.
- Ancilla scopes **numerically verify** clean uncomputation on exit (`DirtyAncillaError`); decoherence (`decoherence.py`) is unitary coupling to marked environment qubits that are never traced out — the reduced view is a choice made by the Inspector, not an operation on the state.
- Physics framing throughout: the classical world *emerges from* the quantum one (decoherence, einselection) — prefer that orientation in explanations over "quantum weirdness added to classical".
