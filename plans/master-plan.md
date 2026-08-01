# qsim — Master Build Plan

This document coordinates the construction of `qsim` as specified in `qsim-design.md` (the design doc, which is authoritative on *what* to build; these plans are authoritative on *how and in what order*). Each phase has its own detailed plan in this directory, written so a Sonnet/Opus subagent can execute it with no other context beyond the files it names.

## The two products

1. **A library** (`qsim/`) you can import to write and run small quantum programs.
2. **An interactive learning environment** (`notebooks/`) that teaches quantum mechanics and quantum computing using that library.

Neither is secondary. Every phase ships both code and its notebook.

## Phase index

| Phase | Plan file | Delivers | Tests | Notebooks |
|---|---|---|---|---|
| 0 | `phase-0-scaffolding.md` | uv/pyproject/pyright/jupyter setup, empty package skeleton | smoke only | — |
| 1 | `phase-1-core.md` | state tensor, Circuit/Qubit/Register, gates, measurement, Inspector, basic viz, rich display | T1–T7 | 01, 02 |
| 1.5 | `phase-1.5-entanglement-demos.md` | CHSH, teleportation, superdense coding | TB1–TB3 | 03 |
| 2 | `phase-2-combinators.md` | record mode, `control`, `adjoint`, ancilla scopes, `@qsim.gate` | T8–T10 | 04 |
| 2.5 | `phase-2.5-decoherence.md` | environment qubits, noise couplings, quantum eraser | TD1–TD7 | 05 |
| 3 | `phase-3-qft-phase-estimation.md` | QFT, approximate QFT, phase estimation, semiclassical PE | T11–T15 | 06 |
| 4 | `phase-4-arithmetic.md` | reversible adders, modular arithmetic, modexp | T16 | — |
| 5 | `phase-5-shor.md` | Shor's algorithm end-to-end, T18 demonstration | T17–T19, T22–T25 | 07 |
| 6 | `phase-6-grover-dj-viz.md` | Grover, Deutsch–Jozsa, circuit diagrams, entropy trace, widgets | T20, T21 | 08 |

Phases run strictly in order. Note one deliberate deviation from the design doc: **T17 (precision comparison) is built in Phase 5, not Phase 3**, because it needs the period-finding circuit that only exists after Phase 4. The design doc's phase list assigned it to Phase 3; these plans override that.

Phases 7 (quantum trajectories, error correction, quantum Darwinism, Trotterized Ising — "emergence of the classical") and 8 (density-matrix backend) are deliberately **not planned yet**. Their plans get written when Phase 6 is done, informed by the real codebase. The design doc §11 sketches both.

## Per-phase workflow

Every phase follows the same five steps:

1. **Interface review (with the project owner, before any code).** Claude presents the phase's concrete public interface as *short usage examples* — "here is what your code will look like" — not as abstract signatures. The owner approves or adjusts. Each phase plan has an "Interface decisions to review" section listing exactly what to present. Nothing in a phase's public API is built before this review.
2. **Build.** A subagent executes the phase plan. The subagent must be given: the phase plan file, `qsim-design.md`, and this file's Conventions section (or this whole file). Subagents implement code + tests + notebook together, not code first and tests as an afterthought — the tests in each plan are the specification.
3. **Verify.** All three gates must pass locally:
   - `uv run pytest -v` — all tests green, including all previous phases' tests, with **100% coverage of `src/qsim/`** (coverage is enforced by the pytest config; see Conventions → Testing).
   - `uv run pyright` — zero errors.
   - `uv run jupyter execute notebooks/<this phase's notebooks>` — executes top to bottom without error. (If `jupyter execute` is unavailable, `uv run jupyter nbconvert --to notebook --execute --stdout <nb> > /dev/null` is the fallback.)
4. **Review.** Claude reads the diff and checks it against the phase plan's "Definition of done" and the design doc's constraints (especially: no 2^n×2^n matrices in the library, teaching-quality error messages, pedagogical comments present).
5. **Commit.** One commit per phase (plus fixups), message `Phase N: <summary>`.

## Conventions (binding on every subagent)

### Audience and pedagogy — the most important section

The project owner — the person who will read every line of this code and every notebook cell — **knows linear algebra (vectors, matrices, complex numbers, eigenvalues) but no quantum mechanics and only basic NumPy.** Therefore:

- **Every QM concept is introduced at first use**, in notebooks *and* in docstrings. Never write "the Bloch vector" or "unitary" or "the Born rule" for the first time without a one-or-two-sentence explanation. Dirac notation (|0⟩, ⟨ψ|, ⊗) must be defined before it is used (notebook 01 does this; later notebooks may use it freely).
- **Every NumPy operation beyond elementwise arithmetic and plain indexing gets an explanatory comment** stating what it computes in terms of the math, not just what the function is called. This applies to: `tensordot`, `moveaxis`, `einsum`, `kron`, `vdot`, `outer`, `svd`, `eigh`, `reshape` (whenever the axis-ordering matters), fancy indexing, and broadcasting tricks. Example of the required style:

  ```python
  # Contract U's column index (axis 1) with the state's axis k. This is
  # matrix-vector multiplication applied along one axis of the tensor:
  # every amplitude pair (a_0, a_1) along axis k becomes U @ (a_0, a_1).
  # tensordot puts the new (row) index first, so move it back to position k.
  psi = np.tensordot(U, psi, axes=([1], [k]))
  psi = np.moveaxis(psi, 0, k)
  ```

- **Teaching comments are the product, not noise.** The usual rule "don't write comments explaining what code does" is inverted for this project: explanatory comments and docstrings are a stated design goal (design doc §12). Every module docstring states the physical fact the module makes concrete.
- **Error messages teach.** `NoCloningError` explains the no-cloning theorem; `DirtyAncillaError` explains why leftover entanglement destroys interference. The design doc gives the required content; write full sentences.
- **Notebooks are prose-first.** A notebook cell pattern of (markdown explaining the idea) → (small code cell) → (markdown interpreting the output) throughout. Target roughly 60% markdown by volume. Each notebook opens with "What you will learn" and closes with "What you now know" + pointers to the next notebook.

### Code style

- Python 3.14, type hints throughout, `uv run pyright` must pass with zero errors (configuration set in Phase 0).
- `os.path` over `pathlib`; f-strings over `.format()`.
- `Qubit` and `Register` must be distinct types a checker can tell apart. No union return types — a function returns one thing (e.g. `alloc()` vs `alloc_many(n)`).
- **Never construct a 2^n × 2^n matrix inside `qsim/`** — not in any code path, ever. Tests may build small ones with `np.kron` for n ≤ 4, in the test file only.
- Prefer a readable loop that matches the physics over a clever vectorization that obscures which axis is which.
- Runtime deps: `numpy` and `matplotlib` only. `matplotlib` imported lazily (inside functions/methods in `viz.py` and display hooks).
- Bit convention, stated once here and repeated in `state.py`'s docstring: `psi[b0, b1, ..., b_{n-1}]` is the amplitude of basis state |b0 b1 … b_{n-1}⟩ and **qubit 0 is the most significant bit** of the integer reading. All code and all tests use this convention; T11's bit-reversal handling documents it by example.

### Testing

- pytest, run as `uv run pytest -v`. Test files `tests/test_*.py`, fixtures over setup/teardown.
- `tests/conftest.py` provides a `rng_seed` fixture and helpers; circuits in tests are constructed with `Circuit(seed=...)` for determinism.
- Numeric tolerances come from the design doc test specs (T1–T25, TB, TD) and are part of the spec — do not loosen a tolerance to make a test pass; find the bug.
- Each acceptance test carries a docstring saying what physical fact it verifies.
- **100% line coverage of `src/qsim/`, enforced** (`pytest-cov`, `fail_under = 100`, configured in Phase 0). This is a teaching framework: an untested line is an undocumented line. Every error path is exercised (every `raise` has a test that triggers it and checks the message teaches), every branch of every gate/kernel runs under test. `# pragma: no cover` is allowed only for genuinely unreachable defensive lines and `TYPE_CHECKING` blocks, never as a shortcut — each use must be justified in the subagent's report. Viz code is covered too: matplotlib runs headless under the `Agg` backend in tests (set in `conftest.py`); call every plot function, assert basic figure structure (number of bars, axis labels), close figures.
- **Tests double as documentation.** They are the second reading surface after the notebooks, so optimize them for a human reader learning the library: descriptive names that state the behavior (`test_measuring_one_half_of_a_bell_pair_collapses_the_other`), arrange/act/assert layout, small focused cases over parametrized walls, a docstring on anything non-obvious, and explicit edge-case tests (empty register, 1-qubit circuit, repeated measurement of the same qubit, gate on a just-measured qubit, θ=0 and θ=2π rotations, …). Someone should be able to learn how to *use* qsim by reading `tests/` alone.

### Subagent ground rules

- Do not add dependencies, rename public API, or deviate from the phase plan's signatures without flagging it in your final report — those decisions belong to the interface-review step.
- If the plan under-specifies something and a reasonable local choice exists, make it and **list it in your final report** under "Decisions made".
- If you hit a genuine contradiction between the phase plan and `qsim-design.md`, the phase plan wins (it is newer); report the contradiction.
- Never "fix" a failing acceptance test by weakening it.

## Known risks (tracked here, addressed in the phase plans)

1. **Bit-ordering bugs** — the single most common quantum-simulator bug class. Mitigation: the convention lives in one place (`state.py` docstring), T11 pins it against `np.fft`, and Phase 1's plan includes targeted ordering tests.
2. **Axis lifecycle** (design doc §2.4) — handles hold stable ids; the Circuit owns the id→axis table. Phase 1 builds it; Phase 2's ancilla deallocation is the first real stress test.
3. **Phase 4 adder architecture** — the design doc asks for both Cuccaro (ripple-carry, Toffoli-based) adders *and* Beauregard's 2n+3-qubit Shor construction, but Beauregard's construction is built on Draper's Fourier-space adder, not Cuccaro's. The Phase 4 plan lays out the resolution (build both; Cuccaro for teaching + T16, Draper/Beauregard for Shor's qubit counts) and the exact tradeoff to present at the Phase 4 interface review, phrased for a non-expert.
4. **Performance cliff in Phase 5** — Shor on N=21 is 13 qubits and thousands of gates; fine. If the interface review instead chooses all-Cuccaro modexp (~20+ qubits for N=15), runtimes grow to minutes. The Phase 4 plan quantifies this.
5. **`jupyter execute` availability** — provided by `nbclient` (installed with jupyterlab). Fallback command in the workflow section above.

## Phases 7 and 8 (sketches only — plan when Phase 6 ships)

- **Phase 7 — Emergence of the classical.** Stochastic Pauli-error trajectories averaged over runs; 3-qubit repetition code, then Shor-9/Steane-7 with syndrome extraction; the quantum-Darwinism demo (`pointer_coupling` to many environment qubits + `inspect.mutual_information` showing redundant records); Trotterized transverse-field Ising evolution. Design doc §11 Phase 7.
- **Phase 8 — Density-matrix backend (optional).** A second backend storing ρ as shape `(2,)*2n` with Kraus channels applied directly. Only worth building if exact channel fidelities are wanted; design doc §11 Phase 8 explains why it is deliberately last.
