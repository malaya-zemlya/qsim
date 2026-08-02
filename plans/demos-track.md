# Demo notebooks track — integration plan

The owner's `qsim-demo-notebooks.md` (repo root) is the authoritative content plan for the demo gallery: five tracks (A–E) of standalone, single-punchline notebooks. This file is the thin layer that connects it to the build machinery. **Read both before building any demo batch.**

## How demos relate to the numbered course

- The numbered series (`notebooks/01-…` … `09-…`) is the *course*: a guided sequence, prose-first, each building on the last.
- Demos live in **`notebooks/demos/`**, named as in the owner's doc (`one_qubit_playground.ipynb`, …), prefixed by track letter only in the doc, not the filename. They are *exhibits*: self-contained, one punchline each, stated in the first cell.
- Overlap with the course is expected and acceptable — same physics, different format. A demo may link to the course notebook that teaches its background, and should, in its opening cell.

## Conventions (additive to master-plan Conventions, demos only)

- Seeded RNG throughout; runs top-to-bottom in **< 30 s**; nothing over ~12 qubits.
- Every demo ends with an **"Assertions" cell** numerically re-checking its central claim — demos double as slow acceptance tests. The full-suite check `uv run jupyter execute notebooks/demos/*.ipynb` joins the phase verification gates once the first batch exists.
- Master-plan pedagogy rules apply unchanged (audience knows linear algebra, no QM; numpy explained; ~60% markdown).
- Pseudocode in `qsim-demo-notebooks.md` uses design-doc-era API — adapt to real signatures; the *shape* of each demo is the requirement (the owner's doc says this; it governs).

## Build batches and gates

| Batch | Demos | Gated on | Notes |
|---|---|---|---|
| AB | A1–A3, B1–B4 | Phase 2.75 shipped | B4 (Wigner's friend) should use `within`/adjoint-of-block for the uncompute, and may use `checkpoint`/`rewind` as a second telling of the same story. |
| B5 | `horizon.ipynb` | **`escape` API** (below) | Do not build before the API lands. |
| E0 | `l1_vs_l2.ipynb` | Phase 2.5 (shipped) | Owner spec verbatim in `plans/demo-l1-vs-l2-spec.md`; it governs. `csim` stays notebook-local (no `qsim.contrib` module without an interface review). |
| C | C1–C4 | Phase 3 | C4 is T17's demo form; coordinate with Phase 5's T17 test (same circuit family, dtype sweep) — build C4 in the Phase 5 window if the period-finding state is wanted, or earlier with a hand-prepared periodic state, clearly labeled. |
| D | D1–D4 | Phase 5 | D3 uses the same escape hatch as T18. |
| E | E1–E4 | Phase 5 | E4: no scipy — build `logm` by hand from `eigh` in the notebook (eigendecompose ρ, take −log of eigenvalues; comment the numpy). E2/E3 mix qsim with raw numpy; that is fine in notebooks (the 2^n×2^n ban applies to `src/qsim/` only). |
| E5 | `error_correction_teaser.ipynb` | Phase 7 | Treat E5 (and the trajectories it needs) as the scoping input for Phase 7 planning — demos drive the API. |

## Feature request queue (from the demos, per the owner's "demos drive the API")

1. **`qc.escape(env)` + `EscapedQubitError`** (for B5): marks environment qubits as beyond the horizon — any later gate touching them raises. Small addition to design doc §4.4 and `errors.py`; the error message carries the FAPP-vs-in-principle irreversibility point. Needs an interface review (naming: `escape` vs `release_to_environment`; whether escaped qubits still count in `n_qubits`; whether `inspect` may still see them — recommendation: yes to both, the *program* can't touch them but the bookkeeping view remains, that asymmetry is the pedagogy). Schedule: ride along whichever phase is next at review time, as a mini-addendum.
2. **Trajectories** (for E5): already sketched under Phase 7 in the master plan; E5's shape defines the minimum viable version.

## Workflow

Each batch is built by a subagent given: `qsim-demo-notebooks.md`, this file, `plans/master-plan.md` (Conventions), `notebooks/CLAUDE.md`, and read access to the shipped course notebooks for cross-linking. Interface review before a batch is light: confirm any API friction the demos surfaced (that is their job) rather than the notebooks' content. One commit per batch (`Demos: track AB`, …).
