# Working in `notebooks/`

Pedagogy rules are binding and live in `plans/master-plan.md` → "Audience and pedagogy"
(~60% markdown, concept before code, every QM term introduced at first use). This file
covers only the mechanics, which are easy to get wrong.

## Building a notebook

- **Never hand-write `.ipynb` JSON.** Write a Python script (in the scratchpad, not the
  repo) that builds the notebook with `nbformat`, and run it.
- Run `nbformat.validator.normalize()` before writing, or cells ship without an `id`
  field and `nbformat.validate` warns.
- **Ship with no stored outputs**: empty `outputs`, `execution_count: null`.

## Gotchas that have already bitten

- **`ruff check .` lints notebooks.** Cells need no unused imports and must stay under
  100 columns, same as `src/`. Import only the gates a notebook actually uses.
- **Assign the figure**: `fig = viz.amplitudes(qc)`, not a bare `viz.amplitudes(qc)`.
  Every `viz` function returns its `Figure`, and a bare call renders twice — once from
  the inline backend and once as the cell result.
- **Draw seeds from one master RNG**, not `range(n)`. Consecutive seeds give correlated
  PCG64 streams; `Circuit(seed=s) for s in range(400)` produced a 2.4σ-biased coin.
  Use `np.random.default_rng(0).integers(0, 2**32, size=n)`.
- Verify every prose claim by running the snippet first. If the library disagrees with
  the markdown, fix the markdown — or report a real library bug, never paper over one.

## The API as taught

`a, b = qc.alloc_many(2)` (not `Circuit(2)`), angles are keyword-only
(`Rz(a, theta=…)`), subsets are lists of qubit **handles** (`entanglement_entropy([a])`,
never `[0]`), and `reg[0]` is the most significant bit. Gates have long aliases —
`Hadamard is H` — usable wherever they read better.

## Gate

`uv run jupyter execute notebooks/*.ipynb` must succeed. It is part of every phase's
definition of done.
