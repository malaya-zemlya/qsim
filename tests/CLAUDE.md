# Working in `tests/`

Testing conventions are binding and live in `plans/master-plan.md` → "Testing". This file
covers the local specifics.

## Randomness

**`for s in range(n)` is fine as a source of seeds.** `np.random.default_rng(s)` puts the
seed through NumPy's `SeedSequence`, which is *designed* so that small consecutive seeds
give independent streams. Measured here over 200,000 sequential seeds: z = +0.99 on a
fair-coin count, χ² = 10.1 on 9 degrees of freedom (critical value 16.9). There is no
correlation to design around.

**What does bite is tightening a statistical bound around the block of seeds you happened
to run.** 400 sequential seeds give a fair-coin count anywhere in roughly 180–220, and
`range(400)` lands on 176 — outside 2σ, purely by chance. So give statistical assertions
real headroom, or use enough samples that the bound is many sigma away. A test that only
passes for one lucky block of seeds is a test that will fail on someone else's machine
the first time the sequence changes.

(An earlier version of this file claimed consecutive seeds were correlated, on the
strength of a single 2.4σ observation. Repeating that measurement across 40 blocks gave a
spread of z with sd 1.03 and |z| > 2 in 2 of 40 blocks — exactly what an unbiased
generator does. The claim was wrong.)

Note also that `inspect.sample()` draws from a separate stream (`Circuit._sample_rng`), so
it deliberately does *not* disturb a seeded sequence of `measure()` calls.

## Fixtures (`conftest.py`)

`qc` (empty seeded circuit), `bell_pair` (returns `(circuit, a, b)`), `seed`, and
`random_state` (a factory: `random_state(n)` gives a normalized random tensor). The `Agg`
matplotlib backend is set here, so viz tests run headless.

## Coverage

100% line **and branch** of `src/qsim/` is enforced by the pytest config, so a new `raise`
needs a test that triggers it — and asserts on its *message*, not just its type, since the
error text is part of the specification. `# pragma: no cover` is for genuinely unreachable
lines only, and each use must be justified.

## Local conventions

- `np.kron` and full 2^n × 2^n matrices are allowed **here only**, for n ≤ 4, to check the
  library against an independent construction. Never in `src/`.
- Name tests for the behavior they state (`test_measuring_one_half_of_a_bell_pair_collapses_the_other`).
  Someone should be able to learn the library by reading this directory.
- Acceptance tests (T1–T25, TB, TD) come from `qsim-design.md` §9 and carry a docstring
  saying which physical fact they pin down. **Never weaken one to make it pass.**
- Reaching into privates (`q._live = False`, `qc._psi`) is fine when testing a guard that
  has no public trigger yet — comment it as such.
