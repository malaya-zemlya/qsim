# Working in `tests/`

Testing conventions are binding and live in `plans/master-plan.md` → "Testing". This file
covers the local specifics.

## Randomness

**Draw seeds from one master RNG, never from `range(n)`.** Consecutive seeds produce
correlated PCG64 streams: `Circuit(seed=s) for s in range(400)` gave a 2.4σ-biased coin
on a fair measurement. Use `np.random.default_rng(0).integers(0, 2**32, size=n)` and pass
each as `Circuit(seed=int(s))`. This will matter again for CHSH sampling (Phase 1.5) and
Shor (Phase 5).

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
