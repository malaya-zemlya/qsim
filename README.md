# qsim

`qsim` is a NumPy state-vector quantum simulator for learning quantum mechanics and quantum
computing by building and inspecting circuits. It is two things at once: an importable library
for writing small quantum programs, and a set of Jupyter notebooks that teach the physics using
that library — every design decision favors the option that makes a physical fact visible,
checkable, or breakable, rather than the option that runs fastest.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14.

```bash
uv sync
```

## Test

```bash
uv run pytest -v
```

## Notebooks

```bash
uv run jupyter lab
```

The notebooks in [notebooks/](notebooks/) are the learning environment, meant to be read in
order; the library in [src/qsim/](src/qsim/) exists to serve them.

## Project documents

- [qsim-design.md](qsim-design.md) — the design document: API surface, algorithms, acceptance tests.
- [plans/master-plan.md](plans/master-plan.md) — build order and the conventions all code follows.
