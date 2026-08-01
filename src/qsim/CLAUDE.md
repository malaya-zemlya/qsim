# Working in `src/qsim/`

The non-negotiable constraints (no 2^n × 2^n matrices, bit convention, handles hold ids,
teaching error messages, numpy/matplotlib only) are in the root `CLAUDE.md`. This file
covers what is easy to trip over while editing these modules.

## Import direction

`circuit.py` imports `measure`, `gates` and `inspector` **inside the methods that use
them**, not at module scope. The dependency genuinely runs that way — those modules build
on the types `circuit.py` defines — so a top-level import is circular. Any new module that
acts *on* a `Circuit` should import from `circuit.py` at the top and be imported lazily by
it in return. Phase 2's `combinators.py` and Phase 2.5's `decoherence.py` will hit this.

`matplotlib` is imported inside each `viz.py` function so the core imports cleanly in a
bare interpreter. Keep it that way.

## Gates

Every gate carries two names: `gate.name` is the short symbol and is what `Op.name`,
`gate_counts()` and circuit diagrams use; `gate.full_name` is spelled out, and each gate is
also exported under it (`Hadamard is H`). `gate.label` combines them for errors and reprs.
When adding a gate, register it in `GATES` — Phase 2 replays recorded history by name.

Controlled gates come from `.controlled()`, which only changes how the state is sliced, and
diagonal gates (Z, S, T, Rz, Phase, CZ, CPhase) must route through `apply_diag` /
`apply_controlled_diag` so that "diagonal ⇒ phases only" stays a structural fact rather
than a claim.

## Precision

Gate data is cast to the circuit's dtype before use (`_GateBase._apply`). Without it a
`complex128` matrix silently promotes a `complex64` state and quietly undoes the
single-precision experiment of design doc §9 (T17).

## Style

Every module docstring opens by stating the physical fact that module makes concrete.
Explanatory comments are the product here, not noise — `tensordot`, `moveaxis`, `reshape`
whenever axis order matters, `vdot`, `svd` and broadcasting tricks each get a comment
saying what they compute *in terms of the math*.
