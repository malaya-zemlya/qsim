# Phase 1 — Core: state tensor, circuits, gates, measurement, inspection, basic viz

**Read first:** `plans/master-plan.md` (Conventions — binding), `qsim-design.md` §1, §2, §3.1, §5, §6, §10.1, and the §9 specs for T1–T7.

**Goal:** after this phase a user can allocate qubits, apply gates, measure, inspect everything (amplitudes, probabilities, entanglement entropy, Bloch vectors), and *see* states rendered as bar charts, Bloch spheres, and Dirac-notation LaTeX in Jupyter.

**Files created:** `src/qsim/state.py`, `circuit.py`, `gates.py`, `measure.py`, `inspector.py`, `viz.py`, updates to `__init__.py`; `tests/conftest.py`, `tests/test_state.py`, `test_circuit.py`, `test_gates.py`, `test_measure.py`, `test_inspector.py`, `test_acceptance_t1_t7.py`; `notebooks/01-states-and-gates.ipynb`, `notebooks/02-entanglement.ipynb`.

---

## 1. `state.py` — the tensor and the gate kernels

This module is pure functions over `np.ndarray`; it knows nothing about `Circuit` or `Qubit`. Everything takes `(psi, axis indices, ...)` and returns a new array (or mutates and returns — pick returning new arrays; simpler to reason about, and performance is a non-goal).

**Module docstring** must state (design doc §1.1): the state of n qubits is one array of shape `(2,)*n`; the axes *are* the tensor factors of the Hilbert space; `psi[b0,...,b_{n-1}]` is the amplitude of |b0…b_{n-1}⟩ with qubit 0 the most significant bit. Explain in the docstring what "amplitude" means for a reader who knows linear algebra but no QM: the state is a unit vector in C^(2^n); the squared magnitude of each component is the probability of finding that bit pattern when you measure (this is the Born rule, introduced properly in `measure.py`).

Functions (all take `psi: np.ndarray` first):

```python
def zero_state(n: int, dtype: np.dtype = np.complex128) -> np.ndarray
    # shape (2,)*n, all amplitude on index (0,...,0)

def apply_1q(psi, U: np.ndarray, k: int) -> np.ndarray
def apply_2q(psi, U: np.ndarray, j: int, k: int) -> np.ndarray
    # U shape (2,2,2,2) indexed [out_j, out_k, in_j, in_k]
def apply_diag(psi, phases: np.ndarray, k: int) -> np.ndarray
    # phases shape (2,): multiply the k=0 slice by phases[0], k=1 slice by phases[1]
def apply_controlled(psi, U, controls: list[int], target_axes: list[int]) -> np.ndarray
    # see below
def measure_axis(psi, k: int, rng) -> tuple[int, np.ndarray]
    # implemented here, wrapped by measure.py
```

Kernel mechanics — copy the design doc §1.2 code exactly, with the master-plan-style comments explaining `tensordot`/`moveaxis`. For `apply_2q`, explain that `tensordot(U, psi, axes=([2,3],[j,k]))` contracts U's two *input* indices against the state's axes j and k, and the two output indices land at axis positions 0 and 1, then `moveaxis([0,1],[j,k])` puts them back.

`apply_diag`: reshape `phases` to broadcast along axis k only:

```python
# Reshape (2,) -> (1,...,1,2,1,...,1) with the 2 at position k, so NumPy
# broadcasting multiplies every amplitude whose k-th index is b by phases[b].
# A diagonal gate can't move probability between basis states — it only
# rotates each amplitude's complex phase. That is a structural fact here:
# this code path cannot change any |amplitude|.
shape = (1,) * k + (2,) + (1,) * (psi.ndim - k - 1)
return psi * phases.reshape(shape)
```

`apply_controlled` — the §1.3 slicing construction, generalized to any number of controls (Phase 2 relies on this, build it fully now):

```python
sl = [slice(None)] * psi.ndim
for c in controls:
    sl[c] = 1                      # select the subspace where every control is |1>
sub = psi[tuple(sl)]               # a VIEW with the control axes dropped
# Axis renumbering: dropping axis c shifts every later axis down by one.
# adjusted_target = t - (number of controls with index < t)
...apply U to sub at the adjusted target axes...
out = psi.copy()
out[tuple(sl)] = new_sub           # write the transformed subspace back
return out
```

Docstring must include the identity CU = |0⟩⟨0|⊗I + |1⟩⟨1|⊗U and explain it in words: "a controlled gate does nothing on the part of the state where the control is 0, and applies U on the part where the control is 1; slicing the array along the control axis is literally that decomposition."

Also in this module: `set_dtype()` / a module-level default-dtype mechanism per design doc §1.5, and the §1.5 precision notes in the docstring (copy them from the design doc, then add one clarifying sentence each for the non-expert).

**Unit tests for this module (`test_state.py`)** — beyond the acceptance tests, pin the bit ordering early:
- `apply_1q` with X on qubit 0 of a 2-qubit |00⟩ gives amplitude 1 at index `[1,0]`, and `state_vector` (once inspector exists) index `0b10 = 2` — qubit 0 is the MSB.
- `apply_2q` with a CNOT tensor equals `apply_controlled` with X.
- `apply_diag` with Z never changes `np.abs(psi)` (assert exactly, not approximately).
- Every kernel on a random normalized state preserves `np.linalg.norm(psi)` to 1e-14. Comment: unitary means norm-preserving; norm = total probability = 1.

## 2. `circuit.py` — Circuit, Qubit, Register, the id→axis table

Implement per design doc §2 with the §2.4 axis-lifecycle rule: a `Qubit` stores a stable integer id; `Circuit` owns `_axis_of: dict[int, int]`. Every operation resolves id→axis at call time.

```python
class Circuit:
    def __init__(self, n: int = 0, *, name: str = "", dtype=np.complex128,
                 seed: int | None = None): ...
    def alloc(self) -> Qubit
    def alloc_many(self, count: int) -> tuple[Qubit, ...]
    def register(self, size: int, *, name: str = "") -> Register
    def measure(self, q: Qubit) -> int
    def measure_all(self, reg: Register) -> int
    def reset(self, q: Qubit) -> None
    @property
    def n_qubits(self) -> int
    @property
    def history(self) -> list[Op]
    def gate_counts(self) -> dict[str, int]
    def depth(self) -> int
    @property
    def inspect(self) -> Inspector
```

- `Circuit(n)` pre-allocates n qubits (equivalent to `alloc_many(n)` at construction); `Circuit()` starts empty.
- Growing the tensor on alloc:

  ```python
  # Tensor product with a fresh qubit in state |0>. np.multiply.outer(psi, [1,0])
  # appends one new axis of length 2 whose index-0 slice is a copy of psi and
  # index-1 slice is zero — i.e. the new qubit is |0> and unentangled.
  self._psi = np.multiply.outer(self._psi, np.array([1, 0], dtype=self._dtype))
  ```

  New qubit's axis = old ndim. Special case n=0→1 (start from `zero_state(1)`).
- `Qubit`: `__slots__`, identity `__eq__`/`__hash__`, `__copy__`/`__deepcopy__` raise `NoCloningError`, **no** `.state`/`.value`/`__bool__`. Class docstring: copy the "handle to an axis" paragraph from design doc §2.2 verbatim — it is the most important idea in the object model. Give it a helpful `__repr__` like `<Qubit q3 of Circuit 'bell'>`.
- `NoCloningError` message at the copy site must explain: "the no-cloning theorem: there is no physical operation that copies an unknown quantum state. Copying the handle would suggest otherwise. If two variables should refer to the same qubit, plain assignment does that."
- Liveness: `Qubit._live` flag; every gate/measure checks it and raises `DeadQubitError` (used in earnest in Phase 2's ancilla scopes; the flag and check exist now).
- `Op` (dataclass in this module or `circuit.py`-adjacent): `name: str`, `qubit_ids: tuple[int, ...]`, `params: tuple[float, ...]`, `controls: tuple[int, ...] = ()`. History append happens in the gate-application path. `depth()`: greedy layering — walk history, a gate starts a new layer iff it shares a qubit with the current layer.
- `Register(Sequence[Qubit])` per design doc §2.3, including `reversed()`, `concat()`, and `encode(value)` (X gates to set |value⟩ from |0…0⟩; raise if the register's qubits are not all in |0⟩ — check via the inspector's probability of the all-zeros outcome on those qubits). Slice returns a `Register` view. Overload `__getitem__` (`int → Qubit`, `slice → Register`) with `@overload` so pyright resolves both.
- The measurement functions delegate to `measure.py` logic (see below) but live on `Circuit` as the public API.

## 3. `gates.py`

Gates are callables that mutate the circuit and return None (design doc §3). Implement a small internal `Gate` class; module-level instances are the public API:

```python
H, X, Y, Z, S, T, SX, CNOT, CZ, SWAP, Toffoli, Fredkin      # fixed gates
Rx, Ry, Rz, Phase, CPhase                                    # parametrized: Rz(q, theta=...)
```

Each `Gate` carries: `name`, its action (a (2,2) matrix, a diagonal `(2,)` phase array, or a control structure over another gate), arity, and — used from Phase 2 on but **declared now** — how to invert it (`Rz(θ)⁻¹ = Rz(−θ)`; fixed gates map to their own inverse or a named partner) and how to control it. Diagonal gates (Z, S, T, Rz, Phase, CZ, CPhase) must route through `apply_diag`/`apply_controlled`-with-diag — design doc §1.4 makes "diagonal ⇒ phases only" structural.

Validation on every call (design doc §3.1), each with a teaching message:
- duplicate qubit arguments → `NoCloningError` ("a qubit cannot control an operation on itself…"),
- dead handle → `DeadQubitError`,
- handles from different circuits → `QsimError` subclass or `ValueError` with explanation.

Gate matrices: write them as module-level constants with a comment giving the matrix in conventional form. For each gate's docstring, one sentence of physics for the newcomer: e.g. H — "rotates |0⟩ to the equal superposition (|0⟩+|1⟩)/√2; it is how 'both at once' enters a computation"; T — "a 45° phase rotation; the non-Clifford ingredient that makes quantum computation hard to simulate classically" (fine to state without proof).

`CNOT`, `CZ`, `CPhase`, `Toffoli`, `Fredkin` are implemented via `apply_controlled` (slicing), not via 4×4/8×8 `apply_2q` matrices — the design doc treats the slicing construction as pedagogy. (`SWAP` may use `apply_2q`, or three CNOTs; pick the explicit `apply_2q` tensor and note in its docstring that SWAP = three alternating CNOTs — notebook 01 shows that equivalence.)

## 4. `measure.py`

Design doc §6. `measure_axis(psi, k, rng)`:

```python
# Probability that qubit k reads 1: sum of |amplitude|^2 over the half of the
# state where axis k has index 1. This is the Born rule — probabilities are
# squared magnitudes of amplitudes.
sl = [slice(None)] * psi.ndim; sl[k] = 1
p1 = float(np.sum(np.abs(psi[tuple(sl)]) ** 2))
outcome = 1 if rng.random() < p1 else 0
# Collapse: zero out the branch that didn't happen, then rescale so total
# probability is 1 again. This is a real projection of the joint state — the
# other qubits' conditional state is now correct automatically.
...
psi_after /= np.sqrt(p_outcome)
```

The docstring must explain *why* projection of the joint state (rather than sampling a marginal) matters: after measuring one qubit of a Bell pair, the other qubit's state must actually be updated — T3 (GHZ) tests exactly this. `reset(q)`: measure, then apply X if the outcome was 1; docstring notes this is how real hardware returns a qubit to |0⟩. RNG: `Circuit(seed=...)` → `np.random.default_rng(seed)` stored on the circuit; `measure_all` measures sequentially from `reg[0]`, returns the integer with `reg[0]` as MSB, and records per-qubit outcomes in the history.

## 5. `inspector.py`

Design doc §5 — module named `inspector.py`, class `Inspector`, accessed as `qc.inspect`. Class docstring: "everything in this namespace is physically impossible on real hardware — you cannot read amplitudes off a real quantum computer; you can only measure. The boundary of this namespace is the boundary between what the math knows and what an experiment can extract."

Implement the full §5 list. Implementation notes for the non-obvious ones:

- `state_vector()`: `psi.reshape(-1)` — comment that C-order reshape makes axis 0 the slowest-varying index, i.e. qubit 0 is the MSB of the flat index, matching the project convention.
- `amplitude("0101")`: index the tensor with the tuple of bits.
- `sample(shots)`: `rng.choice(2**n, size=shots, p=probabilities)` → `Counter` of bitstrings; non-collapsing (works on a copy; the state is untouched).
- `reduced_density_matrix(subset)` — the design doc §5 recipe with full comments:

  ```python
  # Move the kept axes to the front, then flatten to a matrix M whose rows are
  # indexed by the kept qubits' bits and columns by all the other qubits' bits.
  # M[i, j] is the amplitude of (kept-bits=i, rest-bits=j).
  # rho = M @ M.conj().T sums over the "rest" index — this is the partial
  # trace: we're averaging away everything we chose not to look at.
  ```

  Explain "density matrix" at first use in the docstring: the generalization of a state vector that can also describe *statistical mixtures*; its diagonal holds probabilities, its off-diagonal entries ("coherences") hold what remains of superposition.
- `entanglement_entropy(subset, base=2)` via SVD (design doc: numerically better than eigendecomposition, and exposes the Schmidt decomposition): `s = np.linalg.svd(M, compute_uv=False)`; probabilities `p = s**2`; entropy `-sum(p * log(p))` with `p > 1e-15` masking; comment what SVD gives here (the Schmidt coefficients: any bipartite pure state equals a single sum Σ sᵢ |aᵢ⟩|bᵢ⟩, and the sᵢ² are how "spread out" the entanglement is).
- `schmidt_spectrum(cut)`, `is_product(subset, tol)` (entropy < tol), `assert_zero(subset, tol)` (probability of any kept-qubit being 1 below tol — Phase 2's ancilla check reuses this), `norm()`.
- `bloch_vector(q)`: from the 1-qubit reduced ρ: `x = 2*rho[0,1].real`, `y = -2*rho[0,1].imag`, `z = (rho[0,0] - rho[1,1]).real`. Docstring introduces the Bloch sphere from scratch: every single-qubit state ↔ a point in/on a unit sphere; pure states on the surface, mixtures strictly inside, the maximally mixed state at the center. (Phase 2.5 leans on "the vector shrinks inward" heavily.)
- `expectation(pauli, reg=None)`: apply the Pauli letters to a copy of the tensor via `apply_1q`, then `np.vdot(flat_original, flat_transformed).real` — comment that `vdot` conjugates its first argument, so this computes ⟨ψ|P|ψ⟩, the average value you'd get measuring observable P many times.
- `mutual_information(a, b)` = S(A) + S(B) − S(AB).
- `fidelity(other)`: `abs(np.vdot(self_flat, other_flat))**2`.
- `ket(max_terms=8)` returns a `Ket` object: holds the top-|amplitude| terms; `__str__` gives e.g. `0.707|00⟩ + 0.707|11⟩`; `_repr_latex_` gives LaTeX (`$\frac{...}$` style, amplitudes formatted to 3 significant digits, exact common values like 1/√2 NOT special-cased — keep it simple, print decimals). Include `+ …` when terms were truncated.

Environment-qubit awareness (`system_density_matrix` etc.) is Phase 2.5 — do not build it now, but keep `reduced_density_matrix` general enough to take any subset (it already is).

## 6. `viz.py` + rich display (§10.1 hooks)

Lazy-import matplotlib inside each function. This phase builds:

- `viz.amplitudes(qc, *, phase_as_hue=True)` — bar chart over basis states: height |amplitude|, bar color = complex phase mapped through the `hsv` colormap (`np.angle(amp)` → [0,2π) → colormap). X tick labels are bitstrings (rotate 90° when n > 4; cap at 6-qubit default with a `top=` fallback like `probabilities`). Add a small phase-color legend (a colorbar labeled "phase of amplitude"). Docstring explains: two states can have identical bar heights (same measurement statistics *right now*) yet different colors — and the colors are what interference acts on.
- `viz.probabilities(qc, top=32)` — sorted bars of |amp|².
- `viz.bloch(qc, q)` — 3D sphere (matplotlib `plot_surface` wireframe, alpha≈0.1), axes labeled |0⟩ (north pole, +z), |1⟩ (south), |+⟩/|−⟩ on ±x; draw the Bloch vector as an arrow (`quiver`); show its length in the title (length 1 = pure, <1 = mixed).
- `Circuit._repr_html_()` — name, qubit count, and the top-8 basis states as an HTML table: bitstring, a horizontal bar whose width is |amp| and whose color encodes phase (inline CSS, hue = angle in degrees), and the numeric amplitude. No JS, no external assets.
- `Ket._repr_latex_` (in `inspector.py`, see above).

Keep plots minimal and consistent: no seaborn styles, default matplotlib with a figsize argument; every figure has an informative title and labeled axes.

## 7. `__init__.py`

Public surface: `Circuit`, `Register`, `Qubit` (type only), the gate instances re-exported from `qsim.gates`, `errors` symbols, `viz` as a submodule, `set_dtype`. `from qsim.gates import H, X, ...` must work, as must `from qsim import Circuit`.

## 8. Acceptance tests — `tests/test_acceptance_t1_t7.py`

Implement T1–T7 exactly as specified in design doc §9 (tolerances included). Additions/clarifications:

- T3 (GHZ): loop ≥50 seeds; after measuring qubit 0, assert `measure` on qubits 1 and 2 both return the same value as qubit 0. Use a fresh circuit per seed.
- T5: build the random circuit from a seeded rng choosing among {H, X, Y, Z, S, T, Rx, Ry, Rz, CNOT, CZ} with random qubits/angles; assert `qc.inspect.norm()` within 1e-12 of 1 after 200 gates on 8 qubits.
- T6: also assert the *error messages* mention "no-cloning" (substring check) — the teaching content is part of the spec.
- T7: with ancilla scopes not existing until Phase 2, test the liveness mechanism directly: manufacture a dead handle by setting the internal flag (white-box, one line, commented as a Phase-2 preview) and assert gates/measure raise `DeadQubitError`.

`conftest.py`: fixtures `seed` (param or constant 1234), `bell_pair` (returns a fresh 2-qubit circuit with H+CNOT applied), helper `random_state(n, seed)` building a normalized random complex tensor for kernel tests.

## 9. Notebooks

Both notebooks follow the master-plan pedagogy rules (≈60% markdown, concept-before-code, no QM assumed, numpy explained). Cell-level outlines:

**`01-states-and-gates.ipynb` — "A qubit is a unit vector"**
1. What you will learn. What a qubit *is*: not "0 and 1 at the same time" but a unit vector in C²; amplitudes; why complex numbers (promise: interference, shown in this notebook).
2. Dirac notation from scratch: |0⟩ = (1,0), |1⟩ = (0,1), |ψ⟩ = α|0⟩+β|1⟩; ⟨ψ| as conjugate transpose; why the notation earns its keep.
3. First circuit: `Circuit(1)`, apply `H`, show `qc.inspect.ket()` and `viz.amplitudes`. The Born rule: `inspect.probabilities()` vs `inspect.sample(1000)`.
4. Phases are invisible… : Z on |+⟩ — probabilities unchanged (bar heights same, colors differ — point at the hue).
5. …until they interfere: H·Z·H|0⟩ = |1⟩ vs H·H|0⟩ = |0⟩. Walk the amplitude arithmetic by hand in markdown (the 2-path sum with signs). **This is the punchline cell of the notebook.**
6. The Bloch sphere: introduce, `viz.bloch` for |0⟩, |+⟩, T|+⟩; gates are rotations of the sphere.
7. Measurement collapses: measure in a loop, show the state after via `ket()`.
8. Two qubits, tensor product: shape (2,2) tensor, the four basis states, SWAP = 3 CNOTs demo.
9. What you now know / what's next (entanglement).

**`02-entanglement.ipynb` — "States that aren't made of parts"**
1. Product states: H⊗H|00⟩, show the 4 equal bars; each qubit has its own Bloch vector of length 1.
2. The Bell state: H + CNOT; `ket()` shows (|00⟩+|11⟩)/√2. Try to factor it in markdown algebra — show it cannot be written as (a|0⟩+b|1⟩)⊗(c|0⟩+d|1⟩). *Define entanglement as exactly this failure.*
3. What does one half look like? `reduced_density_matrix([0])` — introduce density matrices here in prose; the answer is the maximally mixed state; `bloch_vector` has length 0 (plot it: an arrow of length zero at the center).
4. Quantifying it: entanglement entropy; 0 for product states, 1 bit for Bell; `schmidt_spectrum`.
5. Correlation: measure qubit 0, look at qubit 1's state after (both outcomes, reseeded). GHZ with 3 qubits.
6. `is_product`, entropy across cuts; a random circuit's entropy growth (tiny preview plot).
7. Why a Qubit has no `.state` attribute — show the API refusing (`try/except` around `copy.copy(q)`), quote the no-cloning message; connect to the design.
8. What you now know / next (notebook 03: entanglement is not classical correlation — the CHSH game).

## Definition of done

- `uv run pytest -v`: all Phase 0 + Phase 1 tests pass (T1–T7 among them), **100% coverage of `src/qsim/`** (master plan → Testing). That means, beyond the tests specced above: every `raise` site triggered and its message checked, every gate exercised (including SX, Fredkin, both branches of parametrized inverses), `viz` functions called under the `Agg` backend with basic structural asserts, `_repr_html_`/`_repr_latex_` outputs sanity-checked (contains expected substrings), and edge cases per the master plan list (empty register, 1-qubit circuit, re-measuring a measured qubit, θ=0/2π, `encode` on a non-zero register raising, `Circuit()` with n=0). Tests are named and laid out as documentation.
- `uv run pyright`: 0 errors. `uv run ruff check .`: clean.
- `uv run jupyter execute notebooks/01-*.ipynb notebooks/02-*.ipynb` succeeds.
- No 2^n×2^n matrix anywhere in `src/`; grep for `kron` must hit only tests.
- Module docstrings state their physical fact; error messages teach; numpy comments per master plan.
- Report lists any "Decisions made".

## Interface decisions — resolved

Presented to the owner as usage examples before any code was written, and settled as
follows. **Phase 1 shipped; this section is the record, not a to-do list.**

1. **Gate call style — keyword-only angles.** `H(a)`, `CNOT(a, b)`, `Rz(a, theta=np.pi/4)`.
   *Why:* every positional argument to a gate is a qubit, so a bare float in that position
   is the one thing a reader has to stop and decode — and `CPhase(a, b, theta=…)` would
   otherwise look like it takes a third qubit. Gates stay module-level callables that find
   their circuit from the handles; there is no `qc.h(a)` method form.
2. **`alloc_many` is the taught default**, with a new `qc.qubits -> Register` accessor added.
   *Why:* the one idea notebook 01 must land is that a `Qubit` is a handle the circuit hands
   you, and `alloc_many` shows that happening where `Circuit(2)` makes handles appear from
   nowhere. `Circuit(n)` still works — and needed `qc.qubits`, since this plan specified the
   pre-allocating constructor without saying how to reach what it allocated.
3. **`ket()` prints plain decimals**, three places, eight terms, largest magnitude first,
   complex amplitudes parenthesized, truncation announced as `… (N more terms)`.
   *Why:* recognizing exact surds works beautifully for the handful of textbook states and
   then silently stops the moment you apply a `T` gate — exactly when a learner most needs to
   trust the output. A format that always means the same thing beats one that is occasionally
   prettier.
4. **Visuals approved as rendered**: the vivid cyclic `hsv` wheel for phase-as-hue, in
   `viz.amplitudes`, in the Bloch plot, and in `Circuit._repr_html_`. The alternative
   considered was `twilight` (calmer, friendlier to red-green colour vision deficiency, but
   adjacent phases are harder to tell apart). The HTML repr uses `currentColor` and no
   background, so it reads correctly in both light and dark Jupyter themes.
5. **Bit order confirmed**: `reg[0]` is the most significant bit. A two-qubit register in
   |10⟩ measures as 2, not 1, and sits at flat index 2 of `state_vector()`.

Two additions the owner asked for while the phase was in flight:

6. **`inspect.bra()` and `inspect.overlap(other)`.** `bra()` is display-only and exists for
   one reason: printing it next to its ket makes conjugation *visible* (every `+0.354i`
   becomes `-0.354i`), which is what notebook 01 needs when it introduces ⟨ψ|. `overlap`
   returns the raw complex ⟨other|ψ⟩ — `fidelity` squares the phase away, and the phase is
   exactly what decides how two states add when they interfere.
7. **Every gate has a spelled-out name**, the same object under a longer alias:
   `Hadamard is H`, `PauliX`, `PauliY`, `PauliZ`, `SqrtX`, `Swap`, `ControlledNot`,
   `ControlledZ`, `ControlledControlledNot`, `ControlledSwap`, `RotationX/Y/Z`,
   `ControlledPhase`. `gate.name` stays the short symbol — it is what the history,
   `gate_counts()` and future diagrams use — and `gate.full_name` is the long one;
   `gate.label` combines them for errors and reprs. S and T have no settled name in the
   literature ("the phase gate" being already taken here by the parametrized `Phase`), so
   they are named for what they do: **`S = SqrtZ`** and **`T = FourthRootZ`**.

## Deviations from this plan (as built)

- **`Qubit.__slots__` stores `_id`, not `_axis`.** Design doc §2.2 lists `_axis` in the slots
  while §2.4 requires handles to hold stable ids; the id wins, per §2.4 and this plan's §2.
- **Acceptance tests pass qubit handles, not axis indices.** Design doc §9's T2 snippet writes
  `entanglement_entropy([0])`; the tests use `[a]`. Same physical content, and axis numbers
  never appear in user code.
- **No `n = 0 → 1` special case on allocation.** `zero_state(0)` returns a shape-`()` array
  holding the number 1 — honest (zero qubits span a 1-dimensional space) and it makes
  allocation a single uniform code path, since tensoring |0⟩ onto it yields the 1-qubit state.
- **`apply_controlled_diag` is a separate kernel** from `apply_controlled`, rather than a flag
  on one function.
- **`Op` gained a `result: int | None` field** so measurement outcomes are recorded honestly
  instead of being cast into the float `params` tuple.
- **Gate inversion is `gate.adjoint_op(op) -> Op` plus a `GATES` name registry**, rather than a
  bare `inverse` property — that is the shape Phase 2's `adjoint` actually needs. Private
  `S†`, `T†`, `SX†` gates exist for the three gates that are not their own inverse.
- **`inspect.sample()` draws from a second RNG** (`Circuit._sample_rng`), not the measurement
  stream. Sampling is a simulator cheat, not a physical measurement, so adding a `sample()`
  call to a seeded notebook must not silently rewrite every measurement below it.
- **`entanglement_entropy` adds `+ 0.0`** so a product state reports `0.0` rather than `-0.0`
  (`-1 × log(1)` is negative zero, which would print as "-0.000 bits").
- **`viz.amplitudes` refuses above 6 qubits**, pointing at `viz.probabilities(qc, top=…)`;
  128 bars is not a readable chart.
- **`circuit.py` imports `measure`, `gates` and `inspector` inside methods.** The dependency
  genuinely runs that way — those modules build on the types defined in `circuit.py` — so a
  top-level import is circular.
- **`tests/test_viz.py` was added**, absent from this plan's file list but required by the
  100%-coverage rule.
- **A note for future test authors:** consecutive seeds produce correlated PCG64 streams.
  `Circuit(seed=s) for s in range(400)` gave a 2.4σ-biased coin. Statistical tests draw their
  seeds from a master generator instead.
