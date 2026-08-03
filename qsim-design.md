# `qsim` — Design Document

A NumPy state-vector quantum computer simulator, optimized for **conceptual transparency**, not speed.

---

## 0. Purpose and Non-Goals

**Purpose.** A library for learning quantum mechanics and quantum computing by building and inspecting circuits. Every design decision should favor the option that makes a physical fact visible, checkable, or breakable.

**Deliverable.** Two-headed: an importable library for writing and running small quantum programs, and an interactive learning environment — Jupyter notebooks built on that library (§0.5, §10). Neither is secondary. The API must read well in a plain script *and* render well in a notebook.

**Orientation.** The interesting direction is watching the classical world grow out of the quantum one — decoherence, einselection, redundant environmental records — not bolting quantum effects onto a classical picture. The former is what actually happens. This is why decoherence sits in the middle of the phase plan rather than at the end (Phase 2.5), why the CHSH test is a first-class demo (§8.6), and why Phase 7 is framed as the emergence of the classical (§11).

**Target scale.** 10–20 qubits. Everything runs in well under a second. Memory is never a consideration.

**Explicit non-goals:**

- Performance. No GPU, no gate fusion, no in-place buffer ping-ponging, no `complex64` by default, no multithreading.
- Hardware backends, OpenQASM export, transpilation to native gate sets.
- Large-scale simulation. If a user wants 30 qubits, they should use Qulacs.
- Tensor-network / stabilizer methods (see §11 for why stabilizer sim is a possible later addition).

**Guiding principle.** When a shortcut would produce the right answer while hiding the mechanism, take the long road. The canonical case: modular exponentiation in Shor's algorithm must be compiled from reversible adders, never applied as a precomputed permutation matrix (§8.3).

---

## 0.5 Tooling and Project Scaffolding

The project is managed with **uv** on **Python 3.14**.

```toml
[project]
name = "qsim"
requires-python = ">=3.14"
dependencies = [
    "numpy>=2.3",        # 2.3 is the floor for Python 3.14 support
    "matplotlib>=3.10",
]

[dependency-groups]
dev = [
    "pytest>=8",
    "ruff",
    "jupyterlab",
    "ipympl",            # interactive matplotlib in notebooks
    "ipywidgets",
]
```

Workflow: `uv sync` to set up, `uv run pytest -v` to test, `uv run jupyter lab` to open the notebooks.

- Tests use pytest with fixtures (e.g. a seeded `circuit` fixture) rather than setup/teardown; test files are named `test_*.py`.
- `matplotlib` is a first-class runtime dependency. This is a learning library and visualization is half the point (§10). It is still imported lazily inside `viz.py` and the display hooks, so the core (`state`, `circuit`, `gates`) imports cleanly in a bare interpreter.
- Notebooks are executable documentation: `uv run jupyter execute notebooks/*.ipynb` must succeed, and this check is part of every phase's definition of done (§11).

---

## 1. Core Representation

### 1.1 State tensor

The state of `n` qubits is a single `numpy.ndarray` of shape `(2,) * n` and dtype `complex128`.

**This is not a flattened vector reshaped for convenience.** The axes *are* the tensor factors of the Hilbert space. Axis `k` corresponds to qubit `k`. This identification is the central pedagogical object of the library and should be stated in the module docstring of `state.py`.

Indexing convention: `psi[b_0, b_1, ..., b_{n-1}]` is the amplitude of the computational basis state $|b_0 b_1 \ldots b_{n-1}\rangle$, with **qubit 0 as the most significant bit** when converting to an integer. This must be documented prominently and used consistently; bit-ordering bugs are the single most common source of confusion in quantum simulators.

### 1.2 Gate application

A single-qubit gate `U` (shape `(2,2)`) applied to qubit `k`:

```python
psi = np.tensordot(U, psi, axes=([1], [k]))
psi = np.moveaxis(psi, 0, k)
```

A two-qubit gate `U` (shape `(2,2,2,2)`, indexed `[out_j, out_k, in_j, in_k]`) applied to qubits `j, k`:

```python
psi = np.tensordot(U, psi, axes=([2, 3], [j, k]))
psi = np.moveaxis(psi, [0, 1], [j, k])
```

**Do not construct $2^n \times 2^n$ matrices anywhere in the library**, including in tests. If a test needs a full matrix to compare against, build it from `np.kron` in the test file itself, and only for `n <= 4`.

### 1.3 Controlled gates via slicing

A controlled-`U` with control `c` and target `t` should be implemented by slicing the control axis and applying `U` only to the `|1⟩` subspace:

```python
sl = [slice(None)] * n
sl[c] = 1
sub = psi[tuple(sl)]              # view, one axis dropped
# apply U to sub on the (adjusted) target axis, write back
```

This is not primarily an optimization. It makes the decomposition

$$CU = |0\rangle\langle 0| \otimes I + |1\rangle\langle 1| \otimes U$$

operationally concrete. Include this identity in the docstring.

### 1.4 Diagonal gates

Gates that are diagonal in the computational basis (`Z`, `S`, `T`, `Rz`, `Phase`, `CZ`, controlled-phase) should be applied as elementwise multiplication by a broadcast phase array, not via `tensordot`. Cheaper, but more importantly it makes "diagonal = does nothing to populations, only to phases" a structural fact of the code.

### 1.5 Precision

Default `complex128`. `complex64` selectable via `qsim.set_dtype()` — not for memory, but so users can run the float32-vs-float64 comparison in §9, test 17.

Design notes to record in `state.py`:

- Floating point provides *relative* precision per component, so small amplitudes are not degraded by being small. Underflow is irrelevant: `complex64` holds normals to $2^{-126}$.
- Errors accumulate additively, not multiplicatively, because every operation is unitary ($\kappa = 1$). Over $G$ gates, the 2-norm error is $O(\sqrt{G}\,\varepsilon)$.
- The circuit decomposition never forms a long sum; the QFT's implicit $2^n$-term sum is realized as a depth-$n$ binary tree, giving the pairwise-summation error constant $n\varepsilon$ rather than $2^n\varepsilon$.

---

## 2. Object Model

### 2.1 `Circuit`

Owns the state tensor, the qubit allocation table, and the recorded gate history.

```python
class Circuit:
    def __init__(self, n: int = 0, *, name: str = "", dtype=np.complex128,
                 seed: int | None = None): ...

    def alloc(self) -> Qubit: ...
    def alloc_many(self, count: int) -> tuple[Qubit, ...]: ...
    def register(self, size: int, *, name: str = "") -> Register: ...
    def ancilla(self, count: int = 1) -> AncillaContext: ...   # §4.3

    # Physical operations
    def measure(self, q: Qubit) -> int: ...
    def measure_all(self, reg: Register) -> int: ...
    def reset(self, q: Qubit) -> None: ...

    # Recording — the history is a tape (§4.6)
    @property
    def history(self) -> list[Op]: ...
    def gate_counts(self) -> dict[str, int]: ...
    def depth(self) -> int: ...
    def checkpoint(self) -> Checkpoint: ...
    def rewind(self, mark: Checkpoint) -> None: ...
    def on_op(self, fn: Callable[[Op, Circuit], None]) -> HookHandle: ...

    # Namespaced introspection — see §5
    @property
    def inspect(self) -> Inspector: ...
```

No union return types: `alloc()` returns one `Qubit`, `alloc_many(n)` returns a tuple. A `Qubit | tuple[Qubit, ...]` return would conflate two responsibilities and force a cast or an unverifiable unpack at every call site — the §12 goal of static checkability rules it out.

### 2.2 `Qubit`

A **handle to an axis**, not a value. This is the most important idea in the object model and the docstring must say so directly:

> A `Qubit` does not have a state. There is one joint state tensor owned by the `Circuit`; a `Qubit` names one of its axes. Once entangled, an individual qubit has no pure state to report — asking for one is a category error, not a missing feature. Use `circuit.inspect.reduced_density_matrix()` if you want the mixed state of a subsystem.

```python
class Qubit:
    __slots__ = ("_circuit", "_axis", "_name", "_live")

    def __copy__(self): raise NoCloningError(...)
    def __deepcopy__(self, memo): raise NoCloningError(...)
    def __eq__(self, other): return self is other      # identity, not value
    def __hash__(self): return id(self)
```

There is deliberately **no `Qubit.state` / `.value` / `.amplitude` property**, and no `__bool__`. There *is* a read-only `Qubit.circuit` / `Register.circuit` property (added Phase 3): a handle knows which circuit it names an axis of — that is bookkeeping, not physics — and blocks need it to open `qc.control`/`qc.ancilla` scopes without a redundant circuit parameter.

### 2.3 `Register`

An ordered sequence of qubits. First-class, because Shor's algorithm is entirely register arithmetic and index-juggling would make it unreadable.

```python
class Register(Sequence[Qubit]):
    def __getitem__(self, i: int | slice) -> Qubit | Register: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Qubit]: ...
    def reversed(self) -> Register: ...
    def concat(self, other: Register) -> Register: ...

    # Integer view — simulator-only convenience, must be flagged as such
    def encode(self, value: int) -> None:   # X gates to set |value>, only valid from |0..0>
```

Slicing returns a `Register` view over the same underlying qubits — no copying, no reallocation.

### 2.4 Axis lifecycle

The hardest bookkeeping problem in the implementation, so it is specified here rather than left to chance.

`alloc` mid-circuit grows the state tensor (tensor product with a fresh $|0\rangle$ axis). Deallocating an ancilla *removes* an axis — which shifts the positional index of every axis after it, and would silently invalidate every live handle if handles stored raw axis numbers.

Therefore: **a `Qubit` handle stores a stable id, never an axis number.** The `Circuit` owns a single id → axis table, updated on every allocation and deallocation; every gate-application kernel resolves ids to axes through it at call time. The §1 story "axis $k$ is qubit $k$" survives intact, one indirection away.

Framed differently, the `Circuit` **is** the qubit pool: it owns the allocation table, hands out handles (`alloc`), and reclaims them (ancilla scope exit). The `ancilla` context manager (§4.3) is the syntax sugar over pool acquire/release — with the crucial physical twist that release must *verify* disentanglement, not merely mark the slot free. A separate `Pool` object (`pool.qubit()`) was considered and rejected: it would duplicate the Circuit's ownership role without adding a capability, and two owners for one state tensor is exactly the ambiguity the object model exists to prevent. If a pool-flavored surface is ever wanted, it should be a thin facade over these same `Circuit` methods, not a second owner.

---

## 3. Gates

Gates are **functions that mutate the circuit state and return `None`.**

```python
from qsim.gates import H, X, Y, Z, S, T, SX, Rx, Ry, Rz, Phase, CNOT, CZ, CPhase, SWAP, Toffoli, Fredkin

H(a)
CNOT(a, b)
Rz(a, theta=np.pi/4)
Toffoli(a, b, target)
```

Each gate resolves its `Circuit` from the qubit handles it is given, and raises if handles from different circuits are mixed.

### 3.1 Validation the gates must perform

- **Distinctness.** `CNOT(a, a)` raises `NoCloningError` with a message explaining that a qubit cannot control an operation on itself, and that this is the no-cloning theorem showing up at the API surface.
- **Liveness.** Using a handle for a qubit that has been deallocated (ancilla scope exited) raises.
- **Cross-circuit.** Mixing handles from two `Circuit` objects raises.

### 3.2 Custom gates

```python
@qsim.gate
def my_block(x: Register, y: Register) -> None:
    ...
```

The decorator registers the block so it can be controlled, inverted, counted, and diagrammed as a unit. Decorated blocks must be composed only of other gates and blocks — no direct state manipulation.

---

## 4. Combinators

### 4.1 Execution model

Eager by default. A gate applied is a gate executed; the state tensor is always current and inspectable. The gate history is recorded alongside for diagramming, counting, and replay.

Deferred evaluation exists **only inside combinator scopes**, where it is forced by physics: you cannot lift a block to its controlled form without knowing the block first.

### 4.2 `control` and `adjoint`

```python
with qsim.control(c):
    add(x, y)                 # every gate in the body is lifted to its controlled form

with qsim.control(c1, c2):    # multiply controlled
    X(t)

with qsim.adjoint():
    add(x, y)                 # body recorded, then replayed reversed and daggered
```

Equivalently, for reusable blocks:

```python
add.controlled(c)(x, y)
add.adjoint()(x, y)
```

Implementation: within a combinator scope, the circuit enters *record mode* — gates append to a buffer instead of executing. On scope exit, the buffer is transformed (controlled / reversed+conjugated) and then executed. Scopes nest.

Every gate must declare how it is controlled and inverted. For a gate with no natural controlled form, fall back to decomposition into the universal set.

This declaration extends to classical parameters and to decorated blocks, which is where it is easy to get wrong. The recorder captures classical arguments (`theta`) alongside qubit handles, and parametrized gates declare their inverse in terms of them — $R_z(\theta)^{-1} = R_z(-\theta)$, not a numerically daggered matrix. A decorated block's adjoint is its recorded body replayed in reverse with each entry inverted; its classical arguments are captured at record time, so the same block object can be inverted or controlled long after the call site.

### 4.3 Ancillas and uncomputation

```python
with qc.ancilla(3) as anc:
    ripple_add(x, y, anc)
    # ...
    ripple_add_inv(x, y, anc)
# on __exit__: assert anc is in |000> and unentangled, then deallocate
```

**`__exit__` must numerically verify that the ancillas are back to $|0\rangle$ and unentangled from the rest of the state**, and raise `DirtyAncillaError` otherwise. The error message should explain that leftover garbage stays entangled with the answer register and destroys the interference the algorithm depends on.

This assertion is a simulator superpower — real hardware cannot check it — and it is the mechanism by which the library teaches Bennett's uncomputation trick. It is a hard requirement, not a debug option.

Note in the docstring: dropping a qubit handle does **not** free the qubit. Discarding an entangled qubit is a partial trace, silently converting the pure state to a mixed one. There is no garbage collection for quantum memory; release requires uncomputation. This is physics, not a missing feature.

### 4.4 Environment qubits and decoherence

Mixed states and decoherence do **not** require a density-matrix backend. Every noise channel can be realized as unitary coupling to an environment followed by a partial trace (Stinespring dilation — the "Church of the Larger Hilbert Space"). On a pure-state simulator this costs a few extra qubits and no new machinery.

```python
env = qc.environment(2, name="E")     # persistent allocation, marked as environment
dephasing_coupling(q, env[0], theta=np.pi/3)
qc.inspect.bloch_vector(q)            # shrunken — q is now mixed
```

**Critical design point: `environment()` does not trace anything out.** The environment qubits stay in the state tensor, entangled, forever. The global state remains pure. All that the marking does is tell `inspect` which qubits constitute "the system," so that diagnostics report the reduced state by default.

This is exactly the physics, and the API should make it unmissable: decoherence is not something that happens *to* a state. It is what a subsystem looks like when you decline to track the rest of the world. Put that sentence in the module docstring.

```python
qc.environment(count, *, name="") -> Register
qc.environment_qubit(*, name="E") -> Qubit           # single-qubit convenience (added Phase 3)
qc.inspect.system_density_matrix() -> np.ndarray    # traces out all environment qubits
qc.inspect.system_entropy() -> float
qc.inspect.coherence(q) -> float                    # |rho_01| of the reduced 1-qubit state
```

#### Couplings

Each coupling in `decoherence.py` is a small unitary block on (system, environment) qubits that realizes a named Kraus channel when the environment is traced out. **Each docstring must state the Kraus operators it corresponds to**, so the dilation ↔ Kraus map is explicit rather than asserted.

```python
dephasing_coupling(q, env: Qubit, theta) -> None
amplitude_damping_coupling(q, env: Qubit, theta) -> None
depolarizing_coupling(q, env: Register, p) -> None      # needs 2 environment qubits
pointer_coupling(q, env: Qubit, theta, basis="z") -> None
```

- **Dephasing.** Controlled-$R_y(\theta)$ on the environment, conditioned on the system being $|1\rangle$: $\alpha|0\rangle|0_E\rangle + \beta|1\rangle|0_E\rangle \mapsto \alpha|0\rangle|0_E\rangle + \beta|1\rangle(\cos\tfrac\theta2|0_E\rangle + \sin\tfrac\theta2|1_E\rangle)$. Tracing out the environment leaves populations untouched and multiplies the off-diagonal by $\cos\tfrac\theta2$. At $\theta = \pi$ the environment has perfectly recorded which-path information and coherence is gone.
- **Amplitude damping.** A rotation by $\theta$ within the $\{|1\rangle|0_E\rangle, |0\rangle|1_E\rangle\}$ subspace — excitation transfers to the environment.
- **Depolarizing.** Two environment qubits prepared in a $p$-weighted superposition, controlling application of $I, X, Y, Z$.
- **`pointer_coupling`** takes the system operator through which the environment couples. This is the knob for einselection: coupling through $\sigma_z$ makes the computational basis the pointer basis; coupling through $\sigma_x$ makes $\{|+\rangle,|-\rangle\}$ the pointer basis. Which states survive is determined by the interaction, not by anything intrinsic to the state.

#### The eraser

Because the environment is still there and still unitary, **the coupling can be uncomputed** and coherence returns exactly. This must be a first-class demo, not a footnote.

It is also the conceptual bridge to §8.3 and test T18: a dirty ancilla *is* an environment, and uncomputation *is* erasure. Decoherence and failed uncomputation are one phenomenon, and a student who meets them in separate phases will not notice. Cross-reference in both directions.

### 4.5 Conjugation: `within` and the closed block algebra

The sandwich $V\,U\,V^\dagger$ — do a basis change, act, undo the basis change — is the single most common composite in quantum programs: `pointer_coupling` is dephasing conjugated by H, the Grover oracle is a controlled-Z conjugated by X's, and the Fourier-space adder (§8.3) is phase rotations conjugated by the QFT. It gets a first-class combinator:

```python
with qsim.within(H, q):                    # V = H(q), applied now
    dephasing_coupling(q, e, theta=theta)  # body runs EAGERLY — state inspectable
                                           # V† emitted here on exit
```

Semantics, chosen to preserve the eager execution model:

- `within(V, *args, **kwargs)` applies `V` immediately, *capturing* its ops as it does. The body is **not** recorded — it runs eagerly and the state stays watchable between its statements. On scope exit, the captured ops replay reversed and daggered. (Contrast with `control`, which must record its body: "run conditioned on c" is a counterfactual that has to execute *differently*, while "undo V later" only needs V remembered. This is the same asymmetry as PyTorch's tape: `backward()` is a tape operation, `vmap` is a function transform.)
- `V` is any op-emitting callable — a gate, a `@qsim.gate` block, or a plain function; its qubit arguments identify the circuit. `V` must contain no measurement (nothing irreversible can be undone on exit). When V has a name (a block or gate), the emitted forward ops keep its stamp and the undo half is stamped `name†`, so `block_counts()` and diagrams show the conjugation symmetrically with `block.adjoint()`; anonymous callables stay unstamped (added Phase 3).
- If the body raises, `V†` is **not** applied — consistent with the other scopes: never run half a construct on the way out of an error.
- Conjugation *as a reusable block* is spelled with the existing abstraction mechanism, a `def`:

  ```python
  @qsim.gate
  def x_dephasing(q, e, theta):
      with qsim.within(H, q):
          dephasing_coupling(q, e, theta=theta)
  ```

Two algebraic facts, both docstring-worthy and both tested (TT2, TT3): the adjoint of a conjugation inverts only the middle, $(VUV^\dagger)^\dagger = V\,U^\dagger\,V^\dagger$; and control distributes over products, so lifting every op of the sandwich is correct — though $C(VUV^\dagger) = V\,(CU)\,V^\dagger$ whenever V avoids the control, meaning the basis change never *needs* the control. Implement the uniform (correct) lifting; the optimization is optional and, if added, must be state-equivalent (TT3 checks both forms agree).

**Closed algebra requirement.** Individual gates already satisfy it (`T.adjoint().controlled()` returns a gate). Blocks must too: `Block.adjoint()` and `Block.controlled(...)` return `Block`s — named (`bell†`, `C-bell`), chainable, countable, diagrammable — never bare callables. An operation on a block is a block.

### 4.6 The tape: checkpoints, rewind, and hooks

The recorded history *is* an autograd tape, and the correspondence with PyTorch's define-by-run model is exact enough to be a design guide: eager ops with a graph recorded as a side effect; per-op inverse rules in place of per-op gradient rules; `adjoint` as the reverse-mode tape walk; `@qsim.gate` tracing as `fx.trace`. Two tape features follow, with one deep difference: autograd's backward pass needs saved activations, but ours needs **nothing saved** — unitaries destroy no information, so the tape alone suffices. Reversibility is the physics; "no activation cache" is its software shadow.

```python
mark = qc.checkpoint()          # a position on the tape (plus an allocation fingerprint)
H(q); CNOT(q, e)
qc.rewind(mark)                 # execute the suffix's inverses, newest first
qc.on_op(fn) -> HookHandle      # fn(op, qc) after every executed op; .remove() to detach
```

- **`rewind` keeps the tape honest.** The inverse gates physically run and are appended to the history; the *state* returns exactly to the checkpoint, the *record* shows how (like an editor's undo appearing in the edit log). The tape is a record of what happened — it is never rewritten. For the same reason there is no fx-style graph mutation anywhere in the library.
- **`rewind` raises** if the suffix contains a measurement (the one op with no inverse rule — it severs the tape exactly as a non-differentiable op severs an autograd graph; the error message must draw that line and point at the eraser: keep the record coherent and you may rewind through it), or if qubits were allocated or released since the mark, or inside a combinator scope.
- **Hooks** fire for every executed op, measurements included — which requires the one-funnel fix: `_execute` must route its history append through `_record`, where hooks live. Hooks observe (inspect, collect, plot) but must not emit gates; emitting inside a hook raises. `viz.entropy_trace` (§10) is respecified as a hook client: replay the tape with an entropy hook attached.

---

## 5. Introspection (`circuit.inspect`)

All operations that are impossible on real hardware live behind the `inspect` namespace. The namespace boundary is itself pedagogy: everything inside it is cheating.

The module is `inspector.py` and the class is `Inspector` — never `inspect.py`, which would shadow the stdlib `inspect` module and be a permanent footgun for tooling and contributors. The accessor stays `qc.inspect` (an attribute name cannot shadow a module).

```python
qc.inspect.state_vector() -> np.ndarray           # flat, length 2**n
qc.inspect.state_tensor() -> np.ndarray           # shape (2,)*n
qc.inspect.amplitude("0101") -> complex
qc.inspect.probabilities() -> np.ndarray          # non-collapsing
qc.inspect.sample(shots=1000) -> Counter          # non-collapsing sampling

qc.inspect.reduced_density_matrix(subset) -> np.ndarray
qc.inspect.entanglement_entropy(subset, base=2) -> float
qc.inspect.schmidt_spectrum(cut) -> np.ndarray
qc.inspect.is_product(subset, tol=1e-10) -> bool
qc.inspect.assert_zero(subset, tol=1e-10) -> None
qc.inspect.norm() -> float
qc.inspect.bloch_vector(q) -> tuple[float, float, float]

qc.inspect.marginal(subset) -> np.ndarray             # P over just those qubits, MSB-first in given order
qc.inspect.expectation(pauli: str, reg: Register | None = None) -> float  # "ZZ", "XY", ...
qc.inspect.mutual_information(a, b) -> float          # S(A) + S(B) - S(AB)
qc.inspect.fidelity(other: np.ndarray) -> float
qc.inspect.ket(max_terms: int = 8) -> Ket             # Dirac notation, renders as LaTeX (§10.1)
```

`marginal` exists because `probabilities()[0]` *reads* like P(q₀=0) but is P(|00…0⟩) — a trap that bit three demo notebooks before the helper was added (Phase 3). `expectation` takes a Pauli string over a register (identity elsewhere) — the bread and butter of actual quantum mechanics, and required by the CHSH test (TB1). `mutual_information` is what the quantum-Darwinism demo (§11, Phase 7) reads out. `fidelity` exists mostly to keep acceptance tests (T9, T13) honest and readable.

**Partial trace implementation:** reshape the state tensor into `(2**k, 2**(n-k))` after moving the kept axes to the front, then `rho = M @ M.conj().T`. Entanglement entropy from the eigenvalues, or equivalently from the squared Schmidt coefficients (`np.linalg.svd` of `M`) — the SVD route is numerically better and makes the Schmidt decomposition explicit.

---

## 6. Measurement

```python
qc.measure(q) -> int          # Born rule, collapses, renormalizes
qc.measure_all(reg) -> int    # sequential; integer with reg[0] as MSB; per-qubit outcomes go in the history
qc.reset(q) -> None           # measure, then X if outcome was 1
```

Measurement of qubit `k`: compute $p_1 = \sum |\psi|^2$ over the `k=1` slice, sample, zero the other slice, renormalize.

The collapse must be implemented as an actual projection and renormalization of the joint state — not as sampling from a marginal — so that conditional states on the remaining qubits are correct. This is the mechanism by which the Born rule stops being a formula, and it is required for the semiclassical QFT (§8.2).

RNG: `Circuit(seed=...)` for reproducible tests, backed by `np.random.default_rng`.

---

## 7. Package Layout

```
qsim/
  __init__.py
  state.py            # state tensor, gate application kernels
  circuit.py          # Circuit, Qubit, Register, allocation
  gates.py            # gate definitions, universal set, decompositions
  combinators.py      # control, adjoint, ancilla, record mode
  inspector.py        # Inspector: tomography-style introspection (NOT inspect.py — stdlib shadow)
  measure.py
  decoherence.py      # environment couplings realizing noise channels by dilation
  errors.py           # NoCloningError, DirtyAncillaError, DeadQubitError
  algorithms/
    qft.py            # qft, iqft, approximate_qft, semiclassical_qft
    phase_estimation.py
    arithmetic.py     # Cuccaro adder, modular adder/multiplier, modexp
    shor.py
    grover.py
    deutsch_jozsa.py
    teleportation.py  # teleportation + superdense coding
    chsh.py           # Bell-inequality violation
  viz.py              # amplitude plots, Bloch sphere, widgets, circuit diagram
notebooks/
  01-states-and-gates.ipynb          # Phase 1
  02-entanglement.ipynb              # Phase 1: Bell, GHZ, entropy, Schmidt
  03-bell-tests-teleportation.ipynb  # Phase 1.5: CHSH, teleportation, superdense coding
  04-combinators.ipynb               # Phase 2: control/adjoint/ancilla, DirtyAncillaError live
  05-decoherence.ipynb               # Phase 2.5: visibility curves, eraser, einselection
  06-qft-phase-estimation.ipynb      # Phase 3: watching the comb form
  07-shor.ipynb                      # Phase 5: full trace, T18 interactively
  08-grover-deutsch-jozsa.ipynb      # Phase 6
tests/
```

The notebooks are not documentation *about* the library; they are the interactive learning environment the library exists to serve. Each phase ships its notebook (§11), and the notebooks must execute top-to-bottom (§0.5).

---

## 8. Algorithms

### 8.1 QFT

$$\text{QFT}\,|j\rangle = \frac{1}{\sqrt{2^n}}\sum_{k=0}^{2^n-1} e^{2\pi i jk/2^n}\,|k\rangle$$

Standard circuit: for each qubit, `H` followed by controlled-phase rotations $R_m = \mathrm{diag}(1, e^{2\pi i/2^m})$ from each subsequent qubit. **The circuit produces output in bit-reversed order**; the final SWAP network is required to correct this. Make the SWAP network a keyword argument (`swap=True`) so users can see the reversal for themselves.

```python
qft(reg, *, swap=True, approx=None)   # approx=m drops rotations R_k for k > m
iqft(reg, *, swap=True, approx=None)
```

Document the two precision facts:
- Phase-register size needed for Shor's: $t = 2n + 1 + \lceil \log_2(2 + 1/2\varepsilon)\rceil$, because distinct $s/r$ with $r < N$ differ by at least $1/N^2$.
- Rotation depth needed: only $m = O(\log(t/\varepsilon))$ distinct angles. Error of the approximate QFT scales as $t^2 2^{-m}$.

### 8.2 Phase estimation

```python
phase_estimation(unitary_block, eigenstate: Register, out: Register) -> None
semiclassical_phase_estimation(unitary_block, eigenstate: Register, t: int) -> int
```

The semiclassical version measures the phase register one qubit at a time with classical feedback (Griffiths–Niu), using $O(1)$ extra qubits instead of $t$. Both must be implemented, and a test must confirm they agree — this demonstrates the deferred measurement principle empirically.

### 8.3 Arithmetic — the honest requirement

This section is the heart of the project.

```python
cuccaro_adder(a: Register, b: Register, carry: Qubit) -> None      # b += a
modular_adder(a: Register, b: Register, N: int, anc) -> None
controlled_modular_multiplier(c: Qubit, x: Register, out: Register, a: int, N: int, anc) -> None
modexp(a: int, x: Register, out: Register, N: int, anc) -> None    # |x>|1> -> |x>|a^x mod N>
```

**Hard constraint: `modexp` must be built from Toffoli-based reversible adders. It must not be implemented as a precomputed permutation matrix, a lookup table, or any construction that requires knowing the answer.**

Most published "Shor's in 30 lines" demonstrations — and several experimental papers — quietly compile the modular exponentiation using foreknowledge of the factors. Such an implementation runs and produces correct output while containing none of the algorithm's content. There should be a comment at the top of `arithmetic.py` stating this constraint and why it exists.

Follow Beauregard's construction (`arXiv:quant-ph/0205095`): $2n + 3$ qubits for an $n$-bit $N$.

### 8.4 Shor

```python
shor(N: int, *, a: int | None = None, seed=None, semiclassical=False) -> ShorResult
```

`ShorResult` should expose the full trace: chosen `a`, measured phase register value, continued-fraction convergents, candidate period, verification, factors. Period-finding is probabilistic — return the failure reason rather than looping silently.

Expected sizes: $N=15 \to 11$ qubits, $N=21 \to 13$, $N=35 \to 15$. With `semiclassical=True`, roughly $n+2$.

### 8.5 Grover, Deutsch-Jozsa

Standard. Grover included mainly as an amplitude-amplification test with an exactly predictable success probability: after $k$ iterations, $P = \sin^2((2k+1)\theta)$ with $\sin\theta = 1/\sqrt{N}$.

### 8.6 Entanglement demos — CHSH, teleportation, superdense coding

Cheap (2–3 qubits, Phase 1 machinery only: mid-circuit measurement with classical feedback, `inspect.expectation`) and pedagogically dense.

The CHSH test is the most important single demonstration in the library for a student of quantum mechanics: the empirical proof that entanglement is not classical correlation — no local hidden-variable account survives $S = 2\sqrt2 > 2$. Compute $S$ both ways: analytically via `inspect.expectation`, and empirically by sampled measurement. The notebook should also sweep every deterministic local strategy to show the classical bound of 2 being *computed*, not asserted.

This section carries the library's orientation (§0): CHSH shows the classical picture *failing*; decoherence (§4.4), einselection (TD5), and the Darwinism demo (Phase 7) show where the classical picture *comes from*. Cross-reference in both directions.

Teleportation doubles as the demo that quantum information is moved, not copied — after the protocol the source qubit sits in a computational basis state, and the docstring should point at `NoCloningError` (§3.1).

---

## 9. Acceptance Tests

These double as the specification. Each should be a readable, self-contained example.

**T1 — Superposition.** `H` on $|0\rangle$ gives amplitudes $(1/\sqrt2, 1/\sqrt2)$ to 1e-15. `H` twice is identity.

**T2 — Bell state.** 
```python
qc = Circuit(2); a, b = qc.alloc_many(2)
H(a); CNOT(a, b)
assert np.allclose(qc.inspect.probabilities(), [0.5, 0, 0, 0.5])
assert abs(qc.inspect.entanglement_entropy([0]) - 1.0) < 1e-12
assert not qc.inspect.is_product([0])
```
Entropy exactly 1 bit is the acceptance criterion for the partial-trace implementation.

**T3 — GHZ correlations.** 3-qubit GHZ; measuring qubit 0 forces the other two to the same value, over many seeds.

**T4 — Product state has zero entropy.** Any single-qubit-gates-only circuit gives entropy < 1e-12 across every cut.

**T5 — Unitarity.** 200 random gates on 8 qubits; norm stays within 1e-12 of 1.

**T6 — No-cloning guards.** `copy.copy(q)` raises. `CNOT(a, a)` raises. `Toffoli(a, b, a)` raises. Handles from different circuits raise.

**T7 — Dead handle.** A qubit handle used after its ancilla scope exits raises `DeadQubitError`.

**T8 — Ancilla cleanup enforced.** A block that allocates ancillas, entangles them, and exits without uncomputing raises `DirtyAncillaError`. The matching block *with* uncomputation exits cleanly.

**T9 — Adjoint is inverse.** For a random 6-qubit block `U`: applying `U` then `with adjoint(): U` returns the state to within 1e-13 of the original.

**T10 — Control combinator correctness.** `with control(c): X(t)` produces exactly the same state as `CNOT(c, t)` from a random input state. Same for a multi-gate block against an independently constructed controlled matrix (`np.kron`, n ≤ 4).

---

*Entanglement-demo tests (Phase 1.5). Lettered separately, like the TD group, to keep phase order without renumbering.*

**TB1 — CHSH violation.** With the optimal measurement angles, `inspect.expectation` gives $S = 2\sqrt2$ to 1e-12; sampling $10^5$ shots per setting gives $S > 2.7$. The test also sweeps all deterministic local strategies and asserts their maximum is exactly 2.

**TB2 — Teleportation.** A random single-qubit state teleported through a Bell pair arrives with fidelity 1 to 1e-12, checked across seeds that exercise all four measurement branches. The source qubit ends in a computational basis state — moved, not copied.

**TB3 — Superdense coding.** All four two-bit messages transmitted through one qubit decode correctly with probability 1.

---

*Tape and transform tests (Phase 2.75). Lettered separately, like the TB/TD groups.*

**TT1 — `within` equals the hand-built sandwich.** `within(H, q)` around a dephasing coupling produces a state identical (1e-13) to applying H, coupling, H by hand — and identical to `pointer_coupling(basis="x")`, which must itself be reimplemented via `within`.

**TT2 — Adjoint inverts only the middle.** For a block `V U V†` built with `within`: its `.adjoint()`'s recorded op sequence is V, U†, V† (structural assert on op names/order), and running block-then-adjoint returns a random state to fidelity 1 within 1e-13.

**TT3 — Control distributes over the sandwich.** The controlled version of a `within` block matches an independently `np.kron`-built controlled matrix (n ≤ 4), and matches the optimized form V·(CU)·V† with the basis change uncontrolled.

**TT4 — Rewind is exact and honest.** After a random 20-gate block on 5 qubits, `rewind(mark)` restores the state to fidelity 1 within 1e-13, and the history *grows* by the 20 inverse ops — the tape records the undoing rather than pretending nothing happened.

**TT5 — Measurement severs the tape.** `rewind` across a recorded measurement raises `QsimError`; the message must state that measurement is the one operation with no inverse rule, and point to the eraser (TD3) as the coherent alternative.

**TT6 — Allocation pins the tape.** `rewind` across an `alloc` or an ancilla-scope release raises: the axes the suffix's ops refer to must still exist and mean the same thing.

**TT7 — Hooks see everything and touch nothing.** An `on_op` hook fires for every gate *and* every measurement (count them); a hook computing entanglement entropy live reproduces the entropy trace; after `handle.remove()` no further calls arrive; a hook that tries to apply a gate raises.

**TT8 — The block algebra is closed.** `bell.adjoint()` is a `Block` (isinstance), named `bell†`; `bell.adjoint().adjoint()` acts as `bell`; `bell.adjoint().controlled(c)` chains and matches the kron-built reference; `block_counts()` reports the derived names.

---

*Decoherence tests (Phase 2.5). Lettered separately to keep phase order without renumbering.*

**TD1 — Coherence decays with coupling strength.** Prepare $|+\rangle$, apply `dephasing_coupling` at angle $\theta$, assert the Bloch $x$-component equals $\cos(\theta/2)$ to 1e-12 for $\theta \in \{0, \pi/4, \pi/2, 3\pi/4, \pi\}$. At $\theta = \pi$ the Bloch vector is the origin: maximally mixed.

**TD2 — Interference is destroyed.** `H`, then couple at $\theta$, then `H`. At $\theta = 0$ the outcome is $|0\rangle$ with probability 1; at $\theta = \pi$ it is 50/50. Assert the visibility follows $\cos(\theta/2)$ across the range. This is the two-slit experiment.

**TD3 — Quantum eraser.** Couple at $\theta = \pi$; assert coherence ≈ 0 and system entropy ≈ 1 bit. Then uncompute the coupling (`with adjoint():`); assert coherence restored to 1e-12, entropy back below 1e-12, and the final `H` again gives $|0\rangle$ deterministically. **Decoherence is reversible if you kept the environment.**

**TD4 — Populations vs. coherences.** Dephasing leaves the diagonal of the reduced density matrix unchanged to 1e-12 while off-diagonals decay. Amplitude damping changes both. Assert the distinction.

**TD5 — Einselection.** With `pointer_coupling(basis="z")`: computational-basis populations are preserved, $\{|+\rangle,|-\rangle\}$ coherence is destroyed. With `basis="x"`: exactly the reverse. Assert both directions. The surviving basis is set by the interaction, not by the state.

**TD6 — Dilation reproduces the Kraus channel.** For each coupling, compare the reduced density matrix obtained by tracing out the environment against $\sum_k K_k \rho K_k^\dagger$ computed independently in the test file with explicit small matrices. Agreement to 1e-12. This is the test that proves the dilations are the channels they claim to be, and it is the most important one in this group.

**TD7 — Symmetry of entanglement entropy.** For a globally pure state, the system and environment entropies are equal. Assert over 20 random couplings. A free consistency check on the partial trace, and a genuinely surprising fact worth surfacing.

---

**T11 — QFT against FFT.** For a random 5-qubit state, the QFT output matches `np.fft.ifft(psi_flat) * sqrt(2**n)` to 1e-12, after accounting for bit ordering. Include the bit-reversal handling explicitly in the test so the convention is documented by example.

**T12 — QFT ∘ IQFT = identity** to 1e-13.

**T13 — Approximate QFT error scaling.** Fidelity loss vs. exact QFT decreases roughly as $2^{-m}$ as the truncation level `m` increases. Assert monotonic improvement and that `m = 8` gives fidelity > 0.999 at `t = 12`.

**T14 — Phase estimation exactness.** For $U|1\rangle = e^{2\pi i \phi}|1\rangle$ with $\phi = 3/8$, exactly representable in 3 bits, a 3-qubit phase register returns `0b011` with probability > 0.999.

**T15 — Semiclassical agreement.** Semiclassical phase estimation returns the same distribution as the coherent version over 500 seeded shots (χ² or total-variation-distance check).

**T16 — Adders.** Exhaustive over all inputs for 3- and 4-bit registers: `cuccaro_adder` computes `b + a` correctly, `modular_adder` computes `(a+b) mod N`, `modexp` computes `a^x mod N` for every `x`. These are classical correctness tests and must pass exactly.

**T17 — Precision comparison.** Run the same period-finding circuit in `complex64` and `complex128`. The peak amplitudes agree to ~7 digits; the near-zero valleys disagree completely. Assert both: peaks close, valleys far. This test demonstrates that badly-conditioned outputs are exactly the ones whose values don't matter.

**T18 — Uncomputation matters (the demonstration test).** Run Shor's period-finding on $N=15$ twice: once with correct uncomputation of the arithmetic ancillas, once with the uncompute step deliberately skipped (via a test-only escape hatch that bypasses the §4.3 assertion). The first recovers $r=4$; the second fails to produce a clean peak. Assert the peak-to-background ratio differs by at least an order of magnitude.

This is the most important test in the suite. It is the empirical proof that garbage entanglement destroys interference.

It is also TD2 in different clothing: the dirty ancillas are an environment that has recorded which-path information, and the lost peak is lost visibility. The test docstring should say so and point back to §4.4.

**T19 — Entanglement across modexp.** Measure entanglement entropy between the exponent and work registers before, during, and after modular exponentiation. Assert it starts at 0 and saturates near $\log_2 r$. This makes the algorithm's mechanism visible.

**T20 — Grover.** For $n=6$ ($N=64$) with a single marked item, after $\lfloor \pi/4 \cdot \sqrt{N}\rfloor = 6$ iterations, success probability > 0.99. Also assert the predicted $\sin^2((2k+1)\theta)$ curve is matched to 1e-9 for $k = 0..10$, including the *decrease* past the optimum.

**T21 — Deutsch-Jozsa.** Constant oracle gives all-zeros with probability 1; balanced oracle gives all-zeros with probability 0.

**T22 — Shor end-to-end, N=15.** With `a=7` (period 4), recovers factors {3, 5}. Seeded, deterministic. Assert qubit count is 11.

**T23 — Shor end-to-end, N=21.** Recovers {3, 7}. Allow retries over `a`; assert success within 10 attempts.

**T24 — Shor honesty check.** Assert `qc.gate_counts()` for the `modexp` block contains a substantial number of `Toffoli` gates (>100 for N=15) and that no gate in the history acts on more than 3 qubits. This is a structural test that the modular exponentiation was genuinely compiled rather than shortcut. The >100 floor is deliberately loose — Beauregard's construction for a 4-bit $N$ lands in the thousands; the test's job is to catch a permutation-matrix shortcut (zero Toffolis), not to pin the exact count.

**T25 — Semiclassical Shor.** `shor(15, semiclassical=True)` succeeds using fewer qubits than the coherent version; assert the qubit count.

---

## 10. Visualization (`qsim.viz`)

Not optional. Watching the QFT output go from flat to a comb of peaks at multiples of $2^t/r$ is where Shor's stops being symbol manipulation.

```python
viz.amplitudes(qc, *, phase_as_hue=True)   # bar heights = |amp|, color = arg(amp)
viz.probabilities(qc, top=32)
viz.bloch(q)                               # circuit resolved from the handle (Qubit.circuit, Phase 3)
viz.entropy_trace(qc)                       # entropy vs. gate index, requires replay
viz.circuit(qc)                             # text/matplotlib diagram from history
```

`entropy_trace` replays the recorded history from scratch, sampling entropy after each gate. Slow, and that is fine. Implemented as a tape-hook client (§4.6): re-execute the history on a fresh circuit with an `on_op` entropy hook attached — the plotting is separate from the replaying, and the hook mechanism is the same one users get.

### 10.1 Jupyter rich display

The notebook is a primary surface (§0), so the core objects render themselves:

- `Circuit._repr_html_()` — qubit count plus the top-$k$ basis states as amplitude bars with phase as hue: an HTML miniature of `viz.amplitudes`.
- `qc.inspect.ket()` returns a small `Ket` object whose `_repr_latex_` renders Dirac notation: a Bell state displays as $\frac{1}{\sqrt2}(|00\rangle + |11\rangle)$, not `array([0.707+0j, ...])`. For learning, seeing states in ket notation by default is worth more than any plot.
- `reduced_density_matrix()` returns a thin ndarray wrapper with a Hinton-diagram `_repr_html_` (magnitude as square size, phase as hue). It still behaves as a plain array everywhere else.

### 10.2 Interactive widgets

`viz.interact_*` helpers built on ipywidgets — one per flagship demo, sliders only, not a framework:

- `viz.interact_dephasing(qc, q)` — slider over $\theta$ driving a live Bloch sphere and the two-slit visibility curve. TD2 as a toy you can touch.
- `viz.interact_grover(n)` — iteration-count slider against the $\sin^2((2k+1)\theta)$ curve, including the overshoot past the optimum.
- `viz.interact_qft_comb(N, a)` — watch the comb of peaks at multiples of $2^t/r$ form as the exponent register grows.

---

## 11. Phase Plan

A phase is done when its tests pass **and** its notebook executes top-to-bottom (`uv run jupyter execute`, §0.5).

**Phase 1 — Core.** §1, §2, §3, §5, §6, plus the basic visual surface: `viz.amplitudes`, `viz.probabilities`, `viz.bloch`, and the rich-display hooks (§10.1). These are ~50 lines each and every later phase narrates through them — deferring them to Phase 6 would mean building the whole library blind. Tests T1–T7. Notebooks 01–02. At the end of this phase a user can build Bell and GHZ states, measure entanglement entropy, and *see* all of it.

**Phase 1.5 — Entanglement demos.** §8.6. Tests TB1–TB3. Notebook 03. Needs only Phase 1 machinery (mid-circuit measurement, classical feedback, `inspect.expectation`).

**Phase 2 — Combinators.** §4. Tests T8–T10. Notebook 04. The ancilla assertion is the deliverable that matters.

**Phase 2.5 — Decoherence and mixed states.** §4.4. Tests TD1–TD7. Requires only environment-qubit marking; partial trace and Bloch vectors already exist from Phase 1. Deliverable is `decoherence.py` plus notebook 05 — coherence decay, the two-slit visibility curve, the eraser, einselection — with the `viz.interact_dephasing` slider (§10.2).

Placed here deliberately. Decoherence is arguably the most important conceptual content in the library, it costs almost nothing once combinators exist, and it makes T18 comprehensible rather than merely empirical when Phase 5 arrives.

**Phase 3 — QFT and phase estimation.** §8.1, §8.2. Tests T11–T15, T17. Notebook 06.

**Phase 4 — Arithmetic.** §8.3. Test T16. Expect this to be the largest and most finicky phase.

**Phase 5 — Shor.** §8.4. Tests T18, T19, T22–T25. Notebook 07, including the T18 demonstration run interactively and `viz.interact_qft_comb`.

**Phase 6 — Grover, DJ, remaining visualization.** §8.5, plus the rest of §10: `viz.circuit` diagrams, `viz.entropy_trace`, `viz.interact_grover`. Tests T20, T21. Notebook 08.

**Phase 7 (later) — Emergence of the classical; trajectories, then error correction.** Sample Pauli errors stochastically on the pure state and average over runs. Same physics as a noise channel, no $2^{2n}$ storage, and the unravelling is instructive in its own right. This is the route to small error-correcting codes — 3-qubit repetition, Shor's 9-qubit, Steane's 7-qubit — with syndrome extraction, since those need a noise knob rather than exact channel arithmetic.

Two further demos state the library's orientation (§0) outright — the classical world growing out of the quantum one:

- **Quantum Darwinism.** Couple one system qubit to many environment qubits through `pointer_coupling`; show with `inspect.mutual_information` that each *small fragment* of the environment holds the same, redundant record of the pointer observable. Objectivity = redundancy: many observers agree about the system because each reads a copy of the same which-path record. Costs nothing beyond §4.4 machinery.
- **Trotterized time evolution.** A small transverse-field Ising chain evolved by Trotter steps built from `Rz`/`Rx`/`CNOT`, watching magnetization oscillate. Hamiltonians, not gates, are the native language of quantum mechanics; this is the bridge from circuits back to dynamics, and the setting where the (semi)classical behavior of expectation values can be watched emerging directly.

**Phase 8 (optional) — Density matrix backend.** A second backend storing $\rho$ of shape `(2,)*2n` with Kraus channels applied directly. Deliberately last: it is a parallel reimplementation of gates, measurement, and every algorithm, and forking the codebase before the arithmetic layer is stable means refactoring twice.

Arriving here *after* Phase 2.5 also frames it correctly. Having already built decoherence by dilation, $\rho$ is not a new kind of object — it is bookkeeping for an environment you have chosen not to track. Worth building only if you want exact channel fidelities rather than sampled ones.

---

## 12. Style Notes for Implementation

- Every module docstring should state the physical fact the module makes concrete.
- Error messages are teaching surfaces. `NoCloningError` should explain the theorem, not just report the violation.
- Prefer explicit loops over clever vectorization when the loop matches the physics. A gate applied qubit-by-qubit in a readable loop beats a broadcast trick that obscures which axis is which.
- Type hints throughout. `Qubit` and `Register` should be distinguishable by a type checker so mixing them is caught statically.
- Runtime dependencies are `numpy` and `matplotlib`, nothing else; the Jupyter stack is a dev dependency (§0.5). `matplotlib` is imported lazily inside `viz` and the display hooks so the core library imports in a bare interpreter.
