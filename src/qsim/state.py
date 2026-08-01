"""The state tensor and the gate-application kernels.

**The physical fact this module makes concrete:** the state of n qubits is a single
array of shape ``(2,) * n``, and *the axes of that array are the tensor factors of
the Hilbert space*. Axis k is qubit k. Nothing else in qsim is as important as this
identification, so it is worth being slow and precise about what it means.

What is a state?
----------------
One qubit's state is a unit vector in C^2: a pair of complex numbers (a, b) with
|a|^2 + |b|^2 = 1. Those numbers are called **amplitudes**. They are not
probabilities — they are complex, and they can cancel each other out, which is the
entire reason quantum computing is interesting. You get probabilities by squaring
their magnitudes: measure this qubit and you see 0 with probability |a|^2 and 1 with
probability |b|^2. (That rule is the **Born rule**; ``measure.py`` introduces it
properly.)

n qubits together live in the tensor product of n copies of C^2, a space of dimension
2^n. A vector there needs 2^n complex numbers — one amplitude per bit pattern. We
could store them in a flat array of length 2^n, but we store them in an array of
shape ``(2,) * n`` instead, because then *indexing is physics*: ``psi[0, 1, 1]`` is
the amplitude of the basis state |011>, and "qubit 2" is literally "axis 2". A gate
on one qubit becomes a 2x2 matrix applied along one axis; entangling two qubits ties
two axes together. This is not a flat vector reshaped for convenience — the shape
*is* the structure of the space.

Indexing convention (stated once, obeyed everywhere)
----------------------------------------------------
``psi[b_0, b_1, ..., b_{n-1}]`` is the amplitude of |b_0 b_1 ... b_{n-1}>, and
**qubit 0 is the most significant bit** when you read the bits as an integer. So in
a 2-qubit circuit, ``psi[1, 0]`` is the amplitude of |10>, which as an integer is 2.

This falls out of NumPy for free: a C-order ``reshape(-1)`` walks the last axis
fastest, so axis 0 varies slowest — exactly what "most significant" means.
Bit-ordering mistakes are the most common bug in quantum simulators, so the
convention lives here, in one place, and everything else refers back to it.

A note on style
---------------
Every function here takes ``psi`` first and returns a **new** array rather than
mutating in place. Copying is wasteful and we do not care: this library is for
seeing mechanisms, not for speed, and code that never mutates its input is code you
can reason about one line at a time.

Precision (design doc §1.5)
---------------------------
The default dtype is ``complex128`` (two 64-bit floats per amplitude).
``qsim.set_dtype(np.complex64)`` switches to single precision — not to save memory,
but so you can *watch* precision degrade in the comparison experiment of design doc
§9 (T17). Three facts worth knowing about the error behaviour:

- Floating point gives *relative* precision per component, so a small amplitude is
  not stored any less accurately than a large one just for being small. Underflow is
  irrelevant here: even ``complex64`` holds normal numbers down to 2^-126.
- Errors accumulate **additively, not multiplicatively**, because every gate is
  unitary — a unitary has condition number 1, so it can neither amplify nor damp an
  existing error. Over G gates the 2-norm error grows like O(sqrt(G) * eps), a random
  walk, rather than exponentially.
- Circuits never form one long sum. The QFT's implicit 2^n-term sum is realized as a
  depth-n binary tree of gates, so its rounding error carries the pairwise-summation
  constant n*eps instead of 2^n*eps.
"""

from typing import Any

import numpy as np

# The dtype every new state tensor is built with. A module-level default rather than
# a parameter threaded through every call, because switching precision is a global
# experiment ("rerun everything in float32"), not a per-circuit choice.
_default_dtype: np.dtype[Any] = np.dtype(np.complex128)


def set_dtype(dtype: Any) -> None:
    """Set the default amplitude dtype for states created from now on.

    Only ``complex64`` and ``complex128`` are meaningful. Existing circuits keep the
    dtype they were built with; this affects new ones.
    """
    resolved = np.dtype(dtype)
    if resolved not in (np.dtype(np.complex64), np.dtype(np.complex128)):
        raise ValueError(
            f"amplitudes must be complex, got dtype {resolved}. qsim supports "
            "np.complex128 (the default) and np.complex64 (single precision, for "
            "watching rounding error accumulate). A real dtype cannot represent a "
            "quantum amplitude: the complex phase is what interferes."
        )
    global _default_dtype
    _default_dtype = resolved


def get_dtype() -> np.dtype[Any]:
    """Return the dtype new state tensors are currently built with."""
    return _default_dtype


def zero_state(n: int, dtype: Any = None) -> np.ndarray:
    """Return the n-qubit state |00...0>, as an array of shape ``(2,) * n``.

    All the amplitude sits on one basis state: index (0, 0, ..., 0) is 1 and every
    other entry is 0. This is the state every circuit starts in, and the state an
    ancilla must be returned to before it can be released.

    ``n = 0`` gives an array of shape ``()`` holding the single number 1. That is not
    a degenerate case to be worked around — zero qubits really do span a
    1-dimensional space, and starting there makes allocation uniform: tensoring a
    fresh |0> onto it produces exactly the 1-qubit state.
    """
    if n < 0:
        raise ValueError(f"a circuit cannot have {n} qubits; n must be 0 or more")
    psi = np.zeros((2,) * n, dtype=_default_dtype if dtype is None else np.dtype(dtype))
    # Index a 0-d array with the empty tuple; for n > 0 this is psi[0, 0, ..., 0].
    psi[(0,) * n] = 1.0
    return psi


def apply_1q(psi: np.ndarray, u: np.ndarray, k: int) -> np.ndarray:
    """Apply the single-qubit gate ``u`` (shape (2, 2)) to qubit ``k``.

    Every other qubit is left completely alone — including qubits entangled with k,
    which is the whole trick: a local operation on a shared state.
    """
    # Contract u's column (input) index, axis 1, with the state's axis k. Read it
    # elementwise: for each combination of the *other* qubits' bits, the two
    # amplitudes (a_0, a_1) sitting along axis k get replaced by u @ (a_0, a_1).
    # That is matrix-vector multiplication applied along one axis of the tensor.
    psi = np.tensordot(u, psi, axes=([1], [k]))
    # tensordot always puts the surviving index of the first argument (u's row
    # index, the new value of qubit k) at position 0, so move it back to position k.
    return np.moveaxis(psi, 0, k)


def apply_2q(psi: np.ndarray, u: np.ndarray, j: int, k: int) -> np.ndarray:
    """Apply the two-qubit gate ``u`` to qubits ``j`` and ``k``.

    ``u`` has shape (2, 2, 2, 2), indexed ``[out_j, out_k, in_j, in_k]`` — the same
    4x4 matrix you would write down by hand, with its row index split into the two
    output bits and its column index split into the two input bits.
    """
    # Contract u's two *input* indices (axes 2 and 3) against the state's axes j and
    # k simultaneously. Summing over both at once is what makes this a genuine
    # two-qubit operation rather than two separate one-qubit ones.
    psi = np.tensordot(u, psi, axes=([2, 3], [j, k]))
    # u's two output indices land at positions 0 and 1; put them back on j and k.
    return np.moveaxis(psi, [0, 1], [j, k])


def apply_diag(psi: np.ndarray, phases: np.ndarray, k: int) -> np.ndarray:
    """Apply a diagonal single-qubit gate to qubit ``k``.

    ``phases`` has shape (2,): the amplitudes where qubit k is 0 are multiplied by
    ``phases[0]``, and those where it is 1 by ``phases[1]``. Z, S, T, Rz and Phase
    are all of this form.

    Doing it this way rather than through ``apply_1q`` is a pedagogical choice, not
    an optimization. A diagonal gate cannot move probability between basis states —
    it only rotates each amplitude's complex phase — and here that is a *structural*
    fact: this code path multiplies each amplitude by a fixed unit-modulus number, so
    it could not change any |amplitude| even if it wanted to.
    """
    # Reshape (2,) -> (1, ..., 1, 2, 1, ..., 1) with the 2 at position k. NumPy
    # broadcasting then stretches it along every other axis, so each amplitude is
    # multiplied by phases[b] where b is its own k-th index — and nothing else.
    shape = (1,) * k + (2,) + (1,) * (psi.ndim - k - 1)
    return psi * phases.reshape(shape)


def _adjust_axes(target_axes: list[int], controls: list[int]) -> list[int]:
    """Renumber target axes for the sliced subspace, where control axes are gone.

    Slicing ``psi[..., 1, ...]`` at a control axis *drops* that axis, so every axis
    after it shifts down by one. A target at axis t therefore sits at
    ``t - (how many control axes come before t)`` inside the slice.
    """
    return [t - sum(1 for c in controls if c < t) for t in target_axes]


def apply_controlled(
    psi: np.ndarray, u: np.ndarray, controls: list[int], target_axes: list[int]
) -> np.ndarray:
    """Apply ``u`` to ``target_axes``, but only where every control qubit is 1.

    This implements the identity

        CU = |0><0| (x) I  +  |1><1| (x) U

    which says: a controlled gate does nothing to the part of the state where the
    control is 0, and applies U to the part where the control is 1. Slicing the array
    along the control axis is *literally* that decomposition — the two terms are the
    two halves of the array, and we transform one of them and leave the other alone.

    Note what is not happening: no larger matrix is ever built. A Toffoli here is the
    same 2x2 X matrix as an ordinary X, applied to a quarter of the amplitudes.
    ``u`` is (2, 2) for one target axis and (2, 2, 2, 2) for two.
    """
    # Select the subspace where every control is |1>. The result is a view with the
    # control axes dropped, not a copy.
    sl: list[Any] = [slice(None)] * psi.ndim
    for c in controls:
        sl[c] = 1
    sub = psi[tuple(sl)]

    adjusted = _adjust_axes(target_axes, controls)
    if len(adjusted) == 1:
        new_sub = apply_1q(sub, u, adjusted[0])
    else:
        new_sub = apply_2q(sub, u, adjusted[0], adjusted[1])

    # Write the transformed half back; the control-is-0 half is copied unchanged.
    out = psi.copy()
    out[tuple(sl)] = new_sub
    return out


def apply_controlled_diag(
    psi: np.ndarray, phases: np.ndarray, controls: list[int], target_axis: int
) -> np.ndarray:
    """Apply a diagonal gate to ``target_axis`` only where every control is 1.

    The controlled form of ``apply_diag`` — used by CZ and CPhase. Same slicing
    construction as ``apply_controlled``, and the same guarantee: only phases move.

    CZ is worth staring at. It is "apply Z to qubit b if qubit a is 1", which sounds
    asymmetric, yet it multiplies the |11> amplitude by -1 and nothing else — so it
    is exactly as much "apply Z to a if b is 1". Control and target are not physically
    distinct roles for a diagonal gate; the names are ours, not nature's.
    """
    sl: list[Any] = [slice(None)] * psi.ndim
    for c in controls:
        sl[c] = 1
    sub = psi[tuple(sl)]

    new_sub = apply_diag(sub, phases, _adjust_axes([target_axis], controls)[0])

    out = psi.copy()
    out[tuple(sl)] = new_sub
    return out


def measure_axis(psi: np.ndarray, k: int, rng: np.random.Generator) -> tuple[int, np.ndarray]:
    """Measure qubit ``k`` in the computational basis. Returns (outcome, new state).

    Wrapped by ``measure.py``, which is where the physics of measurement is
    explained. The mechanics:
    """
    # The probability of reading 1 is the total squared magnitude of the half of the
    # state where axis k is 1. This is the Born rule: probabilities are squared
    # magnitudes of amplitudes, summed over everything we are not asking about.
    sl: list[Any] = [slice(None)] * psi.ndim
    sl[k] = 1
    p1 = float(np.sum(np.abs(psi[tuple(sl)]) ** 2))

    # rng.random() is uniform on [0, 1), so this yields 1 with probability p1. The
    # two extremes are safe without special-casing: if p1 == 0 the comparison is
    # never true, and if p1 == 1 it is always true, so we never select a branch that
    # has no amplitude in it — and never divide by zero below.
    outcome = 1 if rng.random() < p1 else 0

    # Collapse. Zero out the branch that did not happen, then rescale what is left so
    # the total probability is 1 again. Doing this to the *joint* state is the whole
    # point: the other qubits are not touched by hand, yet if they were entangled
    # with k their state is now correct, because the amplitudes that described the
    # other outcome are simply gone.
    psi_after = psi.copy()
    sl[k] = 1 - outcome
    psi_after[tuple(sl)] = 0.0
    p_outcome = p1 if outcome == 1 else 1.0 - p1
    psi_after /= np.sqrt(p_outcome)
    return outcome, psi_after
