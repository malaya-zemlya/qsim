"""The Inspector: everything you are not allowed to do.

**The physical fact this module makes concrete:** you cannot read amplitudes off a
real quantum computer. You can only measure, which gives you one bit per qubit and
destroys the superposition that produced it. To learn a real state you would have to
prepare it again and again, measuring in different bases each time, and reconstruct
it statistically — that is *tomography*, and it costs exponentially many runs.

So the boundary of this namespace is the boundary between what the math knows and
what an experiment can extract. Everything reachable through ``qc.inspect`` is
cheating. That is exactly why it is useful for learning, and exactly why it is kept
behind a name that says so out loud.

A note on subsets
-----------------
Several methods take a "subset" of qubits — any sequence of handles, so a
``Register``, a list, or a tuple all work. The subset is *which part of the system
you are choosing to look at*; the rest is what you are choosing to ignore. In
quantum mechanics that choice is not innocent, and ``reduced_density_matrix`` is
where it shows up.
"""

from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from qsim.errors import DirtyAncillaError

if TYPE_CHECKING:
    from qsim.circuit import Circuit, Qubit, Register

# The Pauli matrices, the standard basis for one-qubit observables. I is included so
# that a Pauli string can say "ignore this qubit".
_PAULI: dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=np.complex128),
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.array([[1, 0], [0, -1]], dtype=np.complex128),
}


def basis_label(index: int, n_qubits: int) -> str:
    """The bitstring naming one basis state, qubit 0 first: ``basis_label(2, 3) == "010"``.

    A circuit with no qubits gets the empty label, ``|⟩`` — its one basis state has no
    bits to name it with, which is what a 1-dimensional space looks like.
    """
    return format(index, f"0{n_qubits}b") if n_qubits else ""


def _format_amplitude(amp: complex) -> tuple[str, str]:
    """Return (separator, text) for one term of a Dirac-notation sum.

    Amplitudes print as plain decimals to three places. Recognizing exact values and
    printing "1/sqrt(2)" was considered and rejected: it works beautifully for the
    handful of textbook states and then silently stops the moment you apply a T gate,
    which is precisely when you most need to trust the output.
    """
    if abs(amp.imag) < 5e-4:
        if amp.real < 0:
            return " - ", f"{abs(amp.real):.3f}"
        return " + ", f"{amp.real:.3f}"
    sign = "+" if amp.imag >= 0 else "-"
    return " + ", f"({amp.real:.3f}{sign}{abs(amp.imag):.3f}i)"


class _DiracSum:
    """Shared formatting for ``Ket`` and ``Bra``: a list of (bitstring, amplitude)."""

    def __init__(self, terms: list[tuple[str, complex]], n_hidden: int) -> None:
        self.terms = terms
        self.n_hidden = n_hidden

    def _render(self, left: str, right: str, conjugate: bool) -> str:
        if not self.terms:
            return "0"
        pieces: list[str] = []
        for i, (bits, amp) in enumerate(self.terms):
            value = amp.conjugate() if conjugate else amp
            sep, text = _format_amplitude(value)
            if i == 0:
                pieces.append(f"{'-' if sep == ' - ' else ''}{text}{left}{bits}{right}")
            else:
                pieces.append(f"{sep}{text}{left}{bits}{right}")
        out = "".join(pieces)
        if self.n_hidden:
            out += f" + … ({self.n_hidden} more term{'s' if self.n_hidden > 1 else ''})"
        return out


class Ket(_DiracSum):
    """A state written in Dirac notation: ``0.707|00⟩ + 0.707|11⟩``.

    Dirac notation writes a basis state as |0101⟩ ("ket") and a superposition as a
    weighted sum of them. The weights are the amplitudes. Terms are ordered by
    magnitude, largest first, so a truncated display drops the least important ones.
    """

    def __str__(self) -> str:
        return self._render("|", "⟩", conjugate=False)

    def __repr__(self) -> str:
        return str(self)

    def _repr_latex_(self) -> str:
        # Jupyter renders this as typeset math.
        body = self._render(r"\left|", r"\right\rangle", conjugate=False)
        return f"${body.replace('…', r'\dots')}$"


class Bra(_DiracSum):
    """The dual of a state: ``0.707⟨00| + 0.707⟨11|``.

    A "bra" ⟨ψ| is the conjugate transpose of the ket |ψ⟩ — a row vector where the ket
    is a column, with every amplitude complex-conjugated. It exists so that ⟨φ|ψ⟩ can
    be written as a bra meeting a ket: the inner product, a single complex number
    measuring how much the two states overlap.

    A bra carries no information the ket does not. Printing one is worth it for one
    reason: the conjugation is visible. Every ``+0.354i`` becomes ``-0.354i``.
    """

    def __str__(self) -> str:
        return self._render("⟨", "|", conjugate=True)

    def __repr__(self) -> str:
        return str(self)

    def _repr_latex_(self) -> str:
        body = self._render(r"\left\langle", r"\right|", conjugate=True)
        return f"${body.replace('…', r'\dots')}$"


class Inspector:
    """Read-only access to everything about a circuit's state. Accessed as ``qc.inspect``."""

    def __init__(self, circuit: Circuit) -> None:
        self._circuit = circuit

    # ---- raw state -------------------------------------------------------------

    def state_tensor(self) -> np.ndarray:
        """The state as its native array of shape ``(2,) * n``. A copy, so you cannot edit it."""
        return self._circuit._psi.copy()

    def state_vector(self) -> np.ndarray:
        """The state flattened to length 2**n, indexed by the basis state as an integer."""
        # C-order reshape walks the last axis fastest, so axis 0 varies slowest —
        # which is the statement "qubit 0 is the most significant bit" in code.
        return self._circuit._psi.reshape(-1).copy()

    def amplitude(self, bits: str) -> complex:
        """The amplitude of one basis state, e.g. ``amplitude("0101")``."""
        n = self._circuit.n_qubits
        if len(bits) != n or any(b not in "01" for b in bits):
            raise ValueError(
                f"expected a string of {n} characters, each '0' or '1' — one per qubit, "
                f"qubit 0 first — but got {bits!r}."
            )
        # The bitstring is literally the index tuple: psi[0,1,0,1] is the amplitude
        # of |0101>. This is the payoff of storing the state as a (2,)*n tensor.
        return complex(self._circuit._psi[tuple(int(b) for b in bits)])

    def probabilities(self) -> np.ndarray:
        """The probability of each basis state: |amplitude|^2, flattened. Does not collapse."""
        return np.abs(self.state_vector()) ** 2

    def norm(self) -> float:
        """The length of the state vector, which must always be 1.

        Total probability. Every gate is unitary, meaning length-preserving, so this
        staying at 1 across a long circuit is a running check that nothing is broken.
        """
        return float(np.linalg.norm(self._circuit._psi))

    def sample(self, shots: int = 1000) -> Counter[str]:
        """Simulate measuring the whole circuit ``shots`` times, without collapsing it.

        This is the one thing here a real machine *can* do — except that a real
        machine would have to rerun the entire circuit for each shot, since the first
        measurement destroys the state. Sampling repeatedly from an intact state is
        the cheat.

        Draws come from a separate random stream, so calling this never changes what
        a subsequent ``qc.measure()`` returns: adding a sample() call to a seeded
        notebook cannot silently rewrite the measurements below it.
        """
        n = self._circuit.n_qubits
        outcomes = self._circuit._sample_rng.choice(2**n, size=shots, p=self.probabilities())
        return Counter(format(int(o), f"0{n}b") for o in outcomes)

    # ---- subsystems ------------------------------------------------------------

    def _axes(self, subset: Sequence[Qubit]) -> list[int]:
        return [self._circuit._axis(q) for q in subset]

    def _matricize(self, kept: list[int]) -> np.ndarray:
        """Reshape the state into a matrix: kept qubits index rows, the rest columns."""
        psi = self._circuit._psi
        # Move the kept axes to the front (order preserved), then flatten the two
        # groups. M[i, j] is the amplitude of (kept-bits = i, other-bits = j) — the
        # state rewritten as a matrix over the chosen split of the system.
        moved = np.moveaxis(psi, kept, range(len(kept)))
        return moved.reshape(2 ** len(kept), -1)

    def reduced_density_matrix(self, subset: Sequence[Qubit]) -> np.ndarray:
        """The state of ``subset`` alone, as a density matrix.

        A **density matrix** generalizes a state vector so it can also describe
        statistical mixtures — "this system is in |0> or |1>, we don't know which"
        as opposed to "this system is in a superposition of both". Its diagonal holds
        the probabilities of each basis state; its off-diagonal entries, called
        **coherences**, are what remains of superposition. A pure state has large
        coherences; a classical mixture has none.

        The operation that produces it is the **partial trace**: averaging away
        everything you chose not to look at. If the subset is entangled with the rest,
        the result is a mixture even though the whole system is in a perfectly
        definite pure state — which is the precise sense in which a part of an
        entangled system has no state of its own.
        """
        m = self._matricize(self._axes(subset))
        # Summing over the column index (the qubits we're ignoring) is the partial
        # trace: rho[i, i'] = sum_j M[i, j] * conj(M[i', j]).
        return m @ m.conj().T

    def schmidt_spectrum(self, cut: Sequence[Qubit]) -> np.ndarray:
        """The Schmidt coefficients across the split between ``cut`` and everything else.

        Any pure state of a bipartite system can be written as a single sum
        ``sum_i s_i |a_i>|b_i>`` with orthonormal ``|a_i>`` and ``|b_i>`` — the
        **Schmidt decomposition**. The non-negative numbers ``s_i`` are what the SVD
        of the matricized state returns. One nonzero coefficient means a product
        state; several mean entanglement, and how evenly they are spread is how much.
        """
        m = self._matricize(self._axes(cut))
        # compute_uv=False returns just the singular values, which is all we need.
        return np.linalg.svd(m, compute_uv=False)

    def entanglement_entropy(self, subset: Sequence[Qubit], base: float = 2.0) -> float:
        """How entangled ``subset`` is with the rest of the circuit, in bits.

        Zero means the subset is in a state of its own — no entanglement. One bit is
        the maximum for a single qubit, reached by a Bell pair: knowing everything
        about the pair tells you nothing at all about either half.

        Computed from the squared Schmidt coefficients, which are the eigenvalues of
        the reduced density matrix. Going through the SVD instead of an
        eigendecomposition is both numerically better behaved and more honest about
        where the number comes from.
        """
        s = self.schmidt_spectrum(subset)
        p = s**2
        # Drop numerically-zero terms: p*log(p) tends to 0 as p tends to 0, but
        # log(0) is -inf and 0 * -inf is a NaN.
        p = p[p > 1e-15]
        # The trailing + 0.0 turns -0.0 into 0.0. A product state gives exactly one
        # Schmidt coefficient of 1, and -1 * log(1) is negative zero, which would
        # print as "-0.000 bits of entanglement" — true, but alarming to read.
        return float(-np.sum(p * np.log(p)) / np.log(base)) + 0.0

    def is_product(self, subset: Sequence[Qubit], tol: float = 1e-10) -> bool:
        """Whether ``subset`` is unentangled with the rest of the circuit."""
        return self.entanglement_entropy(subset) < tol

    def mutual_information(self, a: Sequence[Qubit], b: Sequence[Qubit]) -> float:
        """Total correlation between two groups of qubits: S(A) + S(B) - S(AB).

        Counts *all* correlation, classical and quantum together. It is the readout of
        the quantum-Darwinism demonstration in Phase 7: when many parts of an
        environment each share high mutual information with the same system, that
        system's state has been redundantly recorded — which is what makes it look
        objective and classical.
        """
        ab = list(a) + list(b)
        return (
            self.entanglement_entropy(a)
            + self.entanglement_entropy(b)
            - self.entanglement_entropy(ab)
        )

    def _probability_all_zero(self, axes: list[int]) -> float:
        """Total probability that every listed axis reads 0."""
        sl: list[Any] = [slice(None)] * self._circuit._psi.ndim
        for axis in axes:
            sl[axis] = 0
        return float(np.sum(np.abs(self._circuit._psi[tuple(sl)]) ** 2))

    def assert_zero(self, subset: Sequence[Qubit], tol: float = 1e-10) -> None:
        """Raise unless every qubit in ``subset`` is certainly |0> and unentangled.

        Used by Phase 2 to check that scratch qubits were properly uncomputed before
        being released.
        """
        leftover = 1.0 - self._probability_all_zero(self._axes(subset))
        if leftover > tol:
            names = ", ".join(q.name for q in subset)
            raise DirtyAncillaError(
                f"{names} are not in |0>: probability {leftover:.3g} of finding a 1. "
                "Scratch qubits must be uncomputed back to |0> before release, not "
                "merely ignored. Leftover entanglement is a *record* of which branch "
                "of the computation happened, and a branch that has been recorded can "
                "no longer interfere with the others — which is where a quantum "
                "algorithm's advantage comes from. Undo the operations that dirtied "
                "these qubits, in reverse order."
            )

    # ---- single-qubit views ----------------------------------------------------

    def bloch_vector(self, q: Qubit) -> tuple[float, float, float]:
        """The qubit's position on the Bloch sphere, as (x, y, z).

        Every state of a single qubit corresponds to a point in a unit ball. Pure
        states sit on the surface: |0> at the north pole, |1> at the south, |+> and
        |-> at opposite points on the equator. Points strictly inside are mixed
        states, and the exact center is the maximally mixed state — no information at
        all. A qubit maximally entangled with something else sits at the center,
        which is the geometric version of "it has no state of its own".

        The three coordinates are the expected values of X, Y and Z, read off the
        reduced density matrix.
        """
        rho = self.reduced_density_matrix([q])
        return (
            float(2 * rho[0, 1].real),
            float(-2 * rho[0, 1].imag),
            float((rho[0, 0] - rho[1, 1]).real),
        )

    # ---- observables and comparisons -------------------------------------------

    def expectation(self, pauli: str, reg: Register | None = None) -> float:
        """The average value of a Pauli observable, e.g. ``expectation("ZZ", reg)``.

        An **observable** is a physical quantity you could measure; a Pauli string
        names one built from X, Y, Z (and I for "ignore this qubit") on each of the
        listed qubits. Its expectation value is the average result you would get from
        many repeated measurements — always between -1 and +1 here, since each Pauli
        has eigenvalues ±1.

        This is the bread and butter of actual quantum mechanics, and it is what the
        CHSH inequality (Phase 1.5) is stated in terms of.
        """
        qubits = list(reg) if reg is not None else list(self._circuit.qubits)
        if len(pauli) != len(qubits):
            raise ValueError(
                f"Pauli string {pauli!r} has {len(pauli)} letters but {len(qubits)} "
                "qubits were given; there must be exactly one letter per qubit."
            )
        from qsim import state as state_mod

        transformed = self._circuit._psi
        for letter, q in zip(pauli, qubits, strict=True):
            if letter not in _PAULI:
                raise ValueError(
                    f"{letter!r} is not a Pauli operator; use 'I', 'X', 'Y' or 'Z'."
                )
            if letter != "I":
                transformed = state_mod.apply_1q(
                    transformed, _PAULI[letter], self._circuit._axis(q)
                )
        # vdot conjugates its first argument, so this is <psi|P|psi>. The result is
        # real because P is Hermitian; the tiny imaginary part is rounding error.
        return float(np.vdot(self._circuit._psi, transformed).real)

    def overlap(self, other: np.ndarray) -> complex:
        """The inner product ⟨other|ψ⟩, phase and all.

        A complex number saying how much the two states resemble each other. Its
        magnitude squared is the probability that a measurement designed to ask "are
        you in state ``other``?" says yes. Its *phase* is invisible to any single
        measurement — and is exactly what decides how two states add when they
        interfere, which is why this returns the raw number and ``fidelity`` does not.
        """
        return complex(np.vdot(np.asarray(other).reshape(-1), self.state_vector()))

    def fidelity(self, other: np.ndarray) -> float:
        """|⟨other|ψ⟩|^2: how close this state is to another, from 0 to 1.

        1 means identical (up to an overall phase, which is unobservable); 0 means
        perfectly distinguishable.
        """
        return float(abs(self.overlap(other)) ** 2)

    # ---- display ---------------------------------------------------------------

    def _terms(self, max_terms: int) -> tuple[list[tuple[str, complex]], int]:
        n = self._circuit.n_qubits
        flat = self.state_vector()
        significant = [i for i in range(len(flat)) if abs(flat[i]) > 5e-4]
        # Largest amplitudes first, so truncation drops the least important terms.
        significant.sort(key=lambda i: -abs(flat[i]))
        shown = significant[:max_terms]
        terms = [(basis_label(i, n), complex(flat[i])) for i in shown]
        return terms, len(significant) - len(shown)

    def ket(self, max_terms: int = 8) -> Ket:
        """The state in Dirac notation, e.g. ``0.707|00⟩ + 0.707|11⟩``."""
        terms, hidden = self._terms(max_terms)
        return Ket(terms, hidden)

    def bra(self, max_terms: int = 8) -> Bra:
        """The state's dual, ⟨ψ| — the same amplitudes, conjugated."""
        terms, hidden = self._terms(max_terms)
        return Bra(terms, hidden)
