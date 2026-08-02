"""Decoherence: noise as unitary coupling to qubits you decline to track.

**The physical fact this module makes concrete:** decoherence is not something that
happens *to* a state. It is what a subsystem looks like when you decline to track the
rest of the world.

Nothing in this module is stochastic, lossy, or approximate. Every "noise channel"
here is an ordinary unitary block built from Phase 1 gates, acting on the system qubit
together with one or two extra qubits. The global state stays pure throughout. The
mixedness you observe appears only when you ask for the *system's* view — see
:meth:`qsim.Circuit.environment` and
:meth:`qsim.inspector.Inspector.system_density_matrix` — and it appears because the
extra qubits now hold a record correlated with the system.

Why this is enough: Stinespring dilation
-----------------------------------------
There is a theorem — the Stinespring dilation, informally the "Church of the Larger
Hilbert Space" — saying that *every* noise process, however messy, can be written as:

1. a perfectly reversible interaction with some extra qubits, followed by
2. a refusal to look at them.

Not "can be approximated by". Every one of them, exactly. So a simulator that only
knows how to apply unitaries to pure states loses nothing by never implementing noise
directly: it implements step (1) literally, and makes step (2) a choice of *view*
rather than an operation on the state.

The payoff is the quantum eraser. Because step (2) never actually touched anything,
the interaction can simply be run backwards and the coherence returns exactly — see
``06-decoherence.ipynb`` §7, and test TD3. A library that implemented noise as random
gates or as a density-matrix update could not do that, because it would have thrown
the record away rather than merely looked past it.

Kraus operators
---------------
The traditional way to write a noise channel is a set of matrices ``K_0, K_1, ...``
with ``sum_k K_k† K_k = I``, acting as ``rho -> sum_k K_k rho K_k†``. Every coupling
below states its Kraus operators in its docstring, and TD6 checks each one
independently: it builds the channel from our dilation, computes ``sum_k K_k rho K_k†``
from the matrices as written, and demands they agree. The docstrings are therefore
tested claims, not decoration.

Where the Kraus operators come from is worth seeing once, because it is the same
partial trace as everywhere else. Expand the post-coupling state over the environment
basis: the branch in which the environment ends up in ``|k>`` carries the system
through some linear map, and that map is ``K_k``. Tracing out the environment sums the
branches incoherently, because different environment states are orthogonal — which is
the precise sense in which *a record destroys interference*.
"""

import numpy as np

from qsim.circuit import Qubit, Register
from qsim.combinators import gate, within
from qsim.errors import QsimError
from qsim.gates import CNOT, H, Ry, S, X, Y, Z

# ---- textbook parameters -> rotation angles ---------------------------------------
#
# Every coupling below takes ``theta``, the angle actually rotated inside the circuit,
# because that is the thing the gates do and hiding it would hide the mechanism. The
# literature usually quotes a channel-native number instead. These two functions
# convert, so a value looked up in a table can be used without arithmetic at the call
# site: ``amplitude_damping_coupling(q, e, theta=damping_angle(0.25))``.


def damping_angle(gamma: float) -> float:
    """The ``theta`` for which :func:`amplitude_damping_coupling` damps by ``gamma``.

    ``gamma`` is the probability that an excited qubit decays to |0>, which is how
    amplitude damping is normally quoted. Our coupling rotates by ``theta`` and
    achieves ``gamma = sin^2(theta/2)``, so this inverts that.
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma is a probability and must lie in [0, 1]; got {gamma}")
    return float(2.0 * np.arcsin(np.sqrt(gamma)))


def dephasing_angle(lam: float) -> float:
    """The ``theta`` for which :func:`dephasing_coupling` shrinks coherence by ``lam``.

    ``lam`` (often written λ) is the fraction of the off-diagonal that is destroyed:
    ``lam = 0`` leaves the qubit untouched, ``lam = 1`` destroys the superposition
    completely. Our coupling multiplies the off-diagonal by ``cos(theta/2)``, so this
    solves ``cos(theta/2) = 1 - lam``.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam is a fraction and must lie in [0, 1]; got {lam}")
    return float(2.0 * np.arccos(1.0 - lam))


# ---- the couplings -----------------------------------------------------------------
#
# Each is a @qsim.gate block rather than a plain function, which buys two things: the
# ops are recorded as a named group (``block_counts()``), and the whole coupling can be
# inverted with ``.adjoint()`` — the eraser depends on that.


@gate
def dephasing_coupling(q: Qubit, env: Qubit, *, theta: float) -> None:
    """Let ``env`` partially learn whether ``q`` is |0> or |1>.

    The environment starts in |0>. Conditioned on ``q`` being |1>, it is rotated by
    ``theta``; conditioned on |0> it is left alone. So the state

        a|0> + b|1>,  environment |0>

    becomes

        a|0>|0_E> + b|1>(cos(theta/2)|0_E> + sin(theta/2)|1_E>).

    Read the environment's two possible states as an answer to "was the qubit 1?". At
    ``theta = 0`` the answer is the same either way and no information has moved. At
    ``theta = pi`` the environment is in |0_E> or |1_E> exactly according to the qubit,
    a perfect record. In between it is a partial, unreliable record — and coherence
    decays by precisely how reliable it is.

    Tracing out the environment leaves the populations |a|^2 and |b|^2 completely
    untouched and multiplies the off-diagonal by ``cos(theta/2)``. That asymmetry is
    the whole content of TD4, and it is worth pausing on: the environment did not push
    the qubit around, and it did not inject randomness. It *learned something*, and the
    superposition died of the learning.

    Kraus operators (environment measured in its computational basis)::

        K_0 = [[1,             0],        K_1 = [[0,             0],
               [0, cos(theta/2)]]                [0, sin(theta/2)]]

    ``theta`` is the rotation angle; use :func:`dephasing_angle` to give the fraction
    of coherence destroyed instead.
    """
    circuit = q._circuit
    with circuit.control(q):
        Ry(env, theta=theta)


@gate
def amplitude_damping_coupling(q: Qubit, env: Qubit, *, theta: float) -> None:
    """Let ``q`` decay toward |0>, its excitation leaking into ``env``.

    This is spontaneous emission: an excited atom drops to its ground state and a
    photon goes off into the world. Unlike dephasing, it moves populations — energy
    genuinely leaves the qubit — which is why TD4 can tell the two apart.

    Built in two steps. First a controlled rotation writes "was it excited?" into the
    environment, then a CNOT lets the environment carry the excitation away::

        |1>|0_E>  ->  cos(theta/2)|1>|0_E> + sin(theta/2)|0>|1_E>
        |0>|0_E>  ->  |0>|0_E>                       (a ground-state atom cannot decay)

    The second line is the reason this channel is not symmetric, and the reason a real
    qubit left alone ends up in |0> rather than in a 50/50 mixture: the environment is
    cold, so excitation flows one way.

    Kraus operators, with ``gamma = sin^2(theta/2)`` the decay probability::

        K_0 = [[1,             0],        K_1 = [[0, sqrt(gamma)],
               [0, sqrt(1-gamma)]]               [0,           0]]

    ``theta`` is the rotation angle; use :func:`damping_angle` to give ``gamma``.
    """
    circuit = q._circuit
    with circuit.control(q):
        Ry(env, theta=theta)
    # The environment now flags "the qubit was excited". This CNOT makes the flag
    # *carry the excitation away* rather than merely record it: where the environment
    # reads 1, the qubit is flipped down to |0>.
    CNOT(env, q)


@gate
def depolarizing_coupling(q: Qubit, env: Register, *, p: float) -> None:
    """With probability ``p``, replace ``q`` by noise: the least structured channel.

    Depolarizing noise applies one of X, Y or Z at random, each with probability
    ``p/3``, and does nothing with probability ``1 - p``. It is the standard
    worst-case model — noise with no preferred direction, which erodes the Bloch
    vector uniformly toward the origin instead of flattening it onto an axis.

    Needs **two** environment qubits, because there are four outcomes to record. They
    are prepared in

        sqrt(1-p)|00> + sqrt(p/3)(|01> + |10> + |11>)

    and then control which Pauli hits the system: |01> -> X, |10> -> Y, |11> -> Z,
    |00> -> nothing. Since those four environment states are orthogonal, tracing them
    out gives exactly the random-Pauli mixture — no randomness was ever drawn.

    Kraus operators::

        K_0 = sqrt(1-p) I,  K_1 = sqrt(p/3) X,  K_2 = sqrt(p/3) Y,  K_3 = sqrt(p/3) Z

    ``p`` is the depolarizing probability, the parameter this channel is always quoted
    by, so there is no angle to convert here.
    """
    if len(env) != 2:
        raise ValueError(
            f"depolarizing_coupling needs exactly 2 environment qubits, got {len(env)}. "
            "It has four outcomes to record — do nothing, X, Y, Z — and four outcomes "
            "need two qubits. Allocate them with qc.environment(2)."
        )
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p is a probability and must lie in [0, 1]; got {p}")

    circuit = q._circuit
    e0, e1 = env[0], env[1]

    # Prepare the weighted environment state. Ry(a)|0> = cos(a/2)|0> + sin(a/2)|1>, so
    # writing the four target amplitudes as products of the two qubits' amplitudes:
    #
    #   |00>: cos(a/2)cos(b/2) = sqrt(1-p)      |10>: sin(a/2)cos(c/2) = sqrt(p/3)
    #   |01>: cos(a/2)sin(b/2) = sqrt(p/3)      |11>: sin(a/2)sin(c/2) = sqrt(p/3)
    #
    # The last two are equal, so c = pi/2. Then sin(a/2)/sqrt(2) = sqrt(p/3) fixes a,
    # and dividing the first pair gives tan(b/2) = sqrt(p/3)/sqrt(1-p), fixing b.
    alpha = 2.0 * np.arcsin(np.sqrt(2.0 * p / 3.0))
    beta = 2.0 * np.arctan2(np.sqrt(p / 3.0), np.sqrt(1.0 - p))
    Ry(e0, theta=alpha)
    Ry(e1, theta=beta)
    # Rotations about the same axis add, so applying (pi/2 - beta) only where e0 is |1>
    # leaves that branch at exactly pi/2 while the other branch keeps beta.
    with circuit.control(e0):
        Ry(e1, theta=np.pi / 2.0 - beta)

    # Now let the environment choose the Pauli. qsim's control scopes trigger on |1>,
    # so a control on |0> is written by conjugating that qubit with X: flip it, use it
    # as an ordinary control, flip it back.
    X(e0)
    with circuit.control(e0, e1):  # environment |01>
        X(q)
    X(e0)

    X(e1)
    with circuit.control(e0, e1):  # environment |10>
        Y(q)
    X(e1)

    with circuit.control(e0, e1):  # environment |11>
        Z(q)


def _onto_computational_basis(q: Qubit, *, basis: str) -> None:
    """Rotate the chosen basis onto the computational one — the V of the sandwich.

    For x that is H, which maps |+>,|-> to |0>,|1>. For y it is H·S†, which maps the
    Y-eigenstates the same way: S† turns (|0> + i|1>)/sqrt2 into |+>, and H then turns
    |+> into |0>. For z there is nothing to do — the chosen basis is already the
    computational one — and an empty V is a perfectly good V, so the caller needs no
    special case.

    A plain function rather than a ``@qsim.gate`` block, so its gates are stamped with
    the coupling that called it rather than showing up as a block of their own.
    """
    if basis == "y":
        S.adjoint()(q)
    if basis in ("x", "y"):
        H(q)


@gate
def pointer_coupling(q: Qubit, env: Qubit, *, theta: float, basis: str = "z") -> None:
    """Dephase ``q`` in a chosen basis — the knob that selects which states survive.

    This is :func:`dephasing_coupling` conjugated into another basis: rotate so that
    the chosen basis becomes the computational one, dephase, rotate back. With
    ``basis="z"`` it *is* plain dephasing.

    The point is **einselection**. Ask which states of ``q`` come through this coupling
    unharmed, and the answer depends entirely on ``basis``:

    - ``basis="z"``: |0> and |1> survive untouched; |+> and |-> are destroyed.
    - ``basis="x"``: |+> and |-> survive untouched; |0> and |1> are destroyed.

    Nothing about |0> makes it more robust, more classical, or more real than |+>. The
    surviving states — the **pointer states** — are selected by *how the environment
    couples*, and by nothing else. Change the interaction and a different set of states
    becomes the stable, classical-looking one.

    This is the beginning of the answer to why the world looks classical. We see
    definite positions rather than superpositions of them because the interactions that
    dominate — light scattering off things, air molecules hitting them — couple through
    position. Position is not privileged by the laws; it is privileged by the coupling.
    Something with different physics would have different pointer states.

    Kraus operators: those of :func:`dephasing_coupling`, conjugated by the basis
    change U, i.e. ``U† K_k U`` with U = I, H, or HS† for z, x, y.
    """
    if basis not in ("x", "y", "z"):
        raise QsimError(
            f"unknown pointer basis {basis!r}. Valid bases are 'z' (the computational "
            "basis, so |0> and |1> survive), 'x' (so |+> and |-> survive), and 'y'. The "
            "basis is what decides which states the environment leaves alone — see the "
            "einselection section of 06-decoherence.ipynb."
        )
    # The structure *is* the sentence "dephasing, conjugated into another basis":
    # `within` applies the basis change now and undoes it on the way out, so the two
    # halves cannot drift apart, and the body in between is plain dephasing.
    with within(_onto_computational_basis, q, basis=basis):
        dephasing_coupling(q, env, theta=theta)
