"""Teleportation and superdense coding — moving quantum information, never copying it.

**The physical fact this module makes concrete:** an unknown quantum state can be moved
from one qubit to another using a shared entangled pair and two classical bits — and the
original is destroyed in the process, necessarily. Teleportation is not a loophole in the
no-cloning theorem (``NoCloningError``, design doc §3.1); it is that theorem's shape made
visible. You end with one copy, exactly as you began.

Why you cannot do this the obvious way
---------------------------------------
To send someone an unknown qubit state α|0⟩ + β|1⟩ over a telephone, you would have to
learn α and β. You cannot: measuring gives you one bit, chosen randomly, and destroys the
superposition that held the rest. You cannot make a backup first, because copying an
unknown state is exactly what no-cloning forbids. So the state seems stuck.

The protocol
------------
Three qubits. ``msg`` holds the unknown state. ``a`` and ``b`` are the two halves of a
Bell pair, prepared together and then separated — Alice keeps ``msg`` and ``a``, Bob takes
``b`` far away.

1. **Alice entangles the message with her half**: ``CNOT(msg, a)``, then ``H(msg)``.
2. **Alice measures both of her qubits**, getting two ordinary classical bits. This is
   where the message state disappears from her side. She has learned *nothing* about it —
   the two bits are uniformly random regardless of what the state was.
3. **Alice telephones Bob the two bits.** This step is why teleportation carries no
   information faster than light: without the call Bob's qubit is maximally mixed and
   useless.
4. **Bob applies a correction**: ``X`` if the second bit is 1, then ``Z`` if the first is.
   His qubit is now in the message state, exactly.

In this simulator step 3 is a plain Python ``if`` on a measurement result, which is worth
sitting with: "classical communication" in a quantum protocol really is nothing more
exotic than ordinary control flow reading an ordinary bit.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from qsim.circuit import Circuit, Qubit
from qsim.gates import CNOT, H, X, Z


@dataclass(frozen=True)
class TeleportResult:
    """What one run of :func:`teleport` produced."""

    #: Alice's first measurement outcome (the message qubit, after her H).
    m1: int
    #: Alice's second measurement outcome (her half of the Bell pair).
    m2: int
    #: |⟨target|received⟩|² — 1.0 when the state arrived intact.
    fidelity: float
    #: Which fixups Bob applied: "", "X", "Z", or "X then Z".
    corrections: str
    #: What Alice's two qubits are left holding, read off the final state.
    #:
    #: This necessarily equals ``(m1, m2)`` — and that redundancy *is* the point. After
    #: the protocol Alice's qubits carry two random classical bits and no trace of the
    #: message: not a degraded copy, not a partial copy, nothing. The state was moved.
    source_bits: tuple[int, int]
    #: The z-components of Alice's two Bloch vectors, the evidence behind ``source_bits``.
    #:
    #: Each is exactly ±1, which is what "sits at a pole of the Bloch sphere" means: a
    #: definite classical bit, with no superposition and no entanglement left. Anything
    #: short of ±1 would mean part of the message was still lingering on Alice's side —
    #: so this is where the claim "the original is destroyed" is actually checked.
    source_bloch_z: tuple[float, float]


def teleport(
    state_prep: Callable[[Qubit], None], *, seed: int | None = None
) -> TeleportResult:
    """Teleport the state that ``state_prep`` prepares, and report what happened.

    ``state_prep`` is any callable that puts a fresh |0⟩ qubit into the state you want to
    send — ``lambda q: Ry(q, theta=1.1)``, say. It is a function rather than a state
    vector because the simulator must be able to prepare the same state twice: once to
    teleport, and once more on a separate circuit to check what arrived. A real
    experiment could do neither, which is precisely why teleportation is useful.
    """
    qc = Circuit(name="teleport", seed=seed)
    msg, alice, bob = qc.alloc_many(3)

    state_prep(msg)

    # The Bell pair Alice and Bob share. Prepared together, then separated.
    H(alice)
    CNOT(alice, bob)

    # Alice entangles the message with her half and measures both in the computational
    # basis. This is the step that destroys the message on her side.
    CNOT(msg, alice)
    H(msg)
    m1 = qc.measure(msg)
    m2 = qc.measure(alice)

    # "Alice telephones Bob." Classical communication is just control flow reading a bit.
    corrections = []
    if m2 == 1:
        X(bob)
        corrections.append("X")
    if m1 == 1:
        Z(bob)
        corrections.append("Z")

    # Prepare the same state again on its own circuit to see what *should* have arrived.
    reference = Circuit()
    state_prep(reference.alloc())
    target = reference.inspect.state_vector()

    # Bob's qubit is one part of a three-qubit state, so compare against its reduced
    # density matrix: F = ⟨target|rho|target⟩, the probability that a measurement asking
    # "are you the target state?" says yes. vdot conjugates its first argument, giving
    # the bra ⟨target|.
    rho = qc.inspect.reduced_density_matrix([bob])
    fidelity = float(np.vdot(target, rho @ target).real)

    # Read Alice's leftovers off the state rather than trusting m1 and m2. A Bloch
    # z-component of +1 is |0⟩ and -1 is |1⟩, so the sign gives the bit and the
    # magnitude proves there is nothing else there.
    bloch_z = (qc.inspect.bloch_vector(msg)[2], qc.inspect.bloch_vector(alice)[2])
    source_bits = tuple(0 if z > 0 else 1 for z in bloch_z)

    return TeleportResult(
        m1=m1,
        m2=m2,
        fidelity=fidelity,
        corrections=" then ".join(corrections),
        source_bits=(source_bits[0], source_bits[1]),
        source_bloch_z=bloch_z,
    )


def superdense_send(bits: tuple[int, int], *, seed: int | None = None) -> tuple[int, int]:
    """Send two classical bits by handing over one qubit. Returns the decoded pair.

    Superdense coding is teleportation run backwards. Teleportation spends one Bell pair
    and two classical bits to move one qubit; superdense coding spends one Bell pair and
    one qubit to move two classical bits. The same resource, traded in the other
    direction.

    The trick is that the four Bell states are mutually distinguishable, and Alice can
    move between all four by acting on her half alone: doing nothing, X, Z, or both. So
    her one qubit — combined with the half Bob already had — carries two bits. Note the
    bookkeeping is honest: two qubits are involved, Bob just happened to receive one of
    them earlier, before there was any message to send.
    """
    if any(b not in (0, 1) for b in bits):
        raise ValueError(f"superdense coding sends two classical bits, each 0 or 1; got {bits}.")

    qc = Circuit(name="superdense", seed=seed)
    alice, bob = qc.alloc_many(2)

    # The shared Bell pair, distributed in advance.
    H(alice)
    CNOT(alice, bob)

    # Alice encodes both bits into her single qubit, moving the pair to one of the four
    # Bell states. Neither gate touches Bob's qubit.
    if bits[0] == 1:
        Z(alice)
    if bits[1] == 1:
        X(alice)

    # Alice hands her qubit to Bob, who now holds both and undoes the entangling circuit.
    # Reversing H-then-CNOT turns each Bell state back into a distinct basis state.
    CNOT(alice, bob)
    H(alice)
    return (qc.measure(alice), qc.measure(bob))
