"""Measurement: the one operation that destroys information.

**The physical fact this module makes concrete:** measurement is not "looking at a
value that was already there". Before you measure, there is no value — there is a
superposition, and every gate so far has depended on *all* of its parts. Measuring
picks one branch at random, with probability given by the squared magnitude of its
amplitude (the **Born rule**), and throws the rest away. That is why it is the only
non-unitary, non-reversible thing a quantum computer does.

Why collapse must be a projection of the *joint* state
-------------------------------------------------------
It would be much easier to implement measurement as "compute the probability that
this qubit reads 1, flip a weighted coin, report the answer". That would give the
right answer for the qubit you asked about, and the wrong state for everything else.

Take a Bell pair, (|00> + |11>)/sqrt(2). Measure the first qubit and get 0. The
second qubit is now *definitely* 0 — not "probably", and not "0 in our bookkeeping
while its amplitudes still say otherwise". If we only sampled a marginal, the second
qubit would still be recorded as an even superposition, and a later measurement of it
could report 1, which never happens in reality.

Doing it properly is barely more work and is what makes the correlations come out
right: zero the amplitudes of the branch that did not happen, then rescale so the
total probability is 1 again. The conditional state of every other qubit is then
correct automatically, because the amplitudes that described the other outcome are
simply gone. Nothing propagates the information "outward" — it was never stored
separately in the first place.
"""

from typing import TYPE_CHECKING

from qsim import state
from qsim.circuit import Op
from qsim.errors import QsimError

if TYPE_CHECKING:
    from qsim.circuit import Circuit, Qubit, Register


def measure(circuit: Circuit, q: Qubit) -> int:
    """Measure ``q`` in the computational basis. Returns 0 or 1, and collapses the state.

    "In the computational basis" means we are asking the question "is this qubit |0>
    or |1>?" — as opposed to asking whether it is |+> or |->, which is a different
    and equally valid question with a different answer. Every measurement is a
    measurement *of* something; the basis is what you chose to ask.

    Measuring the same qubit twice gives the same answer both times: the first
    measurement leaves it in the state it reported, so the second is no longer
    random. Both are recorded in the history.
    """
    # A hook is an observer of the tape, and measuring would both write to the tape and
    # collapse the state out from under the op the hook was called about. Same message
    # as for a gate emitted from a hook, and it lives in one place.
    circuit._refuse_while_hooked()
    if circuit._record_stack:
        raise QsimError(
            f"cannot measure {q._name} inside a combinator scope. Measurement is the "
            "one irreversible operation in the library: it destroys the superposition "
            "it reports on, so there is nothing to replay backwards and no way to "
            "condition it on a control that is itself in superposition. Every other "
            "operation here is a unitary, and unitaries can always be inverted and "
            "controlled — that is exactly why the scopes work. Measure after the "
            "scope has closed."
        )
    axis = circuit._axis(q)
    outcome, psi = state.measure_axis(circuit._psi, axis, circuit._rng)
    circuit._psi = psi
    circuit._record(Op(name="measure", qubit_ids=(q._id,), result=outcome))
    return outcome


def measure_all(circuit: Circuit, reg: Register) -> int:
    """Measure every qubit of ``reg`` and return the outcome as one integer.

    ``reg[0]`` is the most significant bit, matching the convention everywhere else
    in qsim: a two-qubit register found in |10> returns 2, not 1.

    The qubits are measured one at a time, in order, each collapsing the state before
    the next is measured. For an entangled register that ordering matters to the
    intermediate states — and not at all to the distribution of final results, which
    is a small miracle worth noticing.
    """
    value = 0
    for q in reg:
        # Shift left and drop the new bit in at the bottom: the first qubit measured
        # ends up in the highest bit position.
        value = (value << 1) | measure(circuit, q)
    return value


def reset(circuit: Circuit, q: Qubit) -> None:
    """Return ``q`` to |0>, whatever it was doing before.

    Measure it, and flip it if it read 1. This is how real hardware resets a qubit —
    there is no "set to zero" operation, because that would be a non-reversible
    process applied to an unknown state. Measuring first makes the state known, and
    then flipping it is an ordinary reversible gate.

    Note the cost: this destroys any superposition or entanglement the qubit had.
    Phase 2's ancilla scopes exist precisely so that scratch qubits can be returned
    to |0> *without* measuring them, by undoing the computation that dirtied them.
    """
    from qsim.gates import X

    if measure(circuit, q) == 1:
        X(q)
