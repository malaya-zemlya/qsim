"""Phase estimation — the universal quantum measuring instrument.

**The physical fact this module makes concrete:** every unitary's eigenvalues sit on
the unit circle, so each one is nothing but an angle; and an angle that a quantum
computer can *apply* is an angle a quantum computer can *read*, to as many binary
digits as you are willing to spend qubits on.

Why every eigenvalue is an angle
---------------------------------
A unitary U preserves lengths (that is what "unitary" means, and it is why unitaries
are the operations physics permits). If U|v> = lambda|v> for some eigenvector |v>, then
the length of |v> and the length of lambda|v> must agree, so |lambda| = 1. A complex
number of magnitude 1 is a point on the unit circle, which can always be written

    lambda = exp(2*pi*i*phi)      with phi in [0, 1)

and that number phi — a fraction of a full turn — is called **the phase**. So "find the
eigenvalues of U" and "find the phases of U" are the same job, and there is exactly one
real number per eigenvector to find. **Phase estimation** finds phi to t binary digits.

Why this is the important algorithm
------------------------------------
Almost every quantum algorithm with an exponential speedup is phase estimation wearing a
costume. Shor's factoring algorithm is phase estimation of the map "multiply by a, mod
N": the eigenvalues of that map encode the period of a^x mod N, and the period gives the
factors (Phase 5). Quantum chemistry's ground-state energy algorithms are phase
estimation of exp(-iHt), because the phase you read out *is* the energy. Learn this one
circuit and the others stop being separate things to learn.

Phase kickback, the mechanism
------------------------------
The whole algorithm rests on one small surprise. Suppose the target register holds an
eigenstate |v> of U, and we apply a **controlled**-U with some other qubit as control:

    (|0> + |1>)/sqrt2  (x)  |v>
        --> ( |0>|v> + |1>*U|v> ) / sqrt2
         =  ( |0>|v> + |1>*e^{2*pi*i*phi}|v> ) / sqrt2
         =  ( |0> + e^{2*pi*i*phi}|1> )/sqrt2  (x)  |v>

The target came out **completely unchanged** — an eigenstate is by definition the thing
U does not move — and the phase that U was supposed to contribute ended up on the
*control* qubit instead. That is **phase kickback**: the operation appears to act on the
target, and its entire visible effect lands on the qubit that merely watched. The target
is a catalyst; the control is where the answer accumulates. Notice too that the phase is
now attached to a qubit in superposition, which is the only place a phase can ever be
observed — a global phase on the whole state is invisible, a *relative* phase between
two branches is not.

Repeat that with the j-th control qubit applying U a full 2^j times, and its phase
becomes e^{2*pi*i*phi*2^j} — which is the phase of phi shifted j binary places left.
Arrange the shifts so that the register's qubits carry the successive binary digits of
phi, and the register is holding exactly ``qft(|y>)`` for the integer y = 2^t * phi. One
inverse QFT later, the register holds |y> itself, and measuring it reads phi off in
binary.

Two versions live here
-----------------------
:func:`phase_estimation` is the textbook coherent circuit: t qubits, all measured at the
end. :func:`semiclassical_phase_estimation` (Griffiths and Niu, 1996) gets the same
distribution from **one** reused qubit plus classical arithmetic between the
measurements. That the two agree is the *deferred measurement principle* made
empirical — see T15.
"""

import numpy as np

from qsim.algorithms.qft import iqft
from qsim.circuit import Register
from qsim.combinators import Block
from qsim.errors import QsimError
from qsim.gates import H, Phase


def phase_estimation(unitary: Block, target: Register, out: Register) -> None:
    """Write the phase of ``unitary``'s eigenvalue into ``out``, in binary.

        u = ...                       # a @qsim.gate block acting on `target`
        target.encode(1)              # put the target in an eigenstate of u
        phase_estimation(u, target, out)
        y = qc.measure_all(out)       # y / 2**len(out) is the phase

    ``out`` must start in |0...0>. Afterwards it holds (a superposition sharply peaked
    on) the integer y with ``y / 2**t`` the closest t-bit approximation to phi, where
    ``unitary``'s eigenvalue on ``target``'s state is exp(2*pi*i*phi). ``out[0]`` is the
    most significant bit, as everywhere in qsim. Nothing is measured here — measuring is
    the caller's business, and skipping it lets ``qc.inspect`` look at the whole
    distribution at once, which is how T14 and T15 check this function.

    ``target`` need not be an eigenstate
    -------------------------------------
    The argument is called ``target``, not ``eigenstate``, and the distinction is the
    single most useful thing to understand about this algorithm.

    If ``target`` *is* an eigenstate, the reasoning in the module docstring applies
    directly and ``out`` ends up peaked on that one eigenvalue's phase.

    If it is not, write it in the eigenbasis: any state is some superposition
    ``sum_k c_k |v_k>`` of eigenvectors. The circuit is linear, so it runs on every term
    at once, and the result is ``sum_k c_k |y_k>|v_k>`` — a superposition of *answers*,
    each entangled with the eigenvector it came from. Measuring ``out`` then returns one
    phase phi_k, chosen at random with probability |c_k|², and collapses ``target`` onto
    the matching eigenvector.

    That is not a degraded mode; it is the whole trick. Shor's algorithm has no way to
    prepare an eigenstate of its multiplication map — doing so would require knowing the
    period, which is what it is trying to find — so it feeds in |1>, which happens to be
    an equal superposition of exactly the eigenvectors whose phases are the multiples of
    1/r it wants. It gets one of them at random, and one is enough.

    Cost
    ----
    The controlled-U's are applied by honest repetition: 1 + 2 + 4 + ... + 2^{t-1} =
    2^t - 1 applications in total. That is exponential in t and it is *not* a flaw in
    this implementation — it is where phase estimation's cost actually lives, and any
    real use supplies a unitary whose powers it can build cheaply (Shor's squares
    modular multipliers). Taking a shortcut here would hide the one number that decides
    whether the algorithm is affordable.
    """
    t = len(out)
    if t == 0:
        raise QsimError(
            "phase estimation needs at least one output qubit: `out` is empty, so "
            "there is nowhere to write an answer. The size of `out` is the precision "
            "you are asking for — t qubits read the phase to t binary digits — so an "
            "empty register is a request for a number to zero decimal places."
        )

    # Put the output register into an equal superposition of every integer 0..2^t - 1.
    # Each qubit is now a two-branch interferometer that the kicked-back phases will
    # act on; without this the controls would never be in superposition and nothing
    # would be kicked back at all.
    for q in out:
        H(q)

    # out[i] must end up carrying the i-th most significant digit, so it needs the
    # phase of phi shifted (t-1-i) binary places left — that is, U applied 2^(t-1-i)
    # times. Walking the register backwards makes the exponent count up from 0.
    for j, control in enumerate(reversed(out)):
        controlled_u = unitary.controlled(control)
        for _ in range(2**j):
            controlled_u(target)

    # The register now holds exactly what qft(|y>) looks like: t unentangled qubits
    # whose phases are the binary fractions of y = 2^t * phi. Running the transform
    # backwards collapses that phase pattern into the number itself.
    iqft(out)


def semiclassical_phase_estimation(unitary: Block, target: Register, t: int) -> int:
    """Phase estimation with **one** reusable qubit instead of t. Returns the integer y.

        y = semiclassical_phase_estimation(u, target, t=8)
        phi = y / 2**8

    Same answer, same probability distribution, same number of controlled-U
    applications — and t-1 fewer qubits, which on real hardware is the difference
    between an experiment that fits on a chip and one that does not. Due to Griffiths
    and Niu (1996).

    How one qubit can do the work of t
    -----------------------------------
    The insight is that the inverse QFT, when every qubit is about to be measured
    anyway, does nothing that a classical computer cannot do afterwards. Look at the
    iQFT's structure: the qubit measured first needs no correction at all, and every
    later qubit needs a phase rotation whose angle depends only on bits *already
    measured*. Bits already measured are ordinary classical numbers. So instead of
    keeping t qubits alive to hold t bits, keep **one**, and let a Python variable hold
    the bits already extracted.

    The loop, from the least significant digit upward:

    1. reset the phase qubit to |0> and apply H, opening two branches;
    2. apply controlled-U 2^(t-1-step) times, kicking a phase onto it;
    3. apply the correction rotation ``Phase(theta=-2*pi*known/2^(step+1))``, where
       ``known`` is the digits measured so far read as an integer — this is the iQFT's
       cross-terms, computed classically;
    4. H again, turning the two branches' relative phase into a definite 0 or 1;
    5. measure it, record the digit, and reset for the next round.

    Step 3 is ordinary Python arithmetic feeding an ordinary gate, exactly like Bob's
    correction in teleportation (notebook 03): "classical communication" inside a
    quantum protocol is never anything more exotic than an ``if`` reading a bit.

    Why measuring early is allowed
    -------------------------------
    It looks as though measuring partway through must destroy something — that is the
    usual rule. It does not, and the reason is the **deferred measurement principle**:
    a measurement followed by a classically-controlled gate is *provably* equivalent to
    a quantum-controlled gate followed by a measurement at the end. Moving a measurement
    later never changes any outcome distribution, so moving it earlier does not either,
    as long as nothing downstream depends on interference between the branches the
    measurement separated. Here nothing does — those branches are only ever used to
    decide a rotation angle. T15 checks this empirically rather than taking it on
    faith: 500 seeded runs of this function reproduce the coherent circuit's
    distribution to within a total-variation distance of 0.05.

    The phase qubit is borrowed with ``qc.ancilla(1)``, so the simulator verifies at the
    end that it really did come back to |0> and unentangled — the honest bookkeeping for
    a qubit that has been reused t times.
    """
    if t < 1:
        raise QsimError(
            f"cannot estimate a phase to {t} binary digits; ask for at least 1. The "
            "digit count is the precision of the answer, and a phase read to zero "
            "digits is not an approximation of anything."
        )
    circuit = target.circuit

    # digits[0] is the *least* significant binary digit of the answer, because that is
    # the one this algorithm can extract without knowing any of the others: the largest
    # power of U shifts everything but the last digit past the point where it matters.
    digits: list[int] = []

    with circuit.ancilla(1) as scratch:
        phase_qubit = scratch[0]
        for step in range(t):
            H(phase_qubit)

            # 2^(t-1) applications first, then 2^(t-2), and so on: the same set of
            # powers the coherent version spreads across t separate qubits, run one at
            # a time on the one qubit we have.
            controlled_u = unitary.controlled(phase_qubit)
            for _ in range(2 ** (t - 1 - step)):
                controlled_u(target)

            if step:
                # `known` is the digits already measured, read as an integer with the
                # first-measured digit in the ones place. The rotation undoes the
                # contribution those digits make to this qubit's phase — precisely the
                # controlled rotations the inverse QFT would have applied, except that
                # here the controls are classical bits and the angle is just a number.
                known = sum(digit << i for i, digit in enumerate(digits))
                Phase(phase_qubit, theta=-2 * np.pi * known / 2 ** (step + 1))

            # With the lower digits' contribution removed, the only phase left is
            # e^{2*pi*i*(0 or 1/2)} = +-1, and H turns that sign into a definite bit.
            H(phase_qubit)
            digits.append(circuit.measure(phase_qubit))

            # Hand the qubit back to |0> so the next round starts clean. The measurement
            # inside `reset` is deterministic — this qubit has already collapsed — so
            # this is really just "flip it if it read 1", which is how hardware resets.
            circuit.reset(phase_qubit)

    # digits[i] is the digit worth 2^i, so the list read as-is *is* the answer.
    return sum(digit << i for i, digit in enumerate(digits))
