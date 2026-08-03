"""The Quantum Fourier Transform — reading frequencies off a quantum state.

**The physical fact this module makes concrete:** the Fourier transform, the single
most-used linear map in all of signal processing, can be applied to the 2^n amplitudes
of an n-qubit register by about n²/2 elementary gates. Nothing else in quantum
computing is anywhere near this dramatic a saving — and nothing else illustrates as
sharply that a saving in *doing* is not a saving in *knowing*.

What a Fourier transform is, in one paragraph
---------------------------------------------
Give the discrete Fourier transform (DFT) a list of N numbers and it hands back N
numbers describing the same list as a sum of pure waves: how much of the list wiggles
once across its length, how much wiggles twice, and so on up to N-1 times. The k-th
output is

    y_k = (1/sqrt N) * sum_j x_j * exp(2*pi*i*j*k/N)

— every input multiplied by a complex number of magnitude 1 that spins around the unit
circle at rate k, then added up. If the input really does repeat with period r, the
outputs are near zero everywhere except at multiples of N/r, where the spinning stays
in step with the repetition and the terms reinforce instead of cancelling. That is the
whole idea: **the DFT converts a period into a position.** ``np.fft`` computes it (with
different conventions — T11 in ``tests/test_acceptance_t11_t15.py`` reconciles them
line by line).

The QFT is the same map, applied to amplitudes
-----------------------------------------------
An n-qubit register holds 2^n amplitudes, one per basis state. The QFT is the DFT of
*that* list of numbers:

    QFT |j> = (1/sqrt(2^n)) * sum_{k=0}^{2^n - 1} exp(2*pi*i*j*k/2^n) |k>

and, being linear, it acts on a superposition by acting on each |j> in it. So if the
amplitudes of a register happen to repeat with period r, the QFT concentrates them onto
basis states near multiples of 2^n/r. Shor's algorithm is exactly this sentence used
twice: build a register whose amplitudes repeat with the period you are hunting, then
read the period off as a position.

The miracle, and the catch
---------------------------
**The miracle.** The classical fast Fourier transform needs about N log N = 2^n * n
arithmetic operations. The circuit below needs n Hadamards and n(n-1)/2
controlled-phase gates — about n²/2 gates for 2^n amplitudes. For n = 20 that is 210
gates against twenty million multiply-adds.

**The catch, which matters just as much.** The 2^n output numbers are *amplitudes*, and
you cannot read amplitudes. Measuring gives you one basis state, drawn with probability
|amplitude|². So the QFT does not let you compute a Fourier transform faster — you
cannot get the answer out. It is only useful inside an algorithm arranged so that
**one frequency dominates**, because then the single basis state you are allowed to see
is, with high probability, the one you wanted. Every use of the QFT in this library
(phase estimation, then Shor's) is a construction with that property. A quantum speedup
is never "the same computation, faster"; it is a computation whose answer happens to
survive being measured.

Binary fractions, the notation the circuit is built on
-------------------------------------------------------
Below, ``0.b1 b2 b3 ...`` means a number written in binary *after* the point:

    0.b1 b2 b3 ...  =  b1/2 + b2/4 + b3/8 + ...

so ``0.011`` is 0 + 1/4 + 1/8 = 3/8, exactly as ``0.375`` is 3/10 + 7/100 + 5/1000. The
QFT circuit works because each output qubit ends up carrying one such binary fraction of
the input in its phase, and a controlled-phase gate is precisely the instrument for
adding one more binary digit to one.

Two precision facts worth knowing early (design doc §8.1)
----------------------------------------------------------
**How many qubits the phase register needs.** For Shor's algorithm on an n-bit number,
the phase register wants

    t = 2n + 1 + ceil(log2(2 + 1/(2*epsilon)))

qubits to succeed with probability at least 1 - epsilon. *In plain terms:* the algorithm
has to distinguish fractions s/r with r < N from one another, and two such fractions can
be as close together as 1/N². Telling apart numbers that differ by 1/N² needs about
2*log2(N) = 2n bits of resolution — the +1 and the log term buy the margin that turns
"resolvable in principle" into "resolved with probability 1 - epsilon".

**How small the rotations need to be.** The circuit's smallest rotation is by an angle
2*pi/2^n, which for n = 20 is a millionth of a turn: hardware cannot do that, and it
turns out not to need to. Dropping every rotation finer than ``approx`` binary places
leaves an error that scales as t² * 2^{-approx}, so only ``approx = O(log(t/epsilon))``
distinct angles are needed at all. *In plain terms:* the fine rotations each move a
vanishing amount of amplitude, and the errors are spread out rather than aligned, so
throwing away the smallest ones costs almost nothing — T13 measures exactly how little.
``qft(reg, approx=m)`` builds that truncated circuit, and it is a genuinely useful
approximation, not a toy.
"""

import numpy as np

from qsim.circuit import Register
from qsim.combinators import gate
from qsim.gates import SWAP, CPhase, H


@gate
def qft(reg: Register, *, swap: bool = True, approx: int | None = None) -> None:
    """Apply the Quantum Fourier Transform to ``reg``. See the module docstring.

        qft(reg)                    # the textbook transform
        qft(reg, swap=False)        # ... without fixing the bit reversal
        qft(reg, approx=6)          # ... dropping rotations finer than 1/2^6

    ``reg[0]`` is the most significant bit going in and coming out, matching the
    convention everywhere else in qsim.

    How the circuit works
    ---------------------
    Write the input basis state as |j1 j2 ... jn> with j1 the most significant bit. The
    transform's output factorizes — this is the fact that makes a circuit possible at
    all, and it is worth checking on paper once:

        QFT |j> = (1/sqrt(2^n)) * (|0> + e^{2*pi*i*0.jn} |1>)
                                * (|0> + e^{2*pi*i*0.j_{n-1} jn} |1>)
                                * ...
                                * (|0> + e^{2*pi*i*0.j1 j2 ... jn} |1>)

    Every output qubit is *unentangled* from the others, and each carries one binary
    fraction of the input number in the phase of its |1> component. (It follows that the
    QFT of a basis state is a product state — the transform creates no entanglement at
    all when fed a definite number. It creates plenty when fed a superposition.)

    The circuit builds those factors one at a time:

    - ``H(reg[j])`` turns |j_{j+1}> into (|0> + e^{2*pi*i*0.j_{j+1}} |1>)/sqrt 2, since
      H sends |b> to (|0> + (-1)^b |1>)/sqrt 2 and (-1)^b is exactly e^{2*pi*i*(b/2)}.
      One Hadamard writes the first binary digit into a phase.
    - Each following ``CPhase(reg[k], reg[j], theta=2*pi/2^m)`` adds the next digit,
      one binary place further down. It fires only when ``reg[k]`` is |1>, and when it
      fires it multiplies ``reg[j]``'s |1> amplitude by e^{2*pi*i/2^m} — which is
      exactly "append the bit ``j_k`` at binary place m of the fraction".

    Two details that trip everyone up
    ----------------------------------
    **Bit reversal.** The qubit processed *first* accumulates the *longest* fraction, so
    it ends up holding what should be the *last* output factor. The circuit therefore
    produces the right transform in reversed qubit order, and a network of SWAPs at the
    end puts it right. Pass ``swap=False`` to see the reversal for yourself; the state
    is then the true QFT read with ``reg`` reversed, and T11 undoes it by hand with a
    transpose to show that reversing the qubit order *is* what a transpose over reversed
    axes does.

    **Truncation.** ``approx=m`` skips every controlled rotation of order finer than
    1/2^m — those are the ones whose phases are too small to matter (see the module
    docstring's second precision fact). ``approx=1`` leaves only the Hadamards, which is
    a real transform of its own: it is the tensor product of n independent H gates.
    ``approx=None`` (the default) keeps every rotation and is exact.
    """
    n = len(reg)
    for j in range(n):
        # Write the leading digit of this qubit's fraction into its phase.
        H(reg[j])
        for k in range(j + 1, n):
            # reg[k] carries the digit m-1 binary places below reg[j]'s own, so the
            # phase it contributes is a rotation by 2*pi / 2^m: a half turn from the
            # neighbour, a quarter turn from the next one along, and so on.
            m = k - j + 1
            if approx is not None and m > approx:
                # Too fine to bother with. Skipping it is the approximate QFT.
                continue
            CPhase(reg[k], reg[j], theta=2 * np.pi / 2**m)
    if swap:
        # Undo the bit reversal the construction above produces: the first qubit
        # processed holds the last output factor, the second holds the second-to-last,
        # and so on, so reversing the register puts every factor where it belongs.
        for i in range(n // 2):
            SWAP(reg[i], reg[n - 1 - i])


#: ``qft`` run backwards, as a first-class ``Block`` — the machinery behind :func:`iqft`.
#:
#: Since Phase 2.75 ``Block.adjoint()`` returns a real ``Block``: the recorded body is
#: reversed and every op replaced by its inverse, and the classical parameters (``swap``,
#: ``approx``) come along because they were consumed at record time and are already
#: numbers inside the recorded ops. So the inverse transform needs no second circuit
#: written out by hand — which is the point of having a closed algebra, and worth
#: demonstrating in production code rather than only in a test.
_QFT_DAGGER = qft.adjoint()


@gate
def iqft(reg: Register, *, swap: bool = True, approx: int | None = None) -> None:
    """Apply the **inverse** Quantum Fourier Transform to ``reg``.

        iqft(reg)

    Where :func:`qft` turns a periodic pattern of amplitudes into a peak at a position,
    this turns a phase gradient back into a number — which is what phase estimation
    (``phase_estimation.py``) and Shor's algorithm actually use. Reading a frequency out
    of a register is an *inverse* transform, so this is the one you will meet more often.

    It is implemented as ``qft.adjoint()``: the same recorded body, replayed backwards
    with every gate inverted. That is not a shortcut but the honest construction — the
    inverse of a unitary *is* its gates run in reverse, and writing the circuit out a
    second time by hand would only create a second thing to keep in sync. It stays a
    named public function because ``iqft(out)`` reads better at the call site than
    ``qft.adjoint()(out)``, and because that is the name the literature uses.
    """
    _QFT_DAGGER(reg, swap=swap, approx=approx)
