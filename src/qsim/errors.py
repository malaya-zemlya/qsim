"""Error messages in qsim are teaching surfaces (design doc §12): each exception
corresponds to a physical impossibility, not just an API misuse.

When one of these is raised, the message should leave you knowing something about
quantum mechanics that you did not know before you triggered it. The full,
teaching-quality messages are composed at the places that raise them; the
docstrings here are the one-sentence summaries.
"""


class QsimError(Exception):
    """Base class for all qsim errors, so `except QsimError` catches every one."""


class NoCloningError(QsimError):
    """Raised when you ask for a copy of an unknown quantum state.

    The no-cloning theorem says no physical process can duplicate an arbitrary
    unknown state: copying is not a linear operation, and quantum evolution is
    linear.
    """


class DeadQubitError(QsimError):
    """Raised when you use a qubit that has been measured out or deallocated.

    A qubit handle refers to a live tensor factor of the state; once that factor
    is gone the handle names nothing physical, and the simulator refuses rather
    than silently acting on some other qubit.
    """


class DirtyAncillaError(QsimError):
    """Raised when a scratch (ancilla) qubit is released while still entangled
    with the rest of the state.

    Scratch qubits must be *uncomputed* back to |0⟩ before release: any leftover
    entanglement records which branch of the computation happened, and a recorded
    branch can no longer interfere with the others — which is where a quantum
    algorithm's speedup comes from.
    """
