"""Shared fixtures.

Read the test suite as a usage guide: every test is named for the behavior it
states, and the acceptance tests in ``test_acceptance_t1_t7.py`` are the
specification from design doc §9.
"""

import matplotlib
import numpy as np
import pytest

# Plots must render without a window in CI. Set before anything imports pyplot —
# qsim.viz imports it lazily inside each function, so this is early enough.
matplotlib.use("Agg")

from qsim import Circuit  # noqa: E402
from qsim.circuit import Qubit  # noqa: E402
from qsim.gates import CNOT, H  # noqa: E402

SEED = 1234


@pytest.fixture
def seed() -> int:
    """A fixed seed, so every measurement in the suite is reproducible."""
    return SEED


@pytest.fixture
def qc() -> Circuit:
    """An empty, seeded circuit."""
    return Circuit(name="test", seed=SEED)


@pytest.fixture
def bell_pair() -> tuple[Circuit, Qubit, Qubit]:
    """A circuit holding (|00> + |11>)/sqrt(2), plus handles to both qubits."""
    circuit = Circuit(name="bell", seed=SEED)
    a, b = circuit.alloc_many(2)
    H(a)
    CNOT(a, b)
    return circuit, a, b


@pytest.fixture
def random_state():
    """Factory for a normalized random complex state tensor of shape (2,)*n."""

    def make(n: int, seed: int = SEED) -> np.ndarray:
        rng = np.random.default_rng(seed)
        psi = rng.normal(size=(2,) * n) + 1j * rng.normal(size=(2,) * n)
        return psi / np.linalg.norm(psi)

    return make
