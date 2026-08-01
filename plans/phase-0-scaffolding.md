# Phase 0 — Project Scaffolding

**Read first:** `plans/master-plan.md` (Conventions), `qsim-design.md` §0.5 and §7.

**Goal:** turn the stock `uv init` output into the project skeleton every later phase builds on. No quantum code in this phase.

## Current state (verify before starting)

- `pyproject.toml` is stock `uv init` output: `requires-python = ">=3.13"`, no dependencies.
- `.python-version` says `3.13`; Python 3.14.3 is installed at `/opt/homebrew/bin/python3.14`.
- `main.py` is the uv hello-world stub — delete it.
- Git repo exists with no commits yet (or only an initial one); `.gitignore` exists.

## Tasks

### 1. pyproject.toml

Replace with:

```toml
[project]
name = "qsim"
version = "0.1.0"
description = "A NumPy state-vector quantum simulator built for learning QM and quantum computing"
readme = "README.md"
requires-python = ">=3.14"
dependencies = [
    "numpy>=2.3",
    "matplotlib>=3.10",
]

[dependency-groups]
dev = [
    "pytest>=8",
    "pytest-cov",
    "ruff",
    "pyright",
    "jupyterlab",
    "ipympl",
    "ipywidgets",
]

[build-system]
requires = ["uv_build"]
build-backend = "uv_build"

[tool.pytest.ini_options]
addopts = "-v --cov=qsim --cov-report=term-missing"
testpaths = ["tests"]

[tool.coverage.run]
branch = true

[tool.coverage.report]
fail_under = 100
show_missing = true
exclude_also = ["if TYPE_CHECKING:"]

[tool.pyright]
include = ["src", "tests"]
typeCheckingMode = "standard"

[tool.ruff]
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

Notes:
- Use the `src/` layout (`src/qsim/`) — it is what `uv_build` expects and prevents accidentally importing the source tree instead of the installed package. All design-doc paths like `qsim/state.py` mean `src/qsim/state.py`.
- `typeCheckingMode = "standard"` (not `strict`) to start; Phase 1 may tighten it if it's clean. Do not start at `strict` — numpy's stubs generate noise that would drown real errors.
- Update `.python-version` to `3.14`.

### 2. Package skeleton

Create empty-but-importable modules matching design doc §7 (docstring-only bodies; one-line placeholder docstrings are fine at this stage):

```
src/qsim/__init__.py
src/qsim/errors.py         # real content now — see task 3
tests/conftest.py          # empty for now
tests/test_smoke.py        # see task 4
notebooks/                 # empty directory (add .gitkeep)
```

Do **not** pre-create `state.py`, `circuit.py`, etc. — Phase 1 creates files as it builds them, so the repo never contains dead stubs.

Delete `main.py`. Write a short `README.md`: what qsim is (two sentences, lifted from design doc §0), how to set up (`uv sync`), test (`uv run pytest -v`), and open notebooks (`uv run jupyter lab`).

### 3. errors.py (the one real module in this phase)

The three exception types exist from day one so every later phase can import them:

```python
class QsimError(Exception):
    """Base class for all qsim errors."""

class NoCloningError(QsimError): ...
class DeadQubitError(QsimError): ...
class DirtyAncillaError(QsimError): ...
```

Give each a docstring stating, in one sentence each, (a) when it is raised and (b) the physical principle involved. The *full* teaching-quality messages are composed at raise sites in later phases; the docstrings here are short summaries. Module docstring: "Error messages in qsim are teaching surfaces (design doc §12): each exception corresponds to a physical impossibility, not just an API misuse."

### 4. Smoke test

`tests/test_smoke.py`:

```python
import sys

import numpy as np

import qsim
from qsim.errors import DeadQubitError, DirtyAncillaError, NoCloningError


def test_python_version() -> None:
    assert sys.version_info >= (3, 14)

def test_numpy_present() -> None:
    assert np.zeros((2, 2), dtype=np.complex128).dtype == np.complex128

def test_errors_importable() -> None:
    for exc in (NoCloningError, DeadQubitError, DirtyAncillaError):
        assert issubclass(exc, qsim.errors.QsimError)
```

### 5. Verify

```
uv sync
uv run pytest -v          # 3 passing
uv run pyright            # 0 errors
uv run ruff check .       # clean
uv run jupyter --version  # jupyter installs correctly
uv run python -c "import matplotlib; import ipywidgets"
```

## Definition of done

- All five verify commands pass.
- `main.py` gone; `src/` layout in place; `README.md` written.
- Committed as `Phase 0: project scaffolding`.

## Interface decisions to review with the owner (before building)

None — this phase has no public API. Proceed directly after the owner has seen this plan.

## As built

Shipped as specified. Two small decisions the plan did not cover:

- **`.gitignore` gained `.coverage`, `.pytest_cache/`, `.ruff_cache/`, `.ipynb_checkpoints/`**
  (and later `.DS_Store`) — the pytest config writes a coverage file on every run.
- **Committed on `master`.** The repo had zero commits, so this is the initial commit and
  there was no default branch to protect. `master` is also the default branch on the GitHub
  remote (`git@github.com:malaya-zemlya/qsim.git`), added after Phase 1.
