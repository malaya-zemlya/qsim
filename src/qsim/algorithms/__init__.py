"""Complete quantum algorithms and experiments, built from the gates in ``qsim.gates``.

Everything here differs from the rest of the library in one way worth knowing up front:
these are **self-contained experiments**, not gates. A gate acts on *your* circuit;
a function here builds its own circuit, runs the whole protocol, and hands you the
result. They are things to run and read, not parts to assemble.
"""
