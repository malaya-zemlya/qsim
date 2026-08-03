"""Pictures of quantum states.

**The physical fact this module makes concrete:** an amplitude is a complex number,
so it has a magnitude *and* a phase, and only the magnitude shows up in measurement
statistics. Draw only the magnitudes and you draw a state that looks classical.
Draw the phase too — as color — and the thing that makes quantum mechanics work
becomes visible: two states can have identical bar heights, and therefore identical
measurement statistics right now, while being on their way to completely different
places. The colors are what interference acts on.

``matplotlib`` is imported inside each function rather than at module scope, so that
importing ``qsim`` in a bare interpreter never pulls in a plotting stack.

Every function here returns its ``Figure``. In a notebook, assign it —
``fig = viz.amplitudes(qc)`` — rather than leaving the call as the cell's last
expression, or the figure is drawn twice: once by the inline backend and once as the
cell's result.
"""

from typing import TYPE_CHECKING, Any

import numpy as np

from qsim.inspector import basis_label

if TYPE_CHECKING:
    from qsim.circuit import Circuit, Qubit

# Above this many qubits a bar-per-basis-state chart is unreadable: 2^7 = 128 bars.
_MAX_BAR_QUBITS = 6


def amplitudes(
    qc: Circuit, *, phase_as_hue: bool = True, figsize: tuple[float, float] = (7.0, 3.0)
) -> Any:
    """Bar chart of amplitude magnitudes, colored by phase. Returns the figure.

    Bar height is |amplitude|; bar color is the phase of that amplitude, mapped
    around a color wheel (0 and 2π are the same angle, so the colormap is cyclic).
    Set ``phase_as_hue=False`` for plain bars when the phases are all real and
    positive and the color would be noise.
    """
    import matplotlib.pyplot as plt

    n = qc.n_qubits
    if n > _MAX_BAR_QUBITS:
        raise ValueError(
            f"a bar per basis state means {2**n} bars for {n} qubits, which no one can "
            f"read. Use viz.probabilities(qc, top=...) to see the largest few instead."
        )
    amps = qc.inspect.state_vector()
    mags = np.abs(amps)

    fig, ax = plt.subplots(figsize=figsize)
    if phase_as_hue:
        # np.angle returns the phase in (-pi, pi]; shift to [0, 2pi) and scale to
        # [0, 1) to index the cyclic 'hsv' colormap.
        phases = np.angle(amps) % (2 * np.pi)
        cmap = plt.get_cmap("hsv")
        ax.bar(range(len(amps)), mags, color=cmap(phases / (2 * np.pi)), width=0.72)
        from matplotlib.colors import Normalize

        scalar_map = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 2 * np.pi))
        cbar = fig.colorbar(scalar_map, ax=ax, pad=0.02, fraction=0.045)
        cbar.set_label("phase of amplitude")
        cbar.set_ticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        cbar.set_ticklabels(["0", "π/2", "π", "3π/2", "2π"])
    else:
        ax.bar(range(len(amps)), mags, width=0.72)

    ax.set_xticks(range(len(amps)))
    ax.set_xticklabels(
        [f"|{basis_label(i, n)}⟩" for i in range(2**n)], rotation=90 if n > 4 else 0
    )
    ax.set_ylabel("|amplitude|")
    ax.set_xlabel("basis state")
    ax.set_title(f"amplitudes of {qc.name or 'the state'}")
    return fig


def probabilities(
    qc: Circuit, *, top: int = 32, figsize: tuple[float, float] = (7.0, 3.0)
) -> Any:
    """Bar chart of the ``top`` most likely measurement outcomes. Returns the figure.

    This is the classical shadow of the state: exactly what you could estimate by
    running the circuit many times and counting results. Everything the previous plot
    showed in color is gone from this one.
    """
    import matplotlib.pyplot as plt

    n = qc.n_qubits
    probs = qc.inspect.probabilities()
    order = np.argsort(-probs)[:top]
    labels = [f"|{basis_label(int(i), n)}⟩" for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(order)), probs[order], width=0.72)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=90 if n > 4 else 0)
    ax.set_ylabel("probability")
    ax.set_xlabel("basis state")
    ax.set_title(f"measurement probabilities (top {len(order)})")
    return fig


def bloch(q: Qubit, *, figsize: tuple[float, float] = (4.0, 4.0)) -> Any:
    """Draw one qubit's Bloch vector inside the Bloch sphere. Returns the figure.

        fig = viz.bloch(q)

    The arrow's length tells you how pure the qubit is: length 1 means it has a state
    of its own, and anything shorter means it is entangled with something else or
    otherwise mixed. Length 0 — an arrow at the origin — is a qubit about which
    nothing whatsoever can be known locally, which is what half of a Bell pair looks
    like.

    The circuit is resolved from the handle (``Qubit.circuit``), the same way gates do
    it: ``H(q)`` is never written ``H(qc, q)``, so neither is this.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")
    x, y, z = q.circuit.inspect.bloch_vector(q)
    _draw_bloch(ax, (x, y, z))
    length = float(np.sqrt(x * x + y * y + z * z))
    ax.set_title(f"{q.name} — vector length {length:.2f} (1 = pure)", y=1.02)
    return fig


def _draw_bloch(ax: Any, vector: tuple[float, float, float]) -> None:
    """Draw one Bloch vector inside a wireframe sphere on an existing 3-D axis."""
    # A sparse wireframe sphere: enough lines to read as a ball, few enough to see
    # the vector through it.
    u, v = np.mgrid[0 : 2 * np.pi : 17j, 0 : np.pi : 9j]
    ax.plot_wireframe(
        np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
        color="gray", alpha=0.28, linewidth=0.35,
    )
    for axis in np.eye(3):
        ax.plot(*np.array([-axis, axis]).T, color="gray", alpha=0.35, linewidth=0.6)

    x, y, z = vector
    ax.quiver(0, 0, 0, x, y, z, color="crimson", linewidth=2.4, arrow_length_ratio=0.16)
    for pos, label in [
        ((0, 0, 1.12), "|0⟩"), ((0, 0, -1.32), "|1⟩"),
        ((1.45, 0, -0.1), "|+⟩"), ((-1.6, 0, -0.1), "|−⟩"),
    ]:
        ax.text(pos[0], pos[1], pos[2], label, color="gray", ha="center")

    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-0.85, 0.85)
    ax.set_zlim(-0.85, 0.85)
    ax.view_init(elev=14, azim=-52)


def dephasing_panels(theta: float, *, figsize: tuple[float, float] = (10.5, 3.4)) -> Any:
    """Three views of one dephased qubit at coupling angle ``theta``. Returns the figure.

    Left: the system qubit's Bloch vector, shrinking along x as the environment learns.
    Middle: the two-slit visibility curve cos(θ/2), with a marker at the current θ.
    Right: the reduced density matrix, whose off-diagonal fades while its diagonal
    stays put — decoherence in the one picture that shows both at once.

    Built fresh for each θ rather than by adjusting an existing circuit: a coupling is
    a physical interaction, not a setting, so "θ = 0.4 instead of 0.9" means a different
    experiment, not an edit to this one.
    """
    import matplotlib.pyplot as plt

    from qsim.circuit import Circuit as _Circuit
    from qsim.decoherence import dephasing_coupling
    from qsim.gates import H

    qc = _Circuit()
    q = qc.alloc("q")
    env = qc.environment(1)
    H(q)
    dephasing_coupling(q, env[0], theta=theta)
    rho = qc.inspect.system_density_matrix()

    fig = plt.figure(figsize=figsize)
    ax_bloch = fig.add_subplot(1, 3, 1, projection="3d")
    _draw_bloch(ax_bloch, qc.inspect.bloch_vector(q))
    ax_bloch.set_title(f"Bloch vector\ncoherence {qc.inspect.coherence(q):.3f}", y=1.0)

    ax_vis = fig.add_subplot(1, 3, 2)
    grid = np.linspace(0, np.pi, 200)
    ax_vis.plot(grid, np.cos(grid / 2), color="crimson", linewidth=1.6)
    ax_vis.plot([theta], [np.cos(theta / 2)], "o", color="crimson", markersize=8)
    ax_vis.set_xticks([0, np.pi / 2, np.pi])
    ax_vis.set_xticklabels(["0", "π/2", "π"])
    ax_vis.set_xlabel("θ — how much the environment learns")
    ax_vis.set_ylabel("visibility")
    ax_vis.set_ylim(-0.05, 1.05)
    ax_vis.set_title(f"interference visibility\n{np.cos(theta / 2):.3f}")

    ax_rho = fig.add_subplot(1, 3, 3)
    ax_rho.imshow(np.abs(rho), cmap="magma", vmin=0.0, vmax=0.5)
    for i in range(2):
        for j in range(2):
            # Annotate each cell with its own value; white on the dark end of the
            # colormap, black on the light end, so both stay readable.
            ax_rho.text(
                j, i, f"{abs(rho[i, j]):.3f}", ha="center", va="center",
                color="white" if abs(rho[i, j]) < 0.32 else "black",
            )
    ax_rho.set_xticks([0, 1], ["|0⟩", "|1⟩"])
    ax_rho.set_yticks([0, 1], ["⟨0|", "⟨1|"])
    ax_rho.set_title("reduced density matrix |ρ|\ndiagonal fixed, off-diagonal fading")
    fig.tight_layout()
    return fig


def interact_dephasing() -> None:
    """A slider over the coupling angle θ, redrawing :func:`dephasing_panels` live.

    Drag it and watch the three panels move together: the Bloch vector retracts toward
    the origin, the marker slides down the visibility curve, and the off-diagonal of ρ
    fades while the diagonal does not budge. Three descriptions of one thing.
    """
    try:
        import ipywidgets
    except ImportError as exc:
        raise ImportError(
            "interact_dephasing() needs ipywidgets, a development dependency of qsim "
            "rather than a runtime one — the library itself only requires numpy and "
            "matplotlib. Run `uv sync` to install it, then restart the kernel."
        ) from exc

    import matplotlib.pyplot as plt
    from IPython.display import display

    def show(theta: float) -> None:
        # Draw and show explicitly, returning None. Handing the Figure back instead
        # would render it twice — once by the inline backend and once as the callback's
        # displayed result, the same trap as a bare viz call.
        dephasing_panels(theta)
        plt.show()

    slider = ipywidgets.FloatSlider(
        min=0.0, max=np.pi, step=np.pi / 60, value=0.0, description="θ",
    )
    # ``interactive_output`` rather than the more familiar ``ipywidgets.interact``.
    # ``interact`` deadlocks a headless kernel perhaps half the time — measured here at
    # 2 runs in 4 with a trivial callback and no plotting at all — which would make
    # ``jupyter execute`` on this notebook hang at random. It also inspects the
    # callback's signature and would try to build a second slider out of
    # ``dephasing_panels``' figsize tuple. This form does neither.
    panels = ipywidgets.interactive_output(show, {"theta": slider})
    display(slider, panels)


def circuit_html(qc: Circuit, max_terms: int = 8) -> str:
    """The HTML Jupyter shows for a ``Circuit``: the largest amplitudes as a bar table.

    Deliberately plain: no JavaScript, no images, no external assets, and text in
    ``currentColor`` so it reads correctly in both light and dark notebook themes.
    """
    n = qc.n_qubits
    flat = qc.inspect.state_vector()
    significant = [i for i in range(len(flat)) if abs(flat[i]) > 5e-4]
    significant.sort(key=lambda i: -abs(flat[i]))
    shown = significant[:max_terms]

    rows: list[str] = []
    for i in shown:
        amp = complex(flat[i])
        mag = abs(amp)
        # Hue is the phase in degrees, the same mapping viz.amplitudes uses.
        hue = np.degrees(np.angle(amp)) % 360.0
        sign = "+" if amp.imag >= 0 else "−"
        rows.append(
            f'<tr><td style="padding:.15em .8em .15em 0;white-space:nowrap">'
            f"|{basis_label(i, n)}⟩</td>"
            f'<td style="width:60%;padding:.15em 0">'
            f'<div style="height:.62em;width:{mag * 100:.1f}%;'
            f'background:hsl({hue:.0f} 68% 52%);border-radius:1px"></div></td>'
            f'<td style="padding:.15em 0 .15em .9em;white-space:nowrap;'
            f'font-variant-numeric:tabular-nums;opacity:.78">'
            f"{amp.real:+.3f} {sign} {abs(amp.imag):.3f}i</td></tr>"
        )

    hidden = len(significant) - len(shown)
    more = (
        f'<div style="opacity:.6;padding-top:.5em">… {hidden} more basis '
        f"state{'s' if hidden > 1 else ''} not shown</div>"
        if hidden
        else ""
    )
    label = f"‘{qc.name}’ " if qc.name else ""
    return (
        '<div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;'
        'font-size:.86rem;line-height:1.5;color:currentColor;max-width:34rem">'
        f'<div style="padding-bottom:.5em"><strong>Circuit</strong> {label}· {n} qubits '
        f"· {len(qc.history)} ops · "
        '<span style="opacity:.7">bar = |amplitude|, color = phase</span></div>'
        f'<table style="width:100%;border-collapse:collapse">{"".join(rows)}</table>'
        f"{more}</div>"
    )
