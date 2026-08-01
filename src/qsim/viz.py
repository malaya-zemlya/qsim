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


def bloch(qc: Circuit, q: Qubit, *, figsize: tuple[float, float] = (4.0, 4.0)) -> Any:
    """Draw one qubit's Bloch vector inside the Bloch sphere. Returns the figure.

    The arrow's length tells you how pure the qubit is: length 1 means it has a state
    of its own, and anything shorter means it is entangled with something else or
    otherwise mixed. Length 0 — an arrow at the origin — is a qubit about which
    nothing whatsoever can be known locally, which is what half of a Bell pair looks
    like.
    """
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")

    # A sparse wireframe sphere: enough lines to read as a ball, few enough to see
    # the vector through it.
    u, v = np.mgrid[0 : 2 * np.pi : 17j, 0 : np.pi : 9j]
    ax.plot_wireframe(
        np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
        color="gray", alpha=0.28, linewidth=0.35,
    )
    for axis in np.eye(3):
        ax.plot(*np.array([-axis, axis]).T, color="gray", alpha=0.35, linewidth=0.6)

    x, y, z = qc.inspect.bloch_vector(q)
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
    length = float(np.sqrt(x * x + y * y + z * z))
    ax.set_title(f"{q.name} — vector length {length:.2f} (1 = pure)", y=1.02)
    return fig


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
