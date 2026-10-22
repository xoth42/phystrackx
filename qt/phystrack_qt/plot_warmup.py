"""
Gets the plots ready before the first one is asked for. The first plot used to wait about 2.5 s:
1.9 s importing matplotlib and scipy, and half a second for matplotlib's first font lookups and its
first equation (mathtext) layout. Here a background thread does all of that a little after the main
window opens: the imports, one tiny fit, and one off-screen drawing with the plot's fonts and an
equation, so the caches are filled.

Only thread-safe work is done here: plain imports and an off-screen (Agg) figure, never a Qt widget.
The Qt side of matplotlib (backend_qtagg) is imported on the UI thread when a plot opens; that part
is quick.
"""

import threading

_started = False


def start() -> None:
    """Starts the warm-up once (later calls do nothing)."""
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=_warm, name="plot warm-up", daemon=True).start()


def _warm() -> None:
    try:
        import numpy as np
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        from .equations import equation_png
        from .parent_imports import import_light
        fitting = import_light("core.fit")
        import matplotlib.widgets  # noqa: F401  (the fit panel's slider and menu)

        t = np.arange(120) / 30
        fitting.fit_model(t, np.cos(3 * t) * np.exp(-0.2 * t), fitting.DEFAULT_MODEL)

        figure = Figure(figsize=(4, 3))
        FigureCanvasAgg(figure)
        ax = figure.add_subplot(111)
        ax.plot(t, t, ".")
        ax.set_title("[O-1] warm-up")
        ax.set_xlabel("$T(s)$")
        ax.set_ylabel("$r$ (cm)")
        figure.text(0, 0, "A     = 1.0 ± 0.1 cm", family="monospace", fontsize=8)
        figure.canvas.draw()
        # The fit table's equations, as pictures (the table only loads them)
        for model in fitting.MODELS.values():
            for line in model.equation:
                equation_png(line)
        equation_png(fitting.EQUATION_NOTE)
    except Exception:   # Only a head start: the plot does all of this itself if it must
        pass
