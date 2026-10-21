"""Damped oscillator fit, period checker, and the range slider on the plots (no window needed).

Synthetic data with known parameters: the fit must recover them, the peak spacing must agree
with the fit period, and a range that only covers part of the data must use only that part.
"""

import math

import numpy as np

from tests import harness as h
h.setup_env()

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # The plots are drawn to nothing: no windows open

from gui.components.plot.fit import (best_period, damped, fit_damped,  # noqa: E402
                                     peak_periods)

A, GAMMA, OMEGA, PHI, C = 80.0, 0.15, 1.2, 0.7, 120.0
PERIOD = 2 * math.pi / OMEGA


def _series(noise=0.0, n=900, dt=1 / 30):
    t = np.arange(n) * dt
    rng = np.random.default_rng(1)
    y = damped(t, A, GAMMA, OMEGA, PHI, C) + rng.normal(0, noise, n)
    return t, y


def test_fit_recovers_the_parameters():
    t, y = _series(noise=0.5)
    fit = fit_damped(t, y)
    h.check("fit converges", fit.ok, fit.message)
    h.check("fit R² is high", fit.r2 > 0.99, f"R² {fit.r2:.4f}")
    h.check("omega is recovered within 1%", abs(fit.params["omega"] - OMEGA) / OMEGA < 0.01,
            f"{fit.params['omega']:.4f}")
    h.check("gamma is recovered within 5%", abs(fit.params["gamma"] - GAMMA) / GAMMA < 0.05,
            f"{fit.params['gamma']:.4f}")
    h.check("offset C is recovered", abs(fit.params["C"] - C) < 0.5, f"{fit.params['C']:.3f}")
    h.check("period from the fit", abs(fit.period - PERIOD) < 0.02 * PERIOD,
            f"{fit.period:.4f} vs {PERIOD:.4f}")
    h.check("period has an error estimate", math.isfinite(fit.period_err) and fit.period_err > 0)
    h.check("decay time is 1/gamma", abs(fit.tau - 1 / GAMMA) < 0.5, f"{fit.tau:.3f}")
    h.check("amplitude is reported positive", fit.params["A"] > 0, f"{fit.params['A']:.3f}")


def test_peak_spacing_agrees_with_the_fit():
    t, y = _series(noise=0.5)
    peaks = peak_periods(t, y)
    h.check("peaks are found", peaks.n_maxima >= 2 and peaks.n_minima >= 2,
            f"{peaks.n_maxima} max, {peaks.n_minima} min")
    h.check("peak period matches the true period", abs(peaks.period - PERIOD) < 0.05 * PERIOD,
            f"{peaks.period:.4f} vs {PERIOD:.4f}")


def test_best_period_prefers_a_good_fit():
    t, y = _series(noise=0.5)
    fit, peaks = fit_damped(t, y), peak_periods(t, y)
    period, _, source = best_period(fit, peaks)
    h.check("good fit is the reported period", source == "fit" and abs(period - fit.period) < 1e-12,
            str(source))


def test_best_period_falls_back_to_peaks_for_a_poor_fit():
    rng = np.random.default_rng(2)
    t = np.arange(600) / 30
    y = damped(t, A, GAMMA, OMEGA, PHI, C) + rng.normal(0, 60, t.size)  # Mostly noise
    fit, peaks = fit_damped(t, y), peak_periods(t, y)
    _, _, source = best_period(fit, peaks)
    h.check("poor fit does not claim the period", source in ("peaks", None) or fit.r2 >= 0.9,
            f"source {source}, R² {fit.r2:.3f}")


def test_range_uses_only_the_selected_part():
    """Two parts with different frequencies: a range on each must find its own frequency."""
    dt = 1 / 30
    t1 = np.arange(0, 20, dt)
    t2 = np.arange(20, 40, dt)
    y1 = damped(t1, 50, 0.01, 1.0, 0.0, 100)
    y2 = damped(t2, 50, 0.01, 2.0, 0.0, 100)
    t = np.concatenate([t1, t2])
    y = np.concatenate([y1, y2])
    early = fit_damped(t[t < 20], y[t < 20])
    late = fit_damped(t[t >= 20], y[t >= 20])
    h.check("early range finds its own omega", abs(early.params["omega"] - 1.0) < 0.02,
            f"{early.params['omega']:.4f}")
    h.check("late range finds its own omega", abs(late.params["omega"] - 2.0) < 0.02,
            f"{late.params['omega']:.4f}")


def test_warm_start_gives_the_same_fit():
    """A fit started from a nearby range's result must match a cold fit (it is only faster)."""
    t, y = _series(noise=0.5)
    first = fit_damped(t[t <= 25], y[t <= 25])
    m = (t >= 1.5) & (t <= 24)
    warm = fit_damped(t[m], y[m], warm=first)
    cold = fit_damped(t[m], y[m])
    h.check("warm and cold fits agree on the period",
            abs(warm.period - cold.period) < 1e-3 * cold.period,
            f"{warm.period:.5f} vs {cold.period:.5f}")


def test_distance_r_is_csv_column_and_plot():
    """r = sqrt(x^2 + y^2) from the axes origin: a CSV column and a plot with the fit (headless)."""
    import matplotlib.pyplot as plt
    from gui.components.plot.plot import Plot
    from gui.components.plot.save import Save
    from gui.components.plot.savetype import SaveType

    t = np.arange(900) / 30
    r_true = damped(t, A, GAMMA, OMEGA, PHI, C)  # Distance oscillates about a centre C
    theta = 0.4
    x, y = r_true * np.cos(theta), r_true * np.sin(theta)

    class _Data:
        datacount = 1
        maxcount = t.size
        ocrcount = 0
        timestamps = t
        processed_points = [np.column_stack([x, y])]

    save = Save.__new__(Save)
    save.datamanager = _Data()
    header = save.prepheader([SaveType.XY.name, SaveType.HEADER.name])
    h.check("CSV header has r next to x and y", header == ["x1", "y1", "r1"], str(header))

    plot = Plot.__new__(Plot)
    plot.datamanager = _Data()
    plot._widgets = []
    plot.plotr()
    ax = plt.gcf().axes[0]
    texts = [tx.get_text() for tx in ax.texts]
    h.check("r plot shows the fitted period", any("T fit" in tx and f"{PERIOD:.2f}"[:3] in tx
                                                  for tx in texts), str(texts[:1]))
    plt.close("all")


def test_linear_drift_model_recovers_the_slope():
    """Oscillation settling onto a sloped line: the drift model finds the slope and fits better."""
    from gui.components.plot.fit import fit_model
    t, y = _series(noise=0.5)
    slope = 0.4
    y = y + slope * t
    lin = fit_model(t, y, "damped_linear")
    plain = fit_model(t, y, "damped")
    h.check("drift model converges", lin.ok, lin.message)
    h.check("slope is recovered", abs(lin.params["m"] - slope) < 0.02, f"{lin.params['m']:.4f}")
    h.check("omega is recovered with the drift", abs(lin.params["omega"] - OMEGA) / OMEGA < 0.02,
            f"{lin.params['omega']:.4f}")
    h.check("drift model fits better than the plain one", lin.r2 > plain.r2,
            f"{lin.r2:.4f} vs {plain.r2:.4f}")


def test_plot_fit_type_switch():
    """The fit-type buttons on the plot switch between the plain and the drift model."""
    import matplotlib.pyplot as plt
    from gui.components.plot.plot import Plot
    from gui.components.plot.fit import MODELS

    t, y = _series(noise=0.5, n=600)
    y = y + 0.3 * t

    class _Data:
        pass

    data = _Data()
    data.processed_points = [np.column_stack([np.zeros_like(y), y])]
    data.timestamps = t
    plot = Plot.__new__(Plot)
    plot.datamanager = data
    plot._widgets = []
    plot.ploty()
    radio = plot._widgets[-1]
    ax = plt.gcf().axes[0]
    text = lambda: "\n".join(tx.get_text() for tx in ax.texts)  # noqa: E731
    h.check("drift model is the default", "m     =" in text(), text()[:80])
    keys = list(MODELS)
    radio.set_active(keys.index("damped"))
    h.check("plain model drops the drift line", "m     =" not in text(), text()[:80])
    plt.close("all")


def test_too_few_points_is_refused():
    fit = fit_damped([0, 1, 2], [1, 2, 3])
    h.check("fit refuses a tiny range", not fit.ok and "at least" in fit.message, fit.message)


def test_plot_slider_updates_the_fit():
    """The plot's range slider recomputes the fit for the chosen range (headless)."""
    import matplotlib.pyplot as plt
    from gui.components.plot.plot import Plot

    class _Data:
        pass

    t, y = _series(noise=0.5, n=600)
    data = _Data()
    data.processed_points = [np.column_stack([np.zeros_like(y), y])]
    data.timestamps = t
    plot = Plot.__new__(Plot)  # Skip the GUI parent: the test only drives the plot code
    plot.datamanager = data
    plot._widgets = []
    plot.checkbox = None
    plot.ploty()
    from matplotlib.widgets import RangeSlider
    slider = [w for w in plot._widgets if isinstance(w, RangeSlider)][-1]
    slider.set_val((float(t[0]), float(t[-1]) / 2))  # Only the first half
    ax = plt.gcf().axes[0]
    texts = [tx.get_text() for tx in ax.texts]
    h.check("slider shows a summary", any("T fit" in tx for tx in texts))
    h.check("summary reports the fitted period", any(f"{PERIOD:.2f}"[:3] in tx for tx in texts),
            str(texts[:1]))
    plt.close("all")


def test_quadratic_drag_recovers_the_parameters():
    """Data from the quadratic-drag equation: the drag model must recover omega and the drag k."""
    import time
    from gui.components.plot.fit import drag, fit_model
    t = np.arange(900) / 30
    y = drag(t, 60.0, 0.0, 1.2, 0.02, 120.0)
    y = y + np.random.default_rng(3).normal(0, 0.3, t.size)
    start = time.time()
    fit = fit_model(t, y, "drag")
    took = time.time() - start
    h.check("drag fit converges", fit.ok, fit.message)
    h.check("drag fit R² is high", fit.r2 > 0.98, f"R² {fit.r2:.4f}")
    h.check("drag omega is recovered within 2%",
            abs(fit.params["omega"] - 1.2) / 1.2 < 0.02, f"{fit.params['omega']:.4f}")
    h.check("drag strength k is recovered within 25%",
            abs(fit.params["k"] - 0.02) / 0.02 < 0.25, f"{fit.params['k']:.4f}")
    h.check("drag fit is quick enough for the slider", took < 5.0, f"{took:.2f} s")


def test_quadratic_drag_with_drift_recovers_the_slope():
    from gui.components.plot.fit import drag_linear, fit_model
    t = np.arange(900) / 30
    y = drag_linear(t, 60.0, 0.0, 1.2, 0.02, 120.0, 0.4)
    y = y + np.random.default_rng(4).normal(0, 0.3, t.size)
    fit = fit_model(t, y, "drag_linear")
    h.check("drag + drift converges", fit.ok, fit.message)
    h.check("drag + drift slope is recovered", abs(fit.params["m"] - 0.4) < 0.02,
            f"{fit.params['m']:.4f}")
    h.check("drag + drift omega is recovered within 3%",
            abs(fit.params["omega"] - 1.2) / 1.2 < 0.03, f"{fit.params['omega']:.4f}")


def test_plot_offers_the_drag_models():
    import matplotlib.pyplot as plt
    from gui.components.plot.plot import Plot
    from gui.components.plot.fit import MODELS

    class _Data:
        pass

    t = np.arange(600) / 30
    y = damped(t, A, GAMMA, OMEGA, PHI, C)
    data = _Data()
    data.processed_points = [np.column_stack([np.zeros_like(y), y])]
    data.timestamps = t
    data.unit = "cm"
    plot = Plot.__new__(Plot)
    plot.datamanager = data
    plot._widgets = []
    plot.ploty()
    radio = plot._widgets[-1]
    ax = plt.gcf().axes[0]
    text = lambda: "\n".join(tx.get_text() for tx in ax.texts)  # noqa: E731
    h.check("the fit menu lists 4 models", len(MODELS) == 4, str(list(MODELS)))
    radio.set_active(list(MODELS).index("drag"))
    h.check("the drag model shows its drag strength k", "k     =" in text(), text()[:120])
    h.check("the drag model has no gamma line", "gamma =" not in text())
    plt.close("all")


def test_initial_drop_is_high_to_first_low():
    """A motion that starts high and falls: the drop runs from the first high to the first low."""
    from gui.components.plot.fit import initial_drop
    t = np.arange(0, 6, 1 / 30)
    y = 50 + 40 * np.cos(2 * math.pi * 0.5 * t)  # Starts at 90 (high), first low at t = 1 s, value 10
    drop = initial_drop(t, y)
    h.check("initial drop is found", drop.ok, drop.message)
    h.check("the start is the first high", abs(drop.t_high) < 0.05 and abs(drop.y_high - 90) < 0.5,
            f"t {drop.t_high:.3f}, y {drop.y_high:.2f}")
    h.check("the low is the first turnaround", abs(drop.t_low - 1.0) < 0.05 and
            abs(drop.y_low - 10) < 0.5, f"t {drop.t_low:.3f}, y {drop.y_low:.2f}")
    h.check("drop and duration are right", abs(drop.drop - 80) < 1 and abs(drop.duration - 1.0) < 0.05,
            f"drop {drop.drop:.2f}, duration {drop.duration:.3f}")


def test_plot_shows_drop_markers_and_a_legend():
    import matplotlib.pyplot as plt
    from gui.components.plot.plot import Plot

    class _Data:
        pass

    t = np.arange(600) / 30
    y = 50 + 40 * np.cos(2 * math.pi * 0.5 * t)
    data = _Data()
    data.processed_points = [np.column_stack([np.zeros_like(y), y])]
    data.timestamps = t
    data.unit = "cm"
    plot = Plot.__new__(Plot)
    plot.datamanager = data
    plot._widgets = []
    plot.ploty()
    ax = plt.gcf().axes[0]
    texts = "\n".join(tx.get_text() for tx in ax.texts)
    h.check("the plot reports the initial drop", "Initial drop =" in texts, texts[-120:])
    legend = ax.get_legend()
    labels = [tx.get_text() for tx in legend.get_texts()] if legend else []
    h.check("the plot has a legend with the fit and the drop", "fit" in labels and
            "initial drop" in labels, str(labels))
    h.check("the legend is placed automatically", legend is not None and legend._loc == 0,
            f"loc {legend._loc if legend else None}")
    plt.close("all")


if __name__ == "__main__":
    h.run_tests([test_fit_recovers_the_parameters, test_linear_drift_model_recovers_the_slope, test_plot_fit_type_switch, test_distance_r_is_csv_column_and_plot, test_warm_start_gives_the_same_fit, test_peak_spacing_agrees_with_the_fit,
                 test_best_period_prefers_a_good_fit, test_best_period_falls_back_to_peaks_for_a_poor_fit,
                 test_range_uses_only_the_selected_part, test_too_few_points_is_refused,
                 test_plot_slider_updates_the_fit,
                 test_quadratic_drag_recovers_the_parameters, test_quadratic_drag_with_drift_recovers_the_slope,
                 test_plot_offers_the_drag_models, test_initial_drop_is_high_to_first_low,
                 test_plot_shows_drop_markers_and_a_legend])
