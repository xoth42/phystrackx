"""
fit.py

Fits of a time series with a damped oscillation, and period measurements. No GUI here, so the
maths can be tested on its own.

Models, with tt = t - t[0] (time from the start of the chosen range):

    "damped":        y = C + A * exp(-gamma * tt) * cos(omega * tt + phi)
    "damped_linear": y = C + m * tt + A * exp(-gamma * tt) * cos(omega * tt + phi)
    "drag":          z'' = -omega^2 z - k |z'| z'   (quadratic drag), y = C + z
    "drag_linear":   y = C + m * tt + z, with z solving the same equation

The linear term lets the oscillation settle onto a sloped line rather than a flat one. The drag
models start from z(0) = z0 with speed z'(0) = v0, so the drag strength k is fitted directly. The
period of a drag oscillation depends a little on its amplitude, so the fitted omega is the
small-swing value.

Period: T = 2*pi/omega from the fit, and the spacing of successive peaks and troughs in the data.

Author: Isam Balghari
"""

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

MIN_POINTS = 8
GOOD_FIT_R2 = 0.9  # Above this the fit period is preferred over the peak spacing


def damped(tt, A, gamma, omega, phi, C):
    """Damped oscillation about a constant C. tt is time measured from the start of the range."""
    return C + A * np.exp(-gamma * tt) * np.cos(omega * tt + phi)


def damped_linear(tt, A, gamma, omega, phi, C, m):
    """Damped oscillation about a straight line C + m*tt."""
    return damped(tt, A, gamma, omega, phi, C) + m * tt


def _drag_path(tt, z0: float, v0: float, omega: float, k: float) -> np.ndarray:
    """
    Position of z'' = -omega^2 z - k |z'| z' at the times tt (starting at tt[0] = 0), from
    z(0) = z0 and z'(0) = v0. Each step is one Runge-Kutta (RK4) step over the sample interval.
    """
    def accel(z, v):
        return -omega * omega * z - k * abs(v) * v

    out = np.empty(len(tt))
    z, v = z0, v0
    out[0] = z
    for i in range(1, len(tt)):
        h = tt[i] - tt[i - 1]
        k1z, k1v = v, accel(z, v)
        k2z, k2v = v + h / 2 * k1v, accel(z + h / 2 * k1z, v + h / 2 * k1v)
        k3z, k3v = v + h / 2 * k2v, accel(z + h / 2 * k2z, v + h / 2 * k2v)
        k4z, k4v = v + h * k3v, accel(z + h * k3z, v + h * k3v)
        z += h / 6 * (k1z + 2 * k2z + 2 * k3z + k4z)
        v += h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        out[i] = z
    return out


def drag(tt, z0, v0, omega, k, C):
    """Quadratic-drag oscillation about a constant C."""
    return C + _drag_path(tt, z0, v0, omega, k)


def drag_linear(tt, z0, v0, omega, k, C, m):
    """Quadratic-drag oscillation about a straight line C + m*tt."""
    return drag(tt, z0, v0, omega, k, C) + m * tt


@dataclass(frozen=True)
class FitModel:
    key: str
    label: str
    names: tuple
    func: object


MODELS = {
    "damped": FitModel("damped", "Damped oscillator",
                       ("A", "gamma", "omega", "phi", "C"), damped),
    "damped_linear": FitModel("damped_linear", "Damped oscillator + linear drift",
                              ("A", "gamma", "omega", "phi", "C", "m"), damped_linear),
    "drag": FitModel("drag", "Quadratic drag oscillator",
                     ("z0", "v0", "omega", "k", "C"), drag),
    "drag_linear": FitModel("drag_linear", "Quadratic drag + linear drift",
                            ("z0", "v0", "omega", "k", "C", "m"), drag_linear),
}
DEFAULT_MODEL = "damped_linear"


@dataclass
class FitResult:
    ok: bool
    message: str = ""
    model: str = DEFAULT_MODEL
    params: dict = field(default_factory=dict)   # Named as in the model
    errors: dict = field(default_factory=dict)   # One standard error per parameter
    r2: float = float("nan")
    period: float = float("nan")                 # 2*pi/omega
    period_err: float = float("nan")
    tau: float = float("nan")                    # Decay time 1/gamma
    t0: float = 0.0                              # Start of the range, the origin of tt
    phi_abs: float = 0.0                         # Phase with absolute time (for warm starts)


@dataclass
class PeakPeriods:
    n_maxima: int
    n_minima: int
    period: float        # Mean spacing of peaks and troughs of the same kind, averaged
    period_err: float    # Standard error of that mean
    n_intervals: int


def fit_model(t, y, model: str = DEFAULT_MODEL, warm: "FitResult | None" = None) -> FitResult:
    """
    Fits the chosen model to (t, y). Non-finite values are dropped.

    The frequency starts from the largest peak of the spectrum, and the phase is tried at
    several starting values, so the fit does not stop at a wrong local minimum.

    warm: a previous fit with the same model (for example of a slightly different range). Its
    parameters are tried first, which is much faster when the slider only moves a little.
    """
    spec = MODELS[model]
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(t) & np.isfinite(y)
    t, y = t[keep], y[keep]
    order = np.argsort(t)
    t, y = t[order], y[order]
    if len(t) < MIN_POINTS:
        return FitResult(ok=False, model=model, message=f"Select at least {MIN_POINTS} points")

    t0 = float(t[0])
    tt = t - t0
    dt = float(np.median(np.diff(tt)))
    duration = float(tt[-1])
    if dt <= 0 or duration <= 0:
        return FitResult(ok=False, model=model, message="The times in the range do not increase")

    C0 = float(np.mean(y))
    A0 = float((y.max() - y.min()) / 2) or 1e-6
    linear = "m" in spec.names
    m0 = float(np.polyfit(tt, y, 1)[0]) if linear else None

    # Starting frequency: the strongest peak of the spectrum (zero-padded for finer steps)
    z = (y - C0) * np.hanning(len(y))
    n_fft = 16 * len(z)
    spectrum = np.abs(np.fft.rfft(z, n_fft))
    freqs = np.fft.rfftfreq(n_fft, dt)
    spectrum[0] = 0.0
    w0 = 2 * math.pi * float(freqs[int(np.argmax(spectrum))])
    if w0 <= 0:
        return FitResult(ok=False, model=model, message="No oscillation found in the range", t0=t0)
    g0 = 1.0 / max(duration, dt)

    if spec.key.startswith("drag"):
        return _fit_drag(tt, y, spec, t0, w0, A0, C0, m0, warm)

    n = len(spec.names)
    lower = [-np.inf, 0.0, 0.0, -np.inf] + [-np.inf] * (n - 4)
    upper = [np.inf] * n

    def attempt(p0, maxfev):
        try:
            popt, pcov = curve_fit(spec.func, tt, y, p0=p0, bounds=(lower, upper), maxfev=maxfev)
        except (RuntimeError, ValueError):
            return None
        return float(np.sum((spec.func(tt, *popt) - y) ** 2)), popt, pcov

    sst = float(np.sum((y - y.mean()) ** 2))
    best = None
    if warm is not None and warm.ok and warm.model == model:
        # The warm phase is measured from the old range start: move it to this one
        p_w = [warm.params[k] for k in spec.names]
        p_w[3] = warm.phi_abs + p_w[2] * t0
        best = attempt(p_w, maxfev=400)
        if best is not None and sst > 0 and 1.0 - best[0] / sst >= GOOD_FIT_R2:
            return _result(best, spec, t0, sst, y)  # Good enough: skip the search

    base = [A0, g0, w0, 0.0, C0] + ([m0] if linear else [])
    for phi0 in np.linspace(-math.pi, math.pi, 12, endpoint=False):
        p0 = list(base)
        p0[3] = phi0
        result = attempt(p0, maxfev=20000)
        if result is not None and (best is None or result[0] < best[0]):
            best = result

    if best is None:
        return FitResult(ok=False, model=model,
                         message="The fit did not converge; try a different range", t0=t0)
    return _result(best, spec, t0, sst, y)


def _fit_drag(tt, y, spec: FitModel, t0: float, w0: float, A0: float, C0: float,
              m0: float | None, warm: "FitResult | None") -> FitResult:
    """
    Fit of a quadratic-drag model. Each trial integrates the motion, so the search uses a few
    starting points: the swing from both ends and both directions of the first speed. A warm
    start (the previous fit of this model) is tried first, so a slider move is quick.
    """
    linear = "m" in spec.names
    n = len(spec.names)
    sst = float(np.sum((y - y.mean()) ** 2))
    # z0, v0, omega, k, C (and m): drag strength k and the frequency must stay non-negative
    lower = [-np.inf, -np.inf, 0.0, 0.0, -np.inf] + [-np.inf] * (n - 5)
    upper = [np.inf] * n

    def attempt(p0, maxfev):
        try:
            popt, pcov = curve_fit(spec.func, tt, y, p0=p0, bounds=(lower, upper), maxfev=maxfev)
        except (RuntimeError, ValueError):
            return None
        return float(np.sum((spec.func(tt, *popt) - y) ** 2)), popt, pcov

    best = None
    if warm is not None and warm.ok and warm.model == spec.key:
        p_w = [warm.params[k] for k in spec.names]
        best = attempt(p_w, maxfev=200)
        if best is not None and sst > 0 and 1.0 - best[0] / sst >= GOOD_FIT_R2:
            return _result_drag(best, spec, t0, sst)

    for z0 in (A0, -A0):
        for v0 in (w0 * A0, -w0 * A0):
            base = [z0, v0, w0, 0.01 * w0 * w0 / max(A0, 1e-9), C0] + ([m0] if linear else [])
            result = attempt(base, maxfev=2000)
            if result is not None and (best is None or result[0] < best[0]):
                best = result

    if best is None:
        return FitResult(ok=False, model=spec.key,
                         message="The fit did not converge; try a different range", t0=t0)
    return _result_drag(best, spec, t0, sst)


def _result_drag(best, spec: FitModel, t0: float, sst: float) -> FitResult:
    """Turns a (sse, popt, pcov) drag fit into a FitResult."""
    sse, popt, pcov = best
    popt = np.array(popt, dtype=float)
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    params = dict(zip(spec.names, (float(v) for v in popt)))
    errors = dict(zip(spec.names, (float(np.sqrt(v)) if v >= 0 else float("nan")
                                   for v in np.diag(pcov))))
    omega = params["omega"]
    period = 2 * math.pi / omega if omega > 0 else float("nan")
    period_err = 2 * math.pi / omega ** 2 * errors["omega"] if omega > 0 else float("nan")
    message = "" if r2 >= GOOD_FIT_R2 else "Poor fit: check the range"
    return FitResult(ok=True, message=message, model=spec.key, params=params, errors=errors,
                     r2=r2, period=period, period_err=period_err, tau=float("nan"), t0=t0,
                     phi_abs=0.0)


def fit_damped(t, y, warm: "FitResult | None" = None) -> FitResult:
    """The plain damped oscillator (kept for existing callers)."""
    return fit_model(t, y, "damped", warm)


def _result(best, spec: FitModel, t0: float, sst: float, y) -> FitResult:
    """Turns a (sse, popt, pcov) fit into a FitResult."""
    sse, popt, pcov = best
    popt = np.array(popt, dtype=float)
    # A negative amplitude is the same curve with the phase shifted by pi: report A > 0
    if popt[0] < 0:
        popt[0], popt[3] = -popt[0], popt[3] + math.pi
    popt[3] = (popt[3] + math.pi) % (2 * math.pi) - math.pi  # Phase in (-pi, pi]
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    params = dict(zip(spec.names, (float(v) for v in popt)))
    errors = dict(zip(spec.names, (float(np.sqrt(v)) if v >= 0 else float("nan")
                                   for v in np.diag(pcov))))

    omega = params["omega"]
    period = 2 * math.pi / omega if omega > 0 else float("nan")
    period_err = 2 * math.pi / omega ** 2 * errors["omega"] if omega > 0 else float("nan")
    gamma = params["gamma"]
    tau = 1.0 / gamma if gamma > 0 else float("inf")
    phi_abs = params["phi"] - omega * t0  # Phase with absolute time, to reuse as a warm start

    message = "" if r2 >= GOOD_FIT_R2 else "Poor fit: check the range"
    return FitResult(ok=True, message=message, model=spec.key, params=params, errors=errors,
                     r2=r2, period=period, period_err=period_err, tau=tau, t0=t0,
                     phi_abs=phi_abs)


def peak_periods(t, y) -> PeakPeriods:
    """
    Periods from the data itself: the spacing between successive maxima and between successive
    minima, each counted as a full period. Peaks must stand out by a fifth of the signal's range,
    which ignores noise.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(t) & np.isfinite(y)
    t, y = t[keep], y[keep]
    order = np.argsort(t)
    t, y = t[order], y[order]
    empty = PeakPeriods(0, 0, float("nan"), float("nan"), 0)
    if len(t) < 3:
        return empty

    prominence = 0.2 * float(np.ptp(y)) if np.ptp(y) > 0 else None
    maxima, _ = find_peaks(y, prominence=prominence)
    minima, _ = find_peaks(-y, prominence=prominence)

    intervals = np.concatenate([np.diff(t[maxima]), np.diff(t[minima])])
    if intervals.size == 0:
        return PeakPeriods(len(maxima), len(minima), float("nan"), float("nan"), 0)
    mean = float(np.mean(intervals))
    err = float(np.std(intervals, ddof=1) / math.sqrt(intervals.size)) if intervals.size > 1 \
        else float("nan")
    return PeakPeriods(len(maxima), len(minima), mean, err, int(intervals.size))


@dataclass
class InitialDrop:
    ok: bool
    message: str = ""
    t_high: float = float("nan")  # Start of the movement: the first high in the range
    y_high: float = float("nan")
    t_low: float = float("nan")   # The first low after it: the first turnaround
    y_low: float = float("nan")

    @property
    def drop(self) -> float:
        """How far the value falls from the start to the first low (positive when it falls)."""
        return self.y_high - self.y_low

    @property
    def duration(self) -> float:
        """Time from the start to the first low."""
        return self.t_low - self.t_high


def initial_drop(t, y) -> InitialDrop:
    """
    The initial drop of a motion: from the first high (where the movement starts) to the first low
    after it (the first turnaround of the period).

    Highs and lows must stand out by a fifth of the signal's range, as for peak_periods.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(t) & np.isfinite(y)
    t, y = t[keep], y[keep]
    order = np.argsort(t)
    t, y = t[order], y[order]
    if len(t) < 3 or np.ptp(y) <= 0:
        return InitialDrop(ok=False, message="Select a range with a clear movement")
    prominence = 0.2 * float(np.ptp(y))
    maxima, _ = find_peaks(y, prominence=prominence)
    minima, _ = find_peaks(-y, prominence=prominence)
    # The range may start on the high itself (the movement begins at its first sample): find_peaks
    # never counts the ends, so the first sample counts when the signal falls from it
    starts_high = y[0] > y[1] and minima.size > 0 and y[0] - y[minima[0]] >= prominence
    if starts_high and (maxima.size == 0 or maxima[0] > 0):
        start = 0
    elif maxima.size:
        start = int(maxima[0])
    else:
        return InitialDrop(ok=False, message="No starting high in the range")
    after = minima[minima > start]
    if after.size == 0:
        return InitialDrop(ok=False, message="No first low after the starting high")
    low = int(after[0])
    return InitialDrop(ok=True, t_high=float(t[start]), y_high=float(y[start]),
                       t_low=float(t[low]), y_low=float(y[low]))


def best_period(fit: FitResult, peaks: PeakPeriods):
    """
    The period to report: the fit's when the fit is good, otherwise the peak spacing.

    Returns:
        (period, error, source), with source "fit" or "peaks", or (nan, nan, None) if neither
        could be measured.
    """
    if fit.ok and fit.r2 >= GOOD_FIT_R2 and math.isfinite(fit.period):
        return fit.period, fit.period_err, "fit"
    if math.isfinite(peaks.period):
        return peaks.period, peaks.period_err, "peaks"
    return float("nan"), float("nan"), None
