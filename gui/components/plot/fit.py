"""
fit.py

Fits of a time series with a damped oscillation, and period measurements. No GUI here, so the
maths can be tested on its own.

Models, with tt = t - t[0] (time from the start of the chosen range):

    "damped":        y = C + A * exp(-gamma * tt) * cos(omega * tt + phi)
    "damped_linear": y = C + m * tt + A * exp(-gamma * tt) * cos(omega * tt + phi)

The linear term lets the oscillation settle onto a sloped line rather than a flat one.

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
