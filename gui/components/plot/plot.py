"""
plot.py

Responsible for plotting position and motion derivatives from processed data using matplotlib.

Author: Isam Balghari
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RadioButtons, RangeSlider

from gui.components.checkbox import Checkbox
from .plottype import PlotType
from .datamanager import DataManager
from .fit import DEFAULT_MODEL, MODELS, best_period, fit_model, initial_drop, peak_periods


class Plot:
    def __init__(self, parent, theme: str = 'ggplot') -> None:
        """
        Initializes the Plot component shell.
        Data coupling is deferred until activate() is invoked.

        Args:
            parent: The parent GUI container.
            theme (str): Matplotlib theme style.
        """
        self.parent = parent
        self.datamanager: DataManager | None = None
        self._widgets = []  # Range sliders must stay referenced, or their callbacks stop
        
        # Apply the visual plotting environment profile
        plt.style.use(theme)
        
    def activate(self, datamanager: DataManager, theme: str = 'ggplot') -> None:
    
        """
        Initializes the Plot component for visualizing data.

        Args:
            parent: The parent GUI container.
            datamanager (DataManager): The data manager with transformed coordinates.
            theme (str): Matplotlib theme style.
        """
        self.datamanager = datamanager
        
        self.checkbox = Checkbox(self.parent, PlotType, text="Choose Plots", callback=self.showplots)

    @property
    def points(self):
        return self.datamanager.processed_points

    @property
    def timestamps(self):
        return self.datamanager.timestamps

    def _unit(self) -> str:
        """The unit of the plotted values: the ruler's unit, or px before a scale is set."""
        return getattr(self.datamanager, "unit", "px")

    @staticmethod
    def _axis(symbol: str, unit: str) -> str:
        """Axis label with its unit, e.g. '$x$ (cm)'. Without a unit, just the symbol."""
        return f"{symbol} ({unit})" if unit else symbol

    def showplots(self, selected_plots: list[str]) -> None:
        """Displays all selected plots."""
        plot_map = {
            PlotType.X.name: self.plotx,
            PlotType.Y.name: self.ploty,
            PlotType.R.name: self.plotr,
            PlotType.XY.name: self.plotxy,
            PlotType.DX.name: self.plotdx,
            PlotType.DY.name: self.plotdy,
            PlotType.D2X.name: self.plotd2x,
            PlotType.D2Y.name: self.plotd2y
        }

        for plot_type in selected_plots:
            plot_func = plot_map.get(plot_type)
            if plot_func:
                plot_func()

        plt.show(block=False)

    def _plot(self, xdata, ydata, xlabel, ylabel, title, color):
        plt.figure()
        plt.title(title, fontname="Segoe UI Emoji")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        xmin, xmax = np.min(xdata), np.max(xdata)
        ymin, ymax = np.min(ydata), np.max(ydata)
        plt.xlim((xmin*0.90, xmax*1.10))  # Add a 10% margin for better visibility
        plt.ylim((ymin*0.90, ymax*1.10))  # Add a 10% margin for better visibility
        plt.plot(xdata, ydata, '.', color=color, label="data")
        plt.legend(loc="best", fontsize=8)  # Placed automatically, away from the data
        return plt.gca()

    def _add_fit(self, ax, t, y) -> None:
        """
        Adds a range slider under a time series. Dragging it picks the part of the data to use;
        the damped oscillator fit and the period estimates are then recomputed for that part.
        """
        fig = ax.figure
        fig.subplots_adjust(bottom=0.25, left=0.25)
        t = np.asarray(t, dtype=float)
        y = np.asarray(y, dtype=float)
        order = np.argsort(t)
        t, y = t[order], y[order]
        if len(t) < 2 or t[0] == t[-1]:
            return

        fit_line, = ax.plot([], [], '-', color='k', linewidth=2, label='fit')
        drop_marks, = ax.plot([], [], 'o', color='tab:red', markersize=8, label='initial drop')
        ax.legend(loc="best", fontsize=8)  # Automatic position: the spot with the least data
        info = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left", fontsize=8,
                       family="monospace", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

        slider_ax = fig.add_axes([0.25, 0.08, 0.6, 0.03])
        slider = RangeSlider(slider_ax, "Range (s)", float(t[0]), float(t[-1]),
                             valinit=(float(t[0]), float(t[-1])))

        keys = list(MODELS)
        radio_ax = fig.add_axes([0.01, 0.42, 0.2, 0.14], frameon=False)
        radio_ax.set_title("Fit", fontsize=9, loc="left")
        radio = RadioButtons(radio_ax, [MODELS[k].label for k in keys],
                             active=keys.index(DEFAULT_MODEL))
        for label in radio.labels:
            label.set_fontsize(8)

        last = {"fit": None}  # The previous fit starts the next one, so dragging stays smooth

        def update(_=None):
            lo, hi = slider.val
            mask = (t >= lo) & (t <= hi)
            tp, yp = t[mask], y[mask]
            model = keys[radio.index_selected]
            fit = fit_model(tp, yp, model, warm=last["fit"])
            if fit.ok:
                last["fit"] = fit
            peaks = peak_periods(tp, yp)
            period, perr, source = best_period(fit, peaks)
            drop = initial_drop(tp, yp)
            if drop.ok:
                drop_marks.set_data([drop.t_high, drop.t_low], [drop.y_high, drop.y_low])
            else:
                drop_marks.set_data([], [])

            if fit.ok:
                names = MODELS[fit.model].names
                tt = tp - fit.t0
                fit_line.set_data(tp, MODELS[fit.model].func(tt, *(fit.params[k] for k in names)))
            else:
                fit_line.set_data([], [])
            info.set_text(_summary(fit, peaks, period, perr, source, self._unit(), drop))
            fig.canvas.draw_idle()

        slider.on_changed(update)
        radio.on_clicked(update)
        update()
        self._widgets.extend([slider, radio])

    def plotx(self):
        for k, tpts in enumerate(self.points):
            ax = self._plot(self.timestamps, tpts[:, 0], xlabel=r"$T(s)$", ylabel=self._axis(r"$x$", self._unit()),
                            title=f"[O-{k+1}] " + r"$x$ vs $T$", color='m')
            self._add_fit(ax, self.timestamps, tpts[:, 0])

    def ploty(self):
        for k, tpts in enumerate(self.points):
            ax = self._plot(self.timestamps, tpts[:, 1], xlabel=r"$T(s)$", ylabel=self._axis(r"$y$", self._unit()),
                            title=f"[O-{k+1}] " + r"$y$ vs $T$", color='m')
            self._add_fit(ax, self.timestamps, tpts[:, 1])

    def plotr(self):
        """Distance from the axes origin, r = sqrt(x^2 + y^2), against time, with the fit."""
        for k, tpts in enumerate(self.points):
            r = np.hypot(tpts[:, 0], tpts[:, 1])
            ax = self._plot(self.timestamps, r, xlabel=r"$T(s)$", ylabel=self._axis(r"$r$", self._unit()),
                            title=f"[O-{k+1}] " + r"$r$ vs $T$", color='m')
            self._add_fit(ax, self.timestamps, r)

    def plotxy(self):
        for k, tpts in enumerate(self.points):
            self._plot(tpts[:, 0], tpts[:, 1], xlabel=self._axis(r"$x$", self._unit()), ylabel=self._axis(r"$y$", self._unit()),
                       title=f"[O-{k+1}] " + r"$y$ vs $x$", color='c')

    def plotdx(self):
        for k, tpts in enumerate(self.points):
            dx_dt = np.gradient(tpts[:, 0], self.timestamps)
            self._plot(self.timestamps, dx_dt, xlabel=r"$T(s)$", ylabel=self._axis(r"$\frac{dx}{dt}$", f"{self._unit()}/s"),
                       title=f"[O-{k+1}] " + r"$\frac{dx}{dt}$", color='g')

    def plotdy(self):
        for k, tpts in enumerate(self.points):
            dy_dt = np.gradient(tpts[:, 1], self.timestamps)
            self._plot(self.timestamps, dy_dt, xlabel=r"$T(s)$", ylabel=self._axis(r"$\frac{dy}{dt}$", f"{self._unit()}/s"),
                       title=f"[O-{k+1}] " + r"$\frac{dy}{dt}$", color='g')

    def plotd2x(self):
        for k, tpts in enumerate(self.points):
            dx_dt = np.gradient(tpts[:, 0], self.timestamps)
            d2x_dt2 = np.gradient(dx_dt, self.timestamps)
            self._plot(self.timestamps, d2x_dt2, xlabel=r"$T(s)$", ylabel=self._axis(r"$\frac{d^2x}{dt^2}$", f"{self._unit()}/s²"),
                       title=f"[O-{k+1}] " + r"$\frac{d^2x}{dt^2}$", color='b')

    def plotd2y(self):
        for k, tpts in enumerate(self.points):
            dy_dt = np.gradient(tpts[:, 1], self.timestamps)
            d2y_dt2 = np.gradient(dy_dt, self.timestamps)
            self._plot(self.timestamps, d2y_dt2, xlabel=r"$T(s)$", ylabel=self._axis(r"$\frac{d^2y}{dt^2}$", f"{self._unit()}/s²"),
                       title=f"[O-{k+1}] " + r"$\frac{d^2y}{dt^2}$", color='b')


def _summary(fit, peaks, period, perr, source, unit: str = "", drop=None) -> str:
    """Text box for the fit parameters, period estimates, and the initial drop.
    Amplitude, offset, and drop are in `unit`."""
    lines = []
    if fit.ok:
        p, e = fit.params, fit.errors
        u = f" {unit}" if unit else ""
        if fit.model.startswith("drag"):
            lines += [
                f"z0    = {p['z0']:.3g} ± {e['z0']:.2g}{u}",
                f"v0    = {p['v0']:.3g} ± {e['v0']:.2g}{u}/s",
                f"omega = {p['omega']:.4g} ± {e['omega']:.2g} rad/s",
                f"k     = {p['k']:.3g} ± {e['k']:.2g} /{unit or 'unit'}",
                f"C     = {p['C']:.4g} ± {e['C']:.2g}{u}",
            ]
        else:
            lines += [
                f"A     = {p['A']:.3g} ± {e['A']:.2g}{u}",
                f"gamma = {p['gamma']:.3g} ± {e['gamma']:.2g} /s   tau = {fit.tau:.3g} s",
                f"omega = {p['omega']:.4g} ± {e['omega']:.2g} rad/s",
                f"phi   = {p['phi']:.3g} ± {e['phi']:.2g} rad",
                f"C     = {p['C']:.4g} ± {e['C']:.2g}{u}",
            ]
        lines += ([f"m     = {p['m']:.4g} ± {e['m']:.2g}{u}/s"] if "m" in p else []) + [
            f"R²    = {fit.r2:.4f}",
            f"T fit = {fit.period:.4g} ± {fit.period_err:.2g} s",
        ]
        if fit.message:
            lines.append(fit.message)
    else:
        lines.append(fit.message)
    if peaks.period == peaks.period:  # Not NaN
        lines.append(f"T peaks = {peaks.period:.4g} ± {peaks.period_err:.2g} s "
                     f"({peaks.n_maxima} max, {peaks.n_minima} min)")
    if source is not None:
        lines.append(f"Best T = {period:.4g} ± {perr:.2g} s  [{source}]")
    if drop is not None:
        u = f" {unit}" if unit else ""
        if drop.ok:
            lines.append(f"Initial drop = {drop.drop:.4g}{u} in {drop.duration:.4g} s "
                         f"(high at {drop.t_high:.3g} s, low at {drop.t_low:.3g} s)")
        else:
            lines.append(f"Initial drop: {drop.message}")
    return "\n".join(lines)


def main():
    """
    GUI-based test for the Plot system using dummy circular motion data.
    """
    import numpy as np
    import customtkinter as ctk
    from gui.components.tpoints import FPoint
    from gui.components.axes import Axes
    from gui.components.plot.datamanager import DataManager
    from gui.components.plot.plot import Plot
    from experiments.components.ocr import OCRData

    # --- Setup GUI ---
    ctk.set_appearance_mode("System")
    root = ctk.CTk()
    root.geometry("900x800")
    root.title("Plot and Axes Test")

    canvas = ctk.CTkCanvas(root, width=600, height=500, bg="white")
    canvas.pack(pady=10)

    btn_frame = ctk.CTkFrame(root)
    btn_frame.pack(pady=5)
    axes_btn = ctk.CTkButton(btn_frame, text="Set Axes")
    axes_btn.pack()

    btnlist = {"axes": axes_btn}
    axes = Axes(root, canvas, vwidth=600, vheight=500, btnlist=btnlist, activebtn=axes_btn)

    # --- Generate dummy data ---
    t = np.linspace(0, 2*np.pi, 360)
    x = 100 + 200 + 200 * np.cos(t)
    y = 200 + 200 * np.sin(t)

    fpoints = [[FPoint(x[i], y[i], 0, 0) for i in range(len(x))], 
               [FPoint(x[i], y[i], 0, 0) for i in range(len(x))]]

    # --- Dummy OCR data ---
    ocr_text = [["OCR={:.2f}s".format(i / 24) for i in range(len(x))],
                ["OCR={:.2f}s".format(i / 24) for i in range(len(x))]]
    ocrdata = OCRData(ocr_text)

    # --- Create DataManager ---
    datamanager = DataManager(tpoints=fpoints, ocrdata=ocrdata, axes=axes, vwidth=600, vheight=500,
                              fwidth=600, fheight=500, fps=1, scale=1.0)

    # --- Setup Axes and Plot after marking ---
    def on_axes_applied():
        datamanager.transform()
        Plot(parent=root, datamanager=datamanager, theme='ggplot')

    axes_btn.configure(command=lambda: [axes.markaxes(), root.after(3000, on_axes_applied)])
    root.mainloop()


if __name__ == "__main__":
    main()
