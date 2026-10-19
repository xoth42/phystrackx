"""
axes.py

Defines the Axes class that manages coordinate transformations and GUI-based axis selection.

Author: Isam Balghari
"""

import numpy as np
import customtkinter as ctk
from .anglecontrol import AngleControl
from .buttons import SubmitButton

class Axes:
    def __init__(self, root, canvas, vwidth: int, vheight: int, btnlist: dict,
                 activebtn: ctk.CTkButton):
        self.root = root
        self.canvas = canvas
        self.vwidth = vwidth
        self.vheight = vheight
        self.theta = ctk.DoubleVar(value=0)
        self.angle = None
        self.placed = False  # True once the user has set an origin
        self.view = None  # ViewTransform, set by the app so axes follow zoom and pan
        self.on_apply = None  # Optional callback after Apply (used to leave marking mode)

        self.ox = 0.0
        self.oy = float(self.vheight)

        self.btnlist = btnlist
        self.activebtn = activebtn
        self.applybtn = SubmitButton(self.canvas, command=self.onapply, size=50,
                                     tooltip="Apply Axes")
        self.on_cancel = None  # Optional callback when selection is aborted (used to leave marking)
        self._selecting = False

    def get_state(self) -> dict:
        """Axes placement in canvas pixels, with the canvas size it was measured on."""
        return {"ox": self.ox, "oy": self.oy, "theta": self.theta.get(),
                "canvas": [self.vwidth, self.vheight]}

    def restore_state(self, state: dict) -> None:
        """Re-places the axes, scaling positions if the window size has changed."""
        sx = self.vwidth / state["canvas"][0]
        sy = self.vheight / state["canvas"][1]
        self.ox = state["ox"] * sx
        self.oy = state["oy"] * sy
        self.theta.set(state["theta"])
        self.placed = True
        self.rotate()

    def clear(self):
        """Clear all canvas drawings related to sliders and axes."""
        self.placed = False
        self.theta.set(0)
        self.canvas.delete("axes")

        if self.angle is not None:
            self.angle.hide()

    def markaxes(self):
        """Enable interactive marking of axes with mouse events."""
        self.canvas.delete("oval")
        self.canvas.delete("axes")
        self.canvas.bind("<Motion>", self.onmove)
        self.canvas.bind("<Button-1>", self.onclick)
        self._toplevel().bind("<Escape>", self.cancelaxes)
        self._selecting = True

        for k, btn in self.btnlist.items():
            if btn != self.activebtn:
                btn.configure(state="disabled")

    def _toplevel(self):
        """The window that receives key presses (the frame holding the canvas does not)."""
        return self.canvas.winfo_toplevel()

    def cancelaxes(self, event=None):
        """Escape while choosing the origin: stop choosing, keep any axes already placed."""
        if not self._selecting:
            return
        self._selecting = False
        self.canvas.unbind("<Motion>")
        self.canvas.unbind("<Button-1>")
        self._toplevel().unbind("<Escape>")
        self.canvas.delete("axes")
        if self.placed:
            self.rotate()  # Put back the axes that were there before

        for btn in self.btnlist.values():
            btn.configure(state="normal")
        if self.on_cancel:
            self.on_cancel()

    def drawaxes(self, startx, starty, xp, yp):
        """Draw x and y axes lines with arrow heads and labels.

        The endpoints must already be on screen (see ray_to_edge). They are not clamped here:
        clamping each coordinate separately bends the line, so the angle changes with zoom.
        """
        sx0, sy0 = startx
        sx1, sy1 = starty
        x0, y0 = xp
        x1, y1 = yp

        self.canvas.create_line(sx0, sy0, x0, y0, fill="red", arrow=ctk.LAST,
                                width=2, tags="axes")
        self.canvas.create_line(sx1, sy1, x1, y1, fill="blue", arrow=ctk.LAST,
                                width=2, tags="axes")

        self.canvas.create_text(x0 - 15, y0 - 15, text="x", fill="red",
                                font=("Arial", 15, "bold"), tags="axes")
        self.canvas.create_text(x1 + 15, y1 - 15, text="y", fill="blue",
                                font=("Arial", 15, "bold"), tags="axes")

    def canvas2reg(self, xc, yc, tx, ty):
        """Convert canvas coordinates to regular (math) frame."""
        return xc - tx, -yc + ty

    def reg2canvas(self, xr, yr, tx, ty):
        """Convert regular (math) coordinates to canvas frame."""
        return xr + tx, -yr + ty

    def rotatez(self, x, y, theta):
        """Rotate a point (x, y) by angle `theta` (radians) around the origin."""
        c, s = np.cos(theta).item(), np.sin(theta).item()
        return x * c - y * s, x * s + y * c

    def rotate(self, event=None):
        """Redraw axes based on the current rotation angle."""
        self.canvas.delete("axes")

        theta_rad = -np.deg2rad(self.theta.get())
        # Canvas direction of each axis (y points down on screen, so the regular y flips)
        xdx, xdy = self._canvas_dir(*self.rotatez(self.vwidth, 0, theta_rad))
        ydx, ydy = self._canvas_dir(*self.rotatez(0, self.vheight, theta_rad))

        # Stored positions are unzoomed; map the origin to the screen. Zoom scales both axes
        # equally, so the directions above hold on screen too.
        sx, sy = self.ox, self.oy
        if self.view is not None:
            sx, sy = self.view.to_canvas(sx, sy)

        self._draw_from(sx, sy, (xdx, xdy), (ydx, ydy))

    @staticmethod
    def _canvas_dir(xr: float, yr: float) -> tuple[float, float]:
        """Regular (math) vector -> canvas vector."""
        return xr, -yr

    def _draw_from(self, sx: float, sy: float, xdir, ydir) -> None:
        """Draws both axes through the screen point (sx, sy) along their directions."""
        segs = []
        for direction in (xdir, ydir):
            seg = self.visible_segment(sx, sy, *direction, self.vwidth, self.vheight)
            # An axis that misses the visible canvas is drawn as a point (nothing visible)
            segs.append(seg if seg is not None else ((sx, sy), (sx, sy)))
        (xlo, xhi), (ylo, yhi) = segs
        self.drawaxes(xlo, ylo, xhi, yhi)

    @staticmethod
    def visible_segment(sx: float, sy: float, dx: float, dy: float, w: float, h: float):
        """
        The part of the axis line through (sx, sy) along (dx, dy) that is inside the canvas
        [0, w] x [0, h]. Its far end is the arrow tip, in the +direction.

        If the origin is on the canvas, the axis starts there and runs forward, as before. If the
        origin is off the canvas (for example when zoomed to another part of the video), the
        whole visible stretch of the line is returned, so the axis stays visible.

        Returns:
            ((x, y) near end, (x, y) far end), or None if the line misses the canvas.
        """
        lo, hi = -float("inf"), float("inf")
        for s, d, bound in ((sx, dx, w), (sy, dy, h)):
            if abs(d) < 1e-12:
                if s < 0 or s > bound:
                    return None
                continue
            t1, t2 = (0.0 - s) / d, (bound - s) / d
            lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
        if lo > hi:
            return None

        if 0.0 <= sx <= w and 0.0 <= sy <= h:
            lo = 0.0  # Origin on the canvas: start there, as the axis always has
        return (sx + lo * dx, sy + lo * dy), (sx + hi * dx, sy + hi * dy)

    def onmove(self, event):
        """Update axes preview to follow mouse position."""
        self.canvas.delete("axes")

        # Unrotated preview: x along +x, y along -y (up the screen)
        self._draw_from(event.x, event.y, (1.0, 0.0), (0.0, -1.0))

    def onclick(self, event):
        """Finalize axis origin selection and show slider for rotation."""
        self.ox, self.oy = event.x, event.y
        if self.view is not None:
            self.ox, self.oy = self.view.to_base(event.x, event.y)
        self.placed = True
        self._selecting = False
        self.canvas.unbind("<Motion>")
        self.canvas.unbind("<Button-1>")
        self._toplevel().unbind("<Escape>")

        self.applybtn.pack(side="right", padx=10, pady=10, anchor="se")

        # Strip below the video (not over it), so the picture stays visible
        # Strip below the video (not over it), so the picture stays visible.
        # The narrow bar gives fine control; the text box covers larger angles.
        self.angle = AngleControl(self.root, self.theta, self.rotate)
        self.angle.show(before=self.canvas)

    def onapply(self):
        """Finalize axis placement and restore other UI buttons."""
        self.angle.hide()
        self.applybtn.pack_forget()

        for btn in self.btnlist.values():
            btn.configure(state="normal")

        if self.on_apply:
            self.on_apply()
