"""
The scale ruler: placing it, giving it a real length and unit, dragging its ends and bar with the
mouse, and Auto Ruler (with Undo) on a clip with a coloured bar. Also the axes' geometry, which the
measurements rely on.

    ~/.venvs/phystrackx-qt/bin/python tests/test_ruler.py      (PHYSTRACK_QT_SHOW=1 to watch)
"""

import math
import os

import harness as h  # noqa: F401  (sets up Qt before PySide6 is imported)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PySide6.QtCore import QPointF  # noqa: E402

from core.autoruler import snap_ruler  # noqa: E402
from phystrack_qt.measure import physical  # noqa: E402
from test_tracking import drag  # noqa: E402  (a real mouse drag, as a user makes)

W, H = 640, 480
BAR = (300, 360, 120, 360)  # x0, x1, y0, y1 of the coloured bar in the clip


def make_bar_clip() -> str:
    """A red bar on a grey wall: the colour is what Auto Ruler looks for."""
    path = os.path.join(h.WORK, "red_bar.mp4")
    if os.path.isfile(path):
        return path
    rng = np.random.default_rng(2)
    wall = np.clip(200 + rng.normal(0, 4, (H, W, 1)), 0, 255).astype(np.uint8).repeat(3, 2)
    x0, x1, y0, y1 = BAR
    wall[y0:y1, x0:x1] = (60, 60, 220)  # BGR: red
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for _ in range(10):
        out.write(wall)
    out.release()
    return path


def test_ruler() -> None:
    w = h.window()
    w.load_video(make_bar_clip())
    h.check("proxy is made", h.pump(lambda: w.source.has_proxy, 60))

    w.start_ruler()
    r = w.ruler
    h.check("placing the ruler makes it editable, with its panel", r is not None and r.editable
            and w.ruler_panel.isVisible())
    r.set_ends(QPointF(285, 100), QPointF(285, 380))  # Beside the bar's left edge
    h.check("uncalibrated: the label is in pixels, the scale is 1",
            r.label_text().endswith(" px") and r.scale() == 1.0 and r.unit_label() == "px",
            r.label_text())

    r.set_length(50, "cm")
    h.check("the label shows the typed length and unit", r.label_text() == "50.00 cm",
            r.label_text())
    h.check("the scale is the real length over the picture length",
            abs(r.scale() * r.px_length() - 50) < 1e-9, f"{r.scale() * r.px_length():.6f}")
    h.check("the unit is cm", r.unit_label() == "cm")

    drag(w.view, (285, 380), (345, 380))  # Grab the lower end and stretch it
    h.check("dragging an end moves it", abs(r.p2.x() - 345) < 2, f"p2 at {r.p2}")
    h.check("the typed length is kept when the ruler changes",
            r.real_length == 50 and r.unit == "cm")
    h.check("the scale follows the new picture length",
            abs(r.scale() * r.px_length() - 50) < 1e-9, f"{r.scale():.5f} per px")

    before = (r.p1.y(), r.p2.y())
    mid = ((r.p1.x() + r.p2.x()) / 2, (r.p1.y() + r.p2.y()) / 2)
    drag(w.view, mid, (mid[0], mid[1] + 30))  # Grab the bar and move it down
    h.check("dragging the bar moves both ends together",
            abs(r.p1.y() - before[0] - 30) < 2 and abs(r.p2.y() - before[1] - 30) < 2,
            f"{r.p1.y() - before[0]:.1f}, {r.p2.y() - before[1]:.1f}")

    # Auto Ruler: from roughly beside the bar, the ends go to its top and bottom
    r.set_ends(QPointF(285, 100), QPointF(285, 380))
    w.auto_ruler()
    top, bottom = sorted([r.p1.y(), r.p2.y()])
    h.check("Auto Ruler puts the ends on the bar's top and bottom",
            abs(top - BAR[2]) < 4 and abs(bottom - BAR[3]) < 4, f"ends at y {top:.1f}, {bottom:.1f}")
    h.check("Undo Auto Ruler is available", w.undo_btn.isEnabled())
    w.undo_auto_ruler()
    h.check("Undo puts the ends back where they were",
            abs(r.p1.y() - 100) < 0.5 and abs(r.p2.y() - 380) < 0.5, f"{r.p1.y()}, {r.p2.y()}")
    h.check("and Undo is no longer available", not w.undo_btn.isEnabled())

    # Axes: the picture point that lies along the drawn x axis measures as +x, at any rotation
    w.start_axis()
    w._place_axes(QPointF(150, 200))
    w.axes.setRotation(30)
    origin = w.axes.pos()
    length = w.axes.length
    along_x = w.axes.mapToParent(QPointF(length, 0))
    along_y = w.axes.mapToParent(QPointF(0, -length))
    got = physical([[along_x.x(), along_x.y()], [along_y.x(), along_y.y()]],
                   (origin.x(), origin.y()), w.axes.rotation(), scale=1.0)
    h.check("the drawn axes measure correctly when rotated (x along x, y along y)",
            np.allclose(got, [[length, 0], [0, length]], atol=1e-6), str(got.tolist()))
    h.check("the axis lines run across the whole picture",
            w.axes.boundingRect().width() >= 2 * math.hypot(W, H), f"{w.axes.boundingRect().width():.0f}")
    h.check("and the rotation is what the axes report",
            math.isclose(w.axes.angle(), -30, abs_tol=1e-9))

    w.apply_ruler()
    h.check("applying the ruler stops editing and hides its panel",
            not r.editable and not w.ruler_panel.isVisible())
    w._dirty = False  # Closing would ask about unsaved changes, and a test has nobody to answer
    w.close()


def test_auto_ruler_over_tape() -> None:
    """A brown bar with pale tape across its top and bottom (as a wooden ruler has). The ends must
    land on the bar's left edge and its top and bottom, to about a tenth of a pixel. (Before the
    fix, the tape cut the bar and the ends stopped at it.)"""
    rng = np.random.default_rng(3)
    image = np.clip(200 + rng.normal(0, 4, (H, W, 1)), 0, 255).astype(np.uint8).repeat(3, 2)
    x0, x1, y0, y1 = 300, 325, 120, 380
    image[y0:y1, x0:x1] = (60, 100, 150)              # Brown: a different colour from the wall
    image[150:165, 290:340] = (235, 235, 235)         # Pale tape across the top
    image[360:372, 290:340] = (235, 235, 235)         # ...and near the bottom
    p1, p2 = (312, 100), (312, 400)                   # Over the bar, as a user lays it, with its width on it
    q1, q2 = snap_ruler(image, p1, p2, side="left", half_width=15)
    top, bottom = sorted([q1[1], q2[1]], key=lambda v: v)
    h.check("Auto Ruler runs over tape, from the bar's top to its bottom",
            abs(top - (y0 - 0.5)) < 0.5 and abs(bottom - (y1 - 0.5)) < 0.5,
            f"ends at y {top:.2f}, {bottom:.2f} (bar rows {y0} to {y1 - 1})")
    xs = [q1[0], q2[0]]
    h.check("and on the bar's left edge", all(abs(x - (x0 - 0.5)) < 0.5 for x in xs),
            f"x {xs[0]:.2f}, {xs[1]:.2f} (left edge at {x0 - 0.5})")


def test_axis_knobs() -> None:
    """Both knobs turn the axes: the red x knob puts the x axis at the pointer, the green y knob puts
    the y axis at the pointer. Dragged with the mouse, as a user does."""
    w = h.window()
    w.load_video(make_bar_clip())
    h.pump(lambda: w.source.has_proxy, 60)
    w.start_axis()
    w._place_axes(QPointF(200, 240))
    axes = w.axes
    origin = axes.pos()

    def picture_point(item_point: QPointF) -> tuple[float, float]:
        p = axes.mapToParent(item_point)
        return p.x(), p.y()

    def direction(item_vector: QPointF) -> float:
        """Angle, in degrees on screen, of an axis direction (picture coordinates)."""
        a = picture_point(item_vector)
        return math.degrees(math.atan2(a[1] - origin.y(), a[0] - origin.x()))

    reach = 150.0
    # Drag the x knob to 30 degrees below the horizontal (screen angle +30)
    target = (origin.x() + reach * math.cos(math.radians(30)), origin.y() + reach * math.sin(math.radians(30)))
    drag(w.view, picture_point(QPointF(axes.length, 0)), target)
    x_angle = direction(QPointF(axes.length, 0))
    h.check("the red knob turns the x axis to the pointer", abs(x_angle - 30) < 1.0, f"{x_angle:.1f}°")

    # Drag the green knob to 60 degrees above the horizontal (screen angle -60): the y axis points there
    target = (origin.x() + reach * math.cos(math.radians(-60)), origin.y() + reach * math.sin(math.radians(-60)))
    drag(w.view, picture_point(QPointF(0, -axes.length)), target)
    y_angle = direction(QPointF(0, -axes.length))
    h.check("the green knob turns the y axis to the pointer", abs(y_angle - -60) < 1.0, f"{y_angle:.1f}°")
    h.check("the x axis stays at right angles to the y axis",
            abs(((direction(QPointF(axes.length, 0)) - y_angle) % 360) - 90) < 1.0)

    # The picture, with the axes drawn over it (render_picture: frame pixels, as the tests see it)
    image = h.render_picture(w.view)
    out = os.path.join(h.WORK, "axes_knobs.png")
    cv2.imwrite(out, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    print(f"    (picture with axes at {out})")
    w._dirty = False
    w.close()


if __name__ == "__main__":
    h.app()
    print("test_ruler")
    test_ruler()
    print("test_axis_knobs")
    test_axis_knobs()
    print("test_auto_ruler_over_tape")
    test_auto_ruler_over_tape()
    h.finish()
