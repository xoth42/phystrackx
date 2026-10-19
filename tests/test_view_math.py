"""Zoom and pan transform (no window needed)."""

import random

from tests import harness as h
h.setup_env()

from gui.components.view import ViewTransform  # noqa: E402


def test_round_trip_at_random_zooms():
    random.seed(1)
    v = ViewTransform()
    v.configure((50, 40), (400, 600))
    ok = True
    for _ in range(200):
        v.set_zoom(random.uniform(1, 8))
        v.set_fraction(random.random(), random.random())
        p = (random.uniform(0, 1140), random.uniform(0, 641))
        q = v.to_base(*v.to_canvas(*p))
        ok &= abs(q[0] - p[0]) < 1e-6 and abs(q[1] - p[1]) < 1e-6
    h.check("to_base(to_canvas(p)) == p at random zoom and pan", ok)


def test_window_stays_inside_frame():
    v = ViewTransform()
    v.configure((50, 40), (400, 600))
    v.set_zoom(4)
    v.set_fraction(0.5, 0.5)
    x, y, w, h_ = v.window()
    h.check("zoom 4 window is a quarter of the frame", (w, h_) == (100, 150), str((w, h_)))
    h.check("window lies inside the frame", x + w <= 400 and y + h_ <= 600)
    v.set_fraction(-5, 9)
    h.check("pan clamps to the frame edges", v.ox == 0.0 and v.oy == 600 - 600 / 4, str((v.ox, v.oy)))


def test_zoom_keeps_centre():
    v = ViewTransform()
    v.configure((0, 0), (400, 600))
    v.set_zoom(2)
    v.set_fraction(0.3, 0.6)
    before = v.ox + v.size[0] / v.zoom / 2
    v.set_zoom(6)
    after = v.ox + v.size[0] / v.zoom / 2
    h.check("zooming keeps the centre of the view", abs(before - after) < 1e-6 or v.ox == 0.0,
            str((before, after)))


if __name__ == "__main__":
    h.run_tests([test_round_trip_at_random_zooms, test_window_stays_inside_frame,
                 test_zoom_keeps_centre])
