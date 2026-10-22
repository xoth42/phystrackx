"""
Test harness for the Qt spike. Import this before anything from PySide6.

- No window by default: Qt's "offscreen" platform, so tests can run while you work (set
  PHYSTRACK_QT_SHOW=1 to watch them on screen).
- The video view draws on the CPU in tests: Qt cannot capture an OpenGL view on macOS, and the
  offscreen platform has no OpenGL.
- render_picture() draws the scene (frame + overlays) at frame pixels, so a test can check that an
  overlay sits on the right pixels without depending on window size, zoom, or the screen.
"""

import os
import sys
import tempfile
import time

QT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [QT, os.path.dirname(QT)]  # The spike, then the parent repo (media, core)
if not os.environ.get("PHYSTRACK_QT_SHOW"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PHYSTRACK_QT_VIEWPORT", "cpu")

import numpy as np  # noqa: E402
from PySide6.QtCore import QRectF  # noqa: E402
from PySide6.QtGui import QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

WORK = os.path.join(tempfile.gettempdir(), "phystrack_qt_tests")
os.makedirs(WORK, exist_ok=True)
os.environ["PHYSTRACK_QT_HOME"] = os.path.join(WORK, "home")  # The recent list stays out of the real one

_failures: list[str] = []


def app() -> QApplication:
    return QApplication.instance() or QApplication(sys.argv)


def window():
    """A main window with its proxy cache in the test folder."""
    from phystrack_qt.main_window import MainWindow
    w = MainWindow(cache_dir=os.path.join(WORK, "cache"))
    # Track runs NanoTrack straight away, with no method dialog (test_tracking covers the dialog
    # and the other methods)
    w.ask_track_method = False
    w.track_method = "nanotrack"
    w.resize(1280, 800)
    w.show()
    pump(timeout=0.2)
    return w


def pump(cond=lambda: False, timeout: float = 30.0) -> bool:
    """Runs the event loop until cond() is true (True) or the timeout passes (False)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        app().processEvents()
        if cond():
            app().processEvents()
            return True
        time.sleep(0.005)
    return False


def render_picture(view) -> np.ndarray:
    """The picture and its overlays as an RGB array, one pixel per frame pixel."""
    rect = view.picture.sceneBoundingRect()
    image = QImage(int(rect.width()), int(rect.height()), QImage.Format.Format_RGB888)
    painter = QPainter(image)
    view.scene().render(painter, QRectF(image.rect()), rect)
    painter.end()
    a = np.frombuffer(image.constBits(), np.uint8).reshape(image.height(), image.bytesPerLine())
    return a[:, :image.width() * 3].reshape(image.height(), image.width(), 3).copy()


def check(name: str, cond, detail: str = "") -> bool:
    print(f"  {'PASS' if cond else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        _failures.append(name)
    return bool(cond)


def finish() -> None:
    if _failures:
        print(f"{len(_failures)} failed: {', '.join(_failures)}")
        sys.exit(1)
    print("all passed")
