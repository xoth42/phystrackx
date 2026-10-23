"""
The Qt app's start: the main window on the given videos and projects, and the agent bridge.
Imported by cli.py only after the arguments are checked, so a bad command fails without loading Qt.
"""

import logging
import os
import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from phystrack_qt import paths, recent, session, windows
from phystrack_qt.main_window import MainWindow
from phystrack_qt.version import __version__


def stop_on_ctrl_c(app: QApplication) -> QTimer:
    """Ctrl-C quits the app. Qt's event loop runs in C++, where Python's own Ctrl-C handler never
    gets a turn, and an exception raised inside a Qt callback is printed and dropped; so the
    handler asks the loop to quit, and a short timer keeps Python running often enough to see the
    signal. (As the Tk app does.) Returns the timer, which must be kept alive."""
    # A quit called before the loop runs is lost, so the request is posted to the loop instead (a
    # Ctrl-C while the window is still being built then quits as soon as the loop starts)
    signal.signal(signal.SIGINT, lambda signum, frame: QTimer.singleShot(0, app.quit))
    wake = QTimer()
    wake.timeout.connect(lambda: None)
    wake.start(200)
    return wake


def startup_files(files: list[str]) -> list[str]:
    """What to open at start. Files given: those. None given: the windows that were open when the app was
    last closed (session.py), or, on a first start with no record, the most recent project."""
    if files:
        return files
    restored = session.restorable()
    if restored is None:
        return recent.load()[:1]
    return restored


def run(files: list[str]) -> None:
    """Opens the app. The first file opens in the main window; each further one in a window of its own.
    Each file is a video (opened from its first frame) or a .phystrack project. With no files, the
    last session's windows open again (startup_files)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    app.setApplicationName("PhysTrackX")
    app.setApplicationDisplayName("PhysTrackX")
    app.setApplicationVersion(__version__)

    wake = stop_on_ctrl_c(app)   # noqa: F841  (kept alive for the run)
    window = MainWindow(cache_dir=os.environ.get("PHYSTRACK_QT_CACHE") or paths.cache_dir())
    window.resize(1440, 900)
    window.show()
    for k, path in enumerate(startup_files(files)):
        target = window if k == 0 else windows.new_window()
        if path.endswith(".phystrack"):
            target.open_project_file(path)
        else:
            target.load_video(path)
    # The agent tools, for the MCP shim and ptx.py (docs/AGENT_PLAN.md); PHYSTRACK_BRIDGE=0 turns off
    if os.environ.get("PHYSTRACK_BRIDGE", "1") != "0":
        from phystrack_qt.agent import bridge
        bridge.start()
    sys.exit(app.exec())
