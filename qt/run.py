"""Starts the PhysTrackX Qt spike (experimental; the real app is the Tk one in the parent folder).

    ~/.venvs/phystrackx-qt/bin/python run.py [video or project.phystrack]
"""

import logging
import os
import signal
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The parent repo's packages (media, core) are reused as they are, read-only
sys.path.insert(0, os.path.dirname(HERE))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from phystrack_qt import windows  # noqa: E402
from phystrack_qt.main_window import MainWindow  # noqa: E402


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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    app.setApplicationName("PhysTrackX (Qt spike)")

    wake = stop_on_ctrl_c(app)   # noqa: F841  (kept alive for the run)
    window = MainWindow(cache_dir=os.environ.get("PHYSTRACK_QT_CACHE") or os.path.join(HERE, "cache"))
    window.resize(1440, 900)
    window.show()
    # Each further argument opens in a window of its own (any number of trials)
    for k, path in enumerate(sys.argv[1:]):
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


if __name__ == "__main__":
    main()
