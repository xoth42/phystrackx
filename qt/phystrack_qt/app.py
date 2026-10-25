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
from phystrack_qt.agent import session as agent_session
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
    """Opens the app. The first file opens in the first tab; each further one in a tab of its own.
    Each file is a video (opened from its first frame) or a .phystrack project. With no files, the
    last session's windows open again (startup_files)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    app.setApplicationName("PhysTrackX")
    app.setApplicationDisplayName("PhysTrackX")
    app.setApplicationVersion(__version__)
    from phystrack_qt import settings_dialog
    settings_dialog.name_settings_item(app)   # "Settings…" in the app menu, as macOS names it now

    wake = stop_on_ctrl_c(app)   # noqa: F841  (kept alive for the run)
    # The first trial, in a tab of the first window
    window = windows.new_window(cache_dir=os.environ.get("PHYSTRACK_QT_CACHE") or paths.cache_dir())
    # Nothing else runs before the event loop: until it runs the window is not painted, and macOS does not
    # bring the app forward. The trials, the bridge and the assistant start on its first turn instead.
    QTimer.singleShot(0, lambda: start_at_launch(app, window, files))
    sys.exit(app.exec())


def warm_launch() -> None:
    """The install's hidden launch (build.py warm_launch). The first time macOS opens a new app, it checks
    each program and library as it loads (XProtect, syspolicyd): about 5 s for this app, before its window
    shows. Loading them all here, with no window and nothing saved, moves that wait into the install.
    Loads what a start loads (Qt, the main window and its modules), what the plots load, and ffmpeg."""
    import subprocess
    import tempfile

    from core import find_ffmpeg
    from phystrack_qt import plot_warmup

    os.environ["QT_QPA_PLATFORM"] = "offscreen"   # Nothing is shown
    with tempfile.TemporaryDirectory() as home:
        os.environ["PHYSTRACK_QT_HOME"] = home     # Nothing goes to the user's cache, recent list or session
        app = QApplication(sys.argv)   # noqa: F841  (the main window needs one)
        windows.new_window(show=False, cache_dir=paths.cache_dir())   # The tab window and a trial in it
        plot_warmup.warm_now()
        try:
            subprocess.run([find_ffmpeg(), "-version"], capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            pass


def start_at_launch(app: QApplication, window, files: list[str]) -> None:
    """On the event loop's first turn: brings the window to the front, opens the trials, then starts the
    agent tools and the assistant. Stops the assistant's model runner when the app quits (not when a
    window closes)."""
    windows.focus(window)
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
    assistant = agent_session.shared()
    app.aboutToQuit.connect(assistant.shutdown)
    assistant.warm_up()
