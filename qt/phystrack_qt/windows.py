"""
Every open trial window, by a short id ("w1", "w2", ...), and the agent gate (Space stops an agent).

Any number of windows: each is one video (one trial) with its own setup and history. The agent
tools name windows by id; File > New Window (Ctrl+N) opens another.

QUICK-AND-DIRTY (see docs/AGENT_PLAN.md, "Architecture notes"): a module-level registry, one per
process, rather than an application/document model passed around. Ids are never reused within a
run, so an agent's "w2" never silently means a different trial.
"""

import itertools
import os
import time

from PySide6.QtCore import QObject, Signal

_windows: dict = {}          # id -> MainWindow, in the order they were opened
_ids = itertools.count(1)
active: str | None = None    # The window the user last brought to the front


def add(window) -> str:
    global active
    wid = f"w{next(_ids)}"
    _windows[wid] = window
    active = active or wid
    return wid


def remove(wid: str) -> None:
    global active
    _windows.pop(wid, None)
    if active == wid:
        active = next(reversed(_windows), None) if _windows else None


def activated(wid: str) -> None:
    global active
    if wid in _windows:
        active = wid


def get(wid: str | None = None):
    """The window `wid` (None: the active one), or None."""
    return _windows.get(wid or active) if (wid or active) else None


def all_windows() -> dict:
    return dict(_windows)


def new_window(path: str | None = None, show: bool = True):
    """A new, empty window (with the same proxy cache as the others), opening `path` if given (a
    video or a project)."""
    from .main_window import MainWindow
    cache = next(iter(_windows.values())).cache_dir if _windows else \
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cache")
    window = MainWindow(cache)
    window.resize(1440, 900)
    if show:
        window.show()
    if path:
        if path.endswith(".phystrack"):
            window.open_project_file(path)
        else:
            window.load_video(path)
    return window


def focus(window) -> None:
    """Brings `window` to the front: the user's view follows the agent to it."""
    window.show()
    window.raise_()
    window.activateWindow()
    activated(window.wid)


class AgentGate(QObject):
    """
    Whether an agent is working, and the user's brake on it. Space (in any window) stops a working
    agent: its tool calls are refused until it calls `resume`, which tells it what the user changed
    meanwhile (the history events since the stop).

    QUICK-AND-DIRTY: an outside agent (MCP) counts as working during a tool call and for ACTIVE_S
    after one (it thinks between calls). The in-app assistant says exactly (`session_busy`).
    """
    changed = Signal()
    ACTIVE_S = 30.0

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0          # Tool calls under way
        self.last = 0.0         # When the last one ended
        self.stopped = False
        self.marks: dict = {}   # Window id -> its last event when the user stopped the agent
        self.session_busy = False   # The in-app assistant is working (thinking or calling)

    def busy(self) -> bool:
        """Working right now: a tool call runs, or the in-app assistant is thinking."""
        return not self.stopped and (self.calls > 0 or self.session_busy)

    def running(self) -> bool:
        return self.busy() or (not self.stopped and time.time() - self.last < self.ACTIVE_S)

    def begin(self) -> None:
        self.calls += 1
        self.changed.emit()

    def end(self) -> None:
        self.calls = max(0, self.calls - 1)
        self.last = time.time()
        self.changed.emit()

    def stop(self) -> None:
        """The user's brake: refuses the agent's calls, and stops a track it started."""
        self.stopped = True
        self.marks = {wid: w.history.last() for wid, w in _windows.items()}
        for w in _windows.values():
            if w.job is not None and w.agent_track:   # Only a track the agent started
                w._stop_track()
        self.changed.emit()

    def resume(self) -> dict:
        """Lets the agent go on. Returns, per window, the events since the stop (what the user did)."""
        if not self.stopped:
            return {}
        changes = {wid: w.history.since(self.marks.get(wid, 0)) for wid, w in _windows.items()
                   if w.history.since(self.marks.get(wid, 0))}
        self.stopped = False
        self.last = time.time()
        self.changed.emit()
        return changes


AGENT = AgentGate()
