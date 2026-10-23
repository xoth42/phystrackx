"""
Tabs (each is a window): Cmd+W closes the current one and shows the next; the last one is replaced by a
blank one. Cmd+T opens a blank one. Cmd+Shift+T reopens the last closed one, with its setup as it was.

    uv run python tests/test_tabs.py
"""

import os

import harness as h  # noqa: F401  (sets up Qt before PySide6 is imported)

from PySide6.QtCore import QRectF  # noqa: E402

from phystrack_qt import windows
from phystrack_qt.main_window import MainWindow  # noqa: E402
from test_tracking import make_clip  # noqa: E402


def close_all() -> None:
    """Every window closed and the closed list cleared, so each test starts alone."""
    for w in list(windows.all_windows().values()):
        w._dirty = False
        w.close()
    windows.closed.clear()


def file_action(window, text: str):
    file_menu = [a for a in window.menuBar().actions() if a.text() == "File"][0].menu()
    return [a for a in file_menu.actions() if a.text() == text][0]


def test_close_tab_shows_the_next_window() -> None:
    close_all()
    first, second = h.window(), h.window()
    first.load_video(make_clip())
    second.load_video(make_clip())
    first.close_tab()
    h.check("closing a tab leaves the other one", list(windows.all_windows()) == [second.wid],
            str(list(windows.all_windows())))
    h.check("...and it comes to the front", windows.get() is second)
    close_all()


def test_closing_the_last_tab_shows_a_blank_one() -> None:
    close_all()
    only = h.window()
    only.load_video(make_clip())
    only.close_tab()
    remaining = list(windows.all_windows().values())
    h.check("the last tab is replaced, so there is one window", len(remaining) == 1, str(len(remaining)))
    h.check("...and it is blank (no video)", remaining and remaining[0].source.path in (None, ""))
    h.check("...and it is not the closed one", remaining and remaining[0] is not only)
    close_all()


def test_new_tab_is_blank() -> None:
    close_all()
    existing = h.window()
    existing.load_video(make_clip())
    new = windows.new_window()
    h.check("New Tab adds a window", len(windows.all_windows()) == 2)
    h.check("...that is blank", new.source.path in (None, ""))
    close_all()


def test_reopen_closed_restores_the_setup() -> None:
    close_all()
    clip = make_clip()
    w = h.window()
    w.load_video(clip)
    w._add_object(QRectF(100, 100, 40, 40))
    objects = len(w.objects)
    dirty = w._dirty
    real = MainWindow._confirm_discard
    MainWindow._confirm_discard = lambda self: True   # The user chose to discard the unsaved changes
    try:
        w.close_tab()
    finally:
        MainWindow._confirm_discard = real
    h.check("a closed tab is remembered", len(windows.closed) == 1 and windows.closed[-1]["video"] == os.path.abspath(clip))
    reopened = windows.reopen_closed()
    h.check("Reopen Closed Tab opens a window", reopened is not None)
    if reopened is not None:
        h.check("...with the same video", reopened.source.path == os.path.abspath(clip), str(reopened.source.path))
        h.check("...and its objects as they were", len(reopened.objects) == objects,
                f"{len(reopened.objects)} vs {objects}")
        h.check("...and its unsaved state", reopened._dirty == dirty)
    h.check("nothing is left to reopen after that", windows.reopen_closed() is None)
    close_all()


def test_shortcuts() -> None:
    close_all()
    w = h.window()
    keys = {"Close Tab": "Ctrl+W", "New Tab": "Ctrl+T", "Reopen Closed Tab": "Ctrl+Shift+T"}
    for text, key in keys.items():
        h.check(f"{text} is {key}", file_action(w, text).shortcut().toString() == key,
                file_action(w, text).shortcut().toString())
    close_all()


if __name__ == "__main__":
    h.app()
    for test in (test_close_tab_shows_the_next_window, test_closing_the_last_tab_shows_a_blank_one,
                 test_new_tab_is_blank, test_reopen_closed_restores_the_setup, test_shortcuts):
        print(test.__name__)
        test()
    h.finish()
