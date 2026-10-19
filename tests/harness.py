"""
harness.py

Shared helpers for the PhysTrackX test suite: environment setup (Tcl/Tk on macOS), a
synthetic test clip, a Tk pump, check reporting, and a runner for test functions.

Author: Isam Balghari
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(TESTS_DIR, os.pardir))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

WORK_DIR = os.path.join(tempfile.gettempdir(), "phystrack_tests")

_passes = 0
_failures: list[str] = []
DIALOGS: list[tuple] = []  # Message boxes raised by the app, recorded instead of shown


def setup_env() -> None:
    """Runs from the repo root and points Tcl/Tk at the uv-managed Python if needed (macOS)."""
    os.chdir(REPO)  # The app writes ./temp and ./logs relative to the working directory
    if "TCL_LIBRARY" not in os.environ:
        hits = glob.glob(os.path.expanduser(
            "~/.local/share/uv/python/cpython-3.12*-macos*/lib/tcl9.0"))
        if hits:
            os.environ["TCL_LIBRARY"] = hits[0]
            os.environ["TK_LIBRARY"] = hits[0].replace("tcl9.0", "tk9.0")
    os.makedirs(WORK_DIR, exist_ok=True)
    # Test runs keep their projects out of the real projects/ folder
    import core.project as project
    project.PROJECTS_DIR = os.path.join(WORK_DIR, "projects")


def say(*args) -> None:
    """Prints to the real terminal. The app replaces sys.stdout with a log file when it is not
    attached to a terminal, so the tests write to the original streams."""
    print(*args, file=sys.__stdout__, flush=True)


def check(name: str, cond, detail: str = "") -> bool:
    """Records one check. Prints PASS or FAIL and returns the condition."""
    global _passes
    if cond:
        _passes += 1
        say(f"  PASS {name}")
    else:
        _failures.append(name)
        say(f"  FAIL {name} {detail}")
    return bool(cond)


def pump(root, cond=lambda: False, timeout: float = 60.0) -> bool:
    """Runs the Tk event loop until cond() is true or the timeout passes."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        root.update()
        time.sleep(0.02)
        if cond():
            root.update()
            return True
    return False


def make_clip(seconds: int = 12, size: str = "540x960") -> str:
    """A synthetic portrait clip (cached in the work folder). Tests use this, not a personal video."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is needed to make the test clip")
    path = os.path.join(WORK_DIR, f"clip_{size}_{seconds}s.mp4")
    if not os.path.isfile(path):
        subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-f", "lavfi",
                        "-i", f"testsrc2=duration={seconds}:size={size}:rate=30",
                        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", path],
                       check=True)
    return path


def quiet_dialogs(ra_module) -> None:
    """Replaces the app's message boxes with a recorder, so tests never block on a dialog."""
    ra_module.messagebox.showinfo = lambda t, m: DIALOGS.append(("info", t, m))
    ra_module.messagebox.showerror = lambda t, m: DIALOGS.append(("error", t, m))


def run_tests(tests) -> None:
    """Runs each test function, then prints a summary and exits non-zero on any failure."""
    for fn in tests:
        say(f"[{fn.__module__}.{fn.__name__}]")
        try:
            fn()
        except Exception:
            _failures.append(fn.__name__)
            say(traceback.format_exc())
    say(f"\n{_passes} checks passed, {len(_failures)} failed")
    sys.exit(1 if _failures else 0)
