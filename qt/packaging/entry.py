"""The program a packaged app starts: the same command-line entry as `phystrackx` (phystrack_qt/cli.py)."""

import os
import sys

if sys.stdout is None or sys.stderr is None:
    # The Windows build has no console (phystrackx.spec), so Python has no output streams. Send them to a log
    # file: otherwise a crash would leave no trace. The logging setup in core/logging.py must see these
    # streams, so this runs before the import below.
    from platformdirs import user_log_dir
    log_dir = user_log_dir("PhysTrackX", appauthor=False)
    try:
        os.makedirs(log_dir, exist_ok=True)
        log = open(os.path.join(log_dir, "phystrackx.log"), "a", encoding="utf-8", buffering=1)
    except OSError:
        log = open(os.devnull, "w", encoding="utf-8")
    sys.stdout = sys.stderr = log

from phystrack_qt.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
