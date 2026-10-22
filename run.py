"""Starts the PhysTrackX Qt spike (experimental; the real app is the Tk one in the parent folder).

    .venv/bin/python run.py [video]
"""

import logging
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# The parent repo's packages (media, core) are reused as they are, read-only
sys.path.insert(0, os.path.dirname(HERE))

from PySide6.QtWidgets import QApplication  # noqa: E402

from phystrack_qt.main_window import MainWindow  # noqa: E402


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    app = QApplication(sys.argv)
    app.setApplicationName("PhysTrackX (Qt spike)")
    window = MainWindow(cache_dir=os.path.join(HERE, "cache"))
    window.resize(1440, 900)
    window.show()
    if len(sys.argv) > 1:
        window.load_video(sys.argv[1])
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
