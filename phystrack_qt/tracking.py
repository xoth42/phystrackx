"""
Tracking for the Qt spike: the parent's MLTracker (OpenCV NanoTrack, as the Tk app uses) run over
a frame range on a worker thread.

The worker opens its own reader on the proxy, so the UI can keep scrubbing while it runs (the Tk
app has to block scrubbing, because both share one reader). Results reach the UI through queued
signals; the worker never touches a widget.

Positions are box centres in proxy pixels (the picture's own pixels), keyed by frame.
"""

import logging
import threading
import time

from PySide6.QtCore import QObject, Signal

from core import PixelRect
from media import VideoReader

from .parent_imports import import_light

MLTracker = import_light("experiments.rigid.mltracker").MLTracker

Box = tuple[float, float, float, float]  # x, y, width, height in proxy pixels


class TrackJob(QObject):
    tracked = Signal(int, list)   # Frame index, a Box (or None if lost) per object
    finished = Signal(bool)       # True if it ran to the end, False if stopped
    failed = Signal(str)

    def __init__(self, proxy_path: str, boxes: list[Box], start: int, end: int) -> None:
        super().__init__()
        self.proxy_path, self.boxes, self.start, self.end = proxy_path, boxes, start, end
        self._stop = threading.Event()
        self.logger = logging.getLogger(__name__)

    def run(self) -> None:
        threading.Thread(target=self._work, daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def _work(self) -> None:
        reader = None
        try:
            reader = VideoReader(self.proxy_path)
            first = reader.read(self.start)
            trackers = []
            for x, y, w, h in self.boxes:
                tracker = MLTracker("nanotrack")
                tracker.initialize_tracker(first, PixelRect(int(x), int(y), int(w), int(h)))
                trackers.append(tracker)
            self.tracked.emit(self.start, list(self.boxes))
            t0 = time.perf_counter()
            for index in range(self.start + 1, self.end + 1):
                if self._stop.is_set():
                    self.finished.emit(False)
                    return
                frame = reader.read(index)  # Sequential, so no seeking
                found = []
                for tracker in trackers:
                    ok, rect = tracker.update_tracker(frame)
                    found.append((rect.xmin, rect.ymin, rect.width, rect.height) if ok else None)
                self.tracked.emit(index, found)
            n = self.end - self.start
            self.logger.info(f"Tracked {n} frames in {time.perf_counter() - t0:.1f} s")
            self.finished.emit(True)
        except Exception as e:  # Reported to the user rather than lost on the worker thread
            self.logger.exception("Tracking failed")
            self.failed.emit(f"Tracking failed: {e}")
        finally:
            if reader is not None:
                reader.release()
