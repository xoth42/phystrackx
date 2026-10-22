"""
Opens a video for the Qt spike: makes (or reuses) a proxy with the parent repo's media.proxyvideo
on a worker thread, then reads frames from it on the UI thread with media.VideoReader.

Qt signals carry the results back to the UI thread, so the worker never touches a widget.
"""

import os
import threading

import cv2
from PySide6.QtCore import QObject, Signal

from media import VideoReader, proxyvideo
from media.proxyvideo import _cache_key, display_size

PROXY_MAX_SIDE = 1600  # The proxy's long side. The view scales it on the GPU, so this sets detail


class VideoSource(QObject):
    progress = Signal(float)      # Proxy encoding, 0.0 to 1.0
    loaded = Signal(str)          # Source path, once frames can be read
    failed = Signal(str)          # A message for the user

    def __init__(self, cache_dir: str) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        self.reader: VideoReader | None = None
        self.path: str | None = None

    @property
    def fcount(self) -> int:
        return self.reader.fcount if self.reader else 0

    @property
    def fps(self) -> float:
        return self.reader.fps if self.reader else 0

    def load(self, path: str) -> None:
        """Starts loading; loaded or failed is emitted when it ends."""
        threading.Thread(target=self._make_proxy, args=(path,), daemon=True).start()

    def _make_proxy(self, path: str) -> None:
        try:
            size = display_size(path)
            if size is None:
                self.failed.emit(f"Could not read the video size of {os.path.basename(path)}")
                return
            w, h = size
            scale = min(1.0, PROXY_MAX_SIDE / max(w, h))
            width = max(2, round(w * scale / 2) * 2)  # Even, as libx264 needs
            capture = cv2.VideoCapture(path)
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()

            os.makedirs(self.cache_dir, exist_ok=True)
            name = os.path.splitext(os.path.basename(path))[0]
            # Same cache key as the Tk app, so a changed source or encoder setting makes a new proxy
            out = os.path.join(self.cache_dir, f"{name}_{_cache_key(path, width, -2)}_proxy.mp4")
            if not os.path.isfile(out):
                out = proxyvideo(path, width=width, writepath=out,
                                 on_progress=self.progress.emit, total_frames=total)
            if out is None:
                self.failed.emit(f"Could not make a proxy of {os.path.basename(path)} (see the log)")
                return
            self._proxy = out
            self.loaded.emit(path)
        except Exception as e:  # Reported to the user rather than lost on the worker thread
            self.failed.emit(f"Loading failed: {e}")

    def open_proxy(self, path: str) -> None:
        """Called on the UI thread after loaded: frames are read here from now on."""
        if self.reader is not None:
            self.reader.release()
        self.reader = VideoReader(self._proxy)
        self.path = path

    def frame(self, index: int):
        """The frame at index (BGR), from the proxy."""
        return self.reader.read(index)
