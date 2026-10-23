"""
Opens a video for the Qt spike.

Frames can be shown at once: the source file is opened straight away (about 75 ms) and read,
scaled to the proxy's size, while the proxy is encoded on a worker thread by the parent repo's
media.proxyvideo (about 11 s for a 39 s 1080p HEVC phone clip; decoding alone is 7 s of that, so
no encoder setting makes the wait go away). When the proxy is ready, reading switches to it:
seeking it is several times faster (about 30 ms vs 200 ms). Frames are in proxy pixels throughout,
so overlays keep their positions across the switch.

Qt signals carry the worker's results back to the UI thread, so the worker never touches a widget.
"""

import os
import threading

import cv2
from PySide6.QtCore import QObject, Signal

from media import VideoReader, proxyvideo
from media.proxyvideo import _cache_key
from .timing import Timing, probe_timing

PROXY_MAX_SIDE = 1600  # The proxy's long side. The view scales it on the GPU, so this sets detail
# A frame up to this many ahead is reached by reading forward rather than seeking. On the proxy a
# frame read forward (grab, not converted) costs about 0.7 ms and a seek about 40 ms however near,
# so reading forward wins across a whole keyframe interval (the proxy's is 30 frames)
MAX_READ_AHEAD = 30


class VideoSource(QObject):
    progress = Signal(float)   # Proxy encoding, 0.0 to 1.0
    proxy_ready = Signal()     # Reading has switched to the proxy
    failed = Signal(str)       # A message for the user
    _proxy_made = Signal(str, int)  # Worker to UI thread: the proxy's path and its load's number

    def __init__(self, cache_dir: str) -> None:
        super().__init__()
        self.cache_dir = cache_dir
        self.reader: VideoReader | None = None
        self.path: str | None = None
        self.proxy_path: str | None = None
        self.size = (0, 0)          # Frame size in proxy pixels (width, height)
        self.timing: Timing | None = None  # Read from the source file, so the proxy cannot change it
        self._from_source = False   # Frames come from the source (scaled) until the proxy exists
        self._job = 0               # Which load a finishing worker belongs to
        self._proxy_made.connect(self._use_proxy)  # Queued: runs on the UI thread

    @property
    def fcount(self) -> int:
        return self.reader.fcount if self.reader else 0

    @property
    def fps(self) -> float:
        """The container rate, exact (playback follows it)."""
        return self.timing.container_fps if self.timing else 0

    @property
    def measure_fps(self) -> float:
        """The rate that turns a frame into real time: the capture rate when the file has one."""
        return self.timing.measure_fps if self.timing else 0

    @property
    def fps_text(self) -> str:
        if self.timing and self.timing.capture_fps:
            return f"{self.fps:.2f} fps, recorded at {self.timing.capture_fps:g} fps"
        return f"{self.fps:.2f} fps"

    @property
    def has_proxy(self) -> bool:
        return self.reader is not None and not self._from_source

    def load(self, path: str) -> bool:
        """
        Opens path for reading at once (False, after emitting failed, if it cannot be read). The
        proxy follows: proxy_ready is emitted when frames come from it.
        """
        try:
            reader = VideoReader(path)
        except IOError as e:
            self.failed.emit(str(e))
            return False
        # Phone videos are stored sideways with a rotation flag; the proxy is upright, so the
        # source must be too. (media.VideoReader does not ask for this; set on its capture here.)
        reader._reader.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        h, w = reader.read(0).shape[:2]
        scale = min(1.0, PROXY_MAX_SIDE / max(w, h))
        width = max(2, round(w * scale / 2) * 2)  # Even, as libx264 needs
        self.size = (width, round(h * width / w))

        # The container rate from the file's own report (exact, not rounded down as VideoReader.fps
        # is); OpenCV's unrounded value if ffmpeg cannot be run
        self.timing = probe_timing(path) or Timing(float(reader._reader.get(cv2.CAP_PROP_FPS)))
        if self.reader is not None:
            self.reader.release()
        self.path, self.reader, self._from_source = path, reader, True
        self._job += 1

        os.makedirs(self.cache_dir, exist_ok=True)
        name = os.path.splitext(os.path.basename(path))[0]
        # Same cache key as the Tk app, so a changed source or encoder setting makes a new proxy
        out = os.path.join(self.cache_dir, f"{name}_{_cache_key(path, width, -2)}_proxy.mp4")
        if os.path.isfile(out):
            self._use_proxy(out, self._job)
        else:
            threading.Thread(target=self._make_proxy, args=(path, out, width, self.fcount,
                                                            self._job), daemon=True).start()
        return True

    def _make_proxy(self, path: str, out: str, width: int, total: int, job: int) -> None:
        # Made under a name of its own, then moved into place: two tabs opening the same video at once each make
        # one, and with one name (and one temporary file) the second's rename failed
        own = f"{os.path.splitext(out)[0]}.{os.getpid()}-{id(self)}-{job}.mp4"
        try:
            made = proxyvideo(path, width=width, writepath=own, on_progress=self.progress.emit,
                              total_frames=total)
            if made is not None:
                os.replace(made, out)
                made = out
        except Exception as e:  # Reported to the user rather than lost on the worker thread
            made = None
            self.failed.emit(f"Making the proxy failed: {e}")
        if made is not None:
            # Back on the UI thread through a queued signal; a newer load makes this one stale
            self._proxy_made.emit(made, job)

    def _use_proxy(self, out: str, job: int) -> None:
        if job != self._job:
            return
        self.reader.release()
        self.reader = VideoReader(out)
        self.proxy_path, self._from_source = out, False
        self.proxy_ready.emit()

    def frame(self, index: int):
        """The frame at index (BGR), in proxy pixels."""
        self._read_forward_to(index)
        frame = self.reader.read(index)
        if self._from_source:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        return frame

    def _read_forward_to(self, index: int) -> None:
        """Skips the frames before index by reading them, when index is a little ahead of the next
        frame, so that reading index needs no seek. Playback skips frames whenever one is slow to
        draw; seeking for each skip made the next frame slower still, and playback fell to about a
        third of its speed. (media.VideoReader, shared with the Tk app, seeks for any skip.)"""
        reader = self.reader
        if reader._next is None or not 0 < index - reader._next <= MAX_READ_AHEAD:
            return
        while reader._next < index and reader._reader.grab():
            reader._next += 1
