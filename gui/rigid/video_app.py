"""
videoapp.py

Defines the Video class which handles video display, frame manipulations, and tracking logic.

Author: Isam Balghari
"""

from typing import Optional, List
import os
import logging
import threading
import cv2
from PIL import Image, ImageTk

from customtkinter import CTkCanvas, IntVar
from experiments.components import OCRData
from experiments.rigid.rigid import Rigid, FullResSource
from gui.plugins.crop import Crop
from gui.plugins.filters import Filters
from gui.components.rects.rect import Rect
from core import filexists


class Video:
    """
    Video handler for loading, displaying, trimming, and tracking frames on a canvas.
    """

    def __init__(self, canvas: CTkCanvas, vwidth: int, vheight: int, crop: Crop, filters: Filters) -> None:
        """
        Initialize the Video app.

        Args:
            canvas (CTkCanvas): Canvas to render video frames.
            vwidth (int): Width of the video display area.
            vheight (int): Height of the video display area.
            crop (Crop): Crop handler.
            filters (Filters): Filters to apply on video.
        """
        self.canvas = canvas
        self.vwidth = vwidth
        self.vheight = vheight
        self.crop = crop
        self.filters = filters

        self.frame: Optional[any] = None
        self.on_native_ready = None  # Called (from a worker thread) once the full-resolution copy exists
        self._native_busy = False
        self.imgview = None
        self.tkimg = None
        self._raw = None       # Last decoded frame and its index: panning and zooming do not
        self._raw_idx = None   # change the frame, so it is not decoded again

        tempdir = "temp"
        os.makedirs(tempdir, exist_ok=True)
        self.trimpath = os.path.join(tempdir, "Track_Rigid.mp4")

        self.rigid = Rigid(trimpath=self.trimpath, vwidth=self.vwidth, vheight=self.vheight-50)
        self.frameidx = 0
        # Where the trimmed clip sits in the proxy: first proxy frame, and crop origin in proxy pixels
        self.trimstart = 0
        self.trimorigin = (0, 0)
        self.istrimmed = False  # True while the loaded clip is a trim (not the full proxy)

        self.logger = logging.getLogger(__name__)
        self.logger.info("Video App initialized")
        self.trimvideo = self.rigid.trim
        
        self.imgview = self.canvas.create_image(self.crop.fx, self.crop.fy, anchor="nw")

    @property
    def fcount(self) -> int:
        """Total number of frames in loaded video."""
        return self.rigid.fcount

    @property
    def trackpts(self) -> List[List[List[float]]]:
        """Tracking points recorded from video."""
        return self.rigid.trackpts
    
    @property
    def ocrdata(self) -> OCRData:
        """OCR data extracted from video"""
        return self.rigid.texts

    @property
    def fps(self) -> int:
        """Video frames per second."""
        return self.rigid.fps

    @property
    def fwidth(self) -> int:
        """Current video frame width."""
        return self.rigid.fwidth

    @property
    def fheight(self) -> int:
        """Current video frame height."""
        return self.rigid.fheight

    def set_size(self, fwidth: int, fheight: int) -> None:
        """
        Set the size of the video frame.

        Args:
            fwidth (int): New width for the video frame.
            fheight (int): New height for the video frame.
        """
        self.rigid.set(fwidth, fheight)

    def loadvideo(self, videopath: str, istrim:bool=False, on_progress=None) -> None:
        """
        Load a video from file or fallback to trimmed path.

        Args:
            videopath (str): Path to video file.
            on_progress (callable, optional): Receives proxy encoding progress (0.0 to 1.0).
        """
        self.rigid.release()
        self._raw = self._raw_idx = None  # A new clip: the cached frame is no longer valid

        if istrim:
            self.logger.info("Loading trim video")
            if not filexists(self.trimpath):
                self.logger.error("Trim video not found!")
                return
            self.rigid.addvideo(self.trimpath, istrim, self.crop.fwidth, self.crop.fheight)
            self.istrimmed = True
            self.logger.info("Video added from: %s", self.trimpath)
        else:
            self.set_trim_map(0, (0, 0))
            self.istrimmed = False
            self.crop.fitted = False  # The full video is shown at its own size
            self.rigid.addvideo(videopath, on_progress=on_progress)
            self.logger.info("Video added from: %s", videopath)
            self.crop.set(self.fwidth, self.fheight)

        self.logger.info(f"Video loaded: {videopath}, Frame count: {self.fcount}, FPS: {self.fps},"
                         f"Width: {self.crop.cropwidth}, Height: {self.crop.cropheight}")


    def set_trim_map(self, start: int, origin: tuple[int, int]) -> None:
        """
        Records where the trimmed clip sits in the proxy, so full-resolution tracking reads the
        same frames and the same area.

        Args:
            start (int): Proxy frame that the clip's first frame comes from.
            origin (tuple[int, int]): Top-left of the crop in proxy pixels (0, 0 if not cropped).
        """
        self.trimstart = start
        self.trimorigin = origin

    def showframe(self, idx: int, view=None) -> None:
        """
        Fetch and display a video frame on the canvas.

        Args:
            idx (int): Index of the frame to display.
            view (ViewTransform, optional): Zoom and pan. When zoomed in, only the visible
                window of the frame is shown, stretched to fill the frame area.
        """
        self.frameidx = idx
        if self._raw is None or self._raw_idx != idx:
            self._raw = self.rigid.frame(index=self.frameidx)
            self._raw_idx = idx
        frame = self.filters.apply_filter(self._raw)
        self.frame = self.crop.apply_crop(frame)

        native = self._native_picture(view)
        self.show_array(self.frame, view, native=native)

    def _native_picture(self, view) -> Optional[any]:
        """
        When the view enlarges the picture, the shown pixels come from the full-resolution copy
        (the proxy has fewer pixels, so enlarging it looks blurred). Returns the visible part of the
        frame at full resolution, or None to show the proxy. The copy is made in the background
        the first time it is needed; until then the proxy is shown.
        """
        if view is None or view.zoom * view.fit <= 1:
            return None
        if self.rigid.nativepath is None:
            self._prepare_native()
            return None
        frame_h, frame_w = self.frame.shape[:2]
        if view.zoom > 1:
            x, y, w, h = view.window()
        else:
            x, y, w, h = 0, 0, frame_w, frame_h
        # The clip is cut from the proxy: its frame i is source frame trimstart + i (when trimmed),
        # and its pixels start at trimorigin in the proxy, plus any crop shown on top of it
        ox, oy = self.trimorigin
        if self.crop.croprect is not None:
            ox, oy = ox + self.crop.croprect.xmin, oy + self.crop.croprect.ymin
        source_frame = self.frameidx + (self.trimstart if self.istrimmed else 0)
        if not self.istrimmed and self.rigid.active_duration:
            source_frame = self.rigid.active_duration[self.frameidx]
        part = self.rigid.native_region(source_frame, (ox + x, oy + y, w, h))
        if part is None:
            return None
        return self.filters.apply_filter(part)

    def _prepare_native(self) -> None:
        """Builds the full-resolution copy on a worker thread, then tells the app it is ready."""
        if self._native_busy:
            return
        self._native_busy = True

        def work():
            try:
                self.rigid.ensure_native()
            except Exception:
                self.logger.exception("Could not make the full-resolution copy for zoomed views")
            finally:
                self._native_busy = False
            if self.on_native_ready is not None and self.rigid.nativepath is not None:
                self.on_native_ready()

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _scale(img, size: tuple[int, int]):
        """Resizes without smoothing when enlarging (pixels stay sharp, like macOS zoom), and
        averages when shrinking (so detail is not lost to aliasing)."""
        enlarging = size[0] >= img.shape[1] and size[1] >= img.shape[0]
        return cv2.resize(img, size, interpolation=cv2.INTER_NEAREST if enlarging else cv2.INTER_AREA)

    def show_array(self, frame, view=None, native=None) -> None:
        """Draws a processed (cropped) frame on the canvas, through the zoom and pan view.

        native: the visible part of the frame at full resolution (see _native_picture), which is
        drawn instead of the proxy's pixels.
        """
        # An empty frame comes from a clip that failed to load. Drawing it raises inside the Tk
        # callback, so skip this redraw and keep the last picture on screen.
        if frame is None or frame.size == 0:
            self.logger.warning("Skipping redraw: the frame is empty")
            return

        # The picture is drawn at the frame's size times the fit (so a cropped frame fills the view)
        fit = view.fit if view is not None else 1.0
        out = (max(1, round(frame.shape[1] * fit)), max(1, round(frame.shape[0] * fit)))
        shown = frame
        if view is not None:
            view.configure(view.origin, (frame.shape[1], frame.shape[0]))
        if native is not None and native.size > 0:
            shown = self._scale(native, out)
        elif view is not None:
            if view.zoom > 1:
                x, y, w, h = view.window()
                part = frame[y:y + h, x:x + w]
                if part.size == 0:
                    self.logger.warning("Skipping redraw: the zoom window is outside the frame")
                    return
                shown = self._scale(part, out)
            elif fit != 1:
                shown = self._scale(frame, out)

        img = Image.fromarray(cv2.cvtColor(shown.copy(), cv2.COLOR_BGR2RGB))
        self.tkimg = ImageTk.PhotoImage(image=img)  # Making a new one is faster than paste()

        self.canvas.coords(self.imgview, self.crop.cropx, self.crop.cropy)
        self.canvas.itemconfig(self.imgview, image=self.tkimg)

    def track(self, track_rects: Rect, ocr_rects: Rect, progress: IntVar, on_frame=None,
              should_stop=None, fullres: bool = False, on_tracking=None, filters=None,
              assist: bool = False) -> None:
        """
        Perform object tracking on the video using selected regions.

        Args:
            track_rects (Rect): Region to track.
            ocr_rects (Rect): OCR target region.
            progress (IntVar): Variable for UI progress tracking.
            fullres (bool): Track at the source's full size. The full-size copy is made first
                if it does not exist yet; progress covers that step.
            on_tracking (callable, optional): Called once the tracker starts reading frames.
            filters (optional): The filter to apply, taken on the UI thread (Filters.snapshot()).
                This runs on a worker thread, where the live Filters plugin (Tk variables) must
                not be read.
            assist (bool): Use Segment Anything on hard frames (see experiments/rigid/assist.py).
        """
        self.logger.info("Starting tracking at frame index: %d", self.frameidx)
        self.logger.info("Tracking rectangles: ")
        for rect in track_rects.rects:
            pixelrect = rect.norm2pix(self.crop.cropwidth, self.crop.cropheight)
            self.logger.info("Track Rect: (x=%.02f, y=%.02f, w=%.02f, h=%.02f)", pixelrect.xmin,
                             pixelrect.ymin, pixelrect.width, pixelrect.height)

        self.logger.info("OCR rectangles: ")
        for rect in ocr_rects.rects:
            pixelrect = rect.norm2pix(self.crop.cropwidth, self.crop.cropheight)
            self.logger.info("OCR Rect: (x=%.02f, y=%.02f, w=%.02f, h=%.02f)", pixelrect.xmin,
                             pixelrect.ymin, pixelrect.width, pixelrect.height)

        source = None
        if fullres:
            path = self.rigid.ensure_native(on_progress=lambda f: progress.set(int(f * 100)))
            source = FullResSource(path, start=self.trimstart, origin=self.trimorigin)
        if on_tracking is not None:
            on_tracking()

        engine = None
        if assist:
            from media.segmenter import default_segmenter
            from experiments.rigid.assist import HardEngine
            segmenter = default_segmenter()
            if segmenter is None:
                self.logger.warning("Assist is on, but the Segment Anything model is not installed "
                                    "(run phystrackx.py --download-models); tracking without it.")
            else:
                engine = HardEngine(segmenter)
        self.rigid.track(self.frameidx, track_rects.rects, ocr_rects.rects,
                         filters if filters is not None else self.filters,
                         self.crop, progress, on_frame=on_frame, should_stop=should_stop,
                         fullres=source, assist=engine)

    def clear(self) -> None:
        """
        Clear stored tracking points from memory.
        """
        self.rigid.trackpts.clear()
        self.logger.info("Tracking points cleared from memory.")
