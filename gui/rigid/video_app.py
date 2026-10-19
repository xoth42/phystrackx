"""
videoapp.py

Defines the Video class which handles video display, frame manipulations, and tracking logic.

Author: Isam Balghari
"""

from typing import Optional, List
import os
import logging
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
        self.imgview = None
        self.tkimg = None

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
        frame = self.rigid.frame(index=self.frameidx)
        frame = self.filters.apply_filter(frame)
        self.frame = self.crop.apply_crop(frame)

        self.show_array(self.frame, view)

    def show_array(self, frame, view=None) -> None:
        """Draws a processed (cropped) frame on the canvas, through the zoom and pan view."""
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
            if view.zoom > 1:
                x, y, w, h = view.window()
                part = frame[y:y + h, x:x + w]
                if part.size == 0:
                    self.logger.warning("Skipping redraw: the zoom window is outside the frame")
                    return
                shown = cv2.resize(part, out, interpolation=cv2.INTER_LINEAR)
            elif fit != 1:
                shown = cv2.resize(frame, out, interpolation=cv2.INTER_LINEAR)

        img = Image.fromarray(cv2.cvtColor(shown.copy(), cv2.COLOR_BGR2RGB))
        self.tkimg = ImageTk.PhotoImage(image=img)

        self.canvas.coords(self.imgview, self.crop.cropx, self.crop.cropy)
        self.canvas.itemconfig(self.imgview, image=self.tkimg)

    def track(self, track_rects: Rect, ocr_rects: Rect, progress: IntVar, on_frame=None,
              should_stop=None, fullres: bool = False, on_tracking=None) -> None:
        """
        Perform object tracking on the video using selected regions.

        Args:
            track_rects (Rect): Region to track.
            ocr_rects (Rect): OCR target region.
            progress (IntVar): Variable for UI progress tracking.
            fullres (bool): Track at the source's full size. The full-size copy is made first
                if it does not exist yet; progress covers that step.
            on_tracking (callable, optional): Called once the tracker starts reading frames.
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

        self.rigid.track(self.frameidx, track_rects.rects, ocr_rects.rects, self.filters,
                         self.crop, progress, on_frame=on_frame, should_stop=should_stop,
                         fullres=source)

    def clear(self) -> None:
        """
        Clear stored tracking points from memory.
        """
        self.rigid.trackpts.clear()
        self.logger.info("Tracking points cleared from memory.")
