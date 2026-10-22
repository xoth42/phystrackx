"""
Main window of the Qt spike: a left sidebar of tool buttons, the video view, and a seek bar with
trim controls under it. Experimental; the real app is the Tk one in the parent folder.

Tracking: Mark Object (drag a box, as many as wanted), then Track runs from the current frame to
the end of the trim (or of the video). The view follows the tracker live; scrubbing stays free.
"""

import os
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton, QSlider, QVBoxLayout,
                               QWidget)

from .axes_item import AxesItem
from .tracked_object import TrackedObject
from .tracking import TrackJob
from .video_source import VideoSource
from .video_view import TOOL_BOX, TOOL_CLICK, TOOL_NONE, VideoView

SIDEBAR_WIDTH = 170


class MainWindow(QMainWindow):
    def __init__(self, cache_dir: str) -> None:
        super().__init__()
        self.setWindowTitle("PhysTrackX (Qt spike)")
        self.source = VideoSource(cache_dir)
        self.source.progress.connect(self._on_progress)
        self.source.proxy_ready.connect(self._on_proxy_ready)
        self.source.failed.connect(self._on_failed)
        self.axes: AxesItem | None = None
        self.trim = None            # (start, end) frames once a trim is applied
        self._pending_frame = None  # Latest frame asked for by the seek bar, drawn once per batch
        self.objects: list[TrackedObject] = []
        self.job: TrackJob | None = None
        self._live_frame = None     # Latest frame the tracker finished, shown at most 30 times a s
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(33)
        self._live_timer.timeout.connect(self._follow_tracker)

        # Sidebar
        sidebar = QFrame()
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        side = QVBoxLayout(sidebar)
        self.import_btn = QPushButton("Import Video")
        self.trim_btn = QPushButton("Trim")
        self.axis_btn = QPushButton("Place Axis")
        self.mark_btn = QPushButton("Mark Object")
        self.track_btn = QPushButton("Track")
        for button, action in ((self.import_btn, self.choose_video),
                               (self.trim_btn, self.toggle_trim),
                               (self.axis_btn, self.start_axis),
                               (self.mark_btn, self.start_mark),
                               (self.track_btn, self.toggle_track)):
            button.setMinimumHeight(40)
            button.clicked.connect(action)
            side.addWidget(button)
        side.addStretch()
        self.video_buttons = (self.trim_btn, self.axis_btn, self.mark_btn, self.track_btn)
        for button in self.video_buttons:
            button.setEnabled(False)

        # Video view
        self.view = VideoView()
        self.view.clicked.connect(self._place_axes)
        self.view.box_drawn.connect(self._add_object)
        self.view.zoom_changed.connect(lambda z: self.zoom_label.setText(f"zoom {z:.2f}x"))

        # Seek bar, frame number, and the trim row (shown while trimming)
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setEnabled(False)
        self.seek.valueChanged.connect(self._seek_to)
        self.frame_label = QLabel("no video")
        self.frame_label.setMinimumWidth(130)
        seek_row = QHBoxLayout()
        seek_row.addWidget(self.seek, 1)
        seek_row.addWidget(self.frame_label)

        self.trim_row = QWidget()
        trim = QHBoxLayout(self.trim_row)
        trim.setContentsMargins(0, 0, 0, 0)
        self.trim_start_btn = QPushButton("Start here")
        self.trim_end_btn = QPushButton("End here")
        self.trim_label = QLabel()
        self.trim_label.setMinimumWidth(320)  # Room for the range and the warning
        apply_btn = QPushButton("Apply Trim")
        clear_btn = QPushButton("Whole Video")
        self.trim_start_btn.clicked.connect(lambda: self._mark_trim(0))
        self.trim_end_btn.clicked.connect(lambda: self._mark_trim(1))
        apply_btn.clicked.connect(self._apply_trim)
        clear_btn.clicked.connect(self._clear_trim)
        for w in (self.trim_start_btn, self.trim_end_btn, self.trim_label):
            trim.addWidget(w)
        trim.addStretch()
        trim.addWidget(clear_btn)
        trim.addWidget(apply_btn)
        self.trim_row.hide()
        self._trim_marks = [0, 0]

        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 8, 8)
        right.addWidget(self.view, 1)
        right.addLayout(seek_row)
        right.addWidget(self.trim_row)
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.addWidget(sidebar)
        body.addLayout(right, 1)
        central = QWidget()
        central.setLayout(body)
        self.setCentralWidget(central)

        # Status bar: messages on the left; load progress, zoom and draw time on the right
        self.progress = QProgressBar()
        self.progress.setMaximumWidth(200)
        self.progress.hide()
        self.zoom_label = QLabel()
        self.draw_label = QLabel()
        for w in (self.progress, self.zoom_label, self.draw_label):
            self.statusBar().addPermanentWidget(w)
        self.statusBar().showMessage("Import a video to start")

        # One frame at a time with the arrows; Escape ends a tool
        QShortcut(QKeySequence(Qt.Key.Key_Left), self, lambda: self._step(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self, lambda: self._step(1))
        QShortcut(QKeySequence(Qt.Key.Key_Escape), self, self._cancel_tool)
        QShortcut(QKeySequence("Ctrl+0"), self, self.view.fit)  # Cmd 0 on macOS, as in the Tk app

    # --- Import ----------------------------------------------------------------------------------
    def choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import Video", os.path.expanduser("~"),
                                              "Videos (*.mp4 *.mov *.m4v *.avi *.mkv);;All (*)")
        if path:
            self.load_video(path)

    def load_video(self, path: str) -> None:
        """Shows the video at once; scrubbing gets faster when the proxy is ready."""
        t0 = time.perf_counter()
        if not self.source.load(path):
            return
        self._on_opened()
        name = os.path.basename(path)
        took = (time.perf_counter() - t0) * 1000
        if self.source.has_proxy:
            self.statusBar().showMessage(f"{name}: {self.source.fcount} frames, "
                                         f"{self.source.fps} fps (opened in {took:.0f} ms)")
        else:
            self._proxy_since = time.perf_counter()
            self.progress.setValue(0)
            self.progress.show()
            self.statusBar().showMessage(f"{name} opened in {took:.0f} ms. Making a copy for fast "
                                         "scrubbing; you can look around meanwhile")

    def _on_progress(self, fraction: float) -> None:
        self.progress.setValue(round(fraction * 100))

    def _on_proxy_ready(self) -> None:
        self.progress.hide()
        if hasattr(self, "_proxy_since"):
            took = time.perf_counter() - self._proxy_since
            self.statusBar().showMessage(f"Fast scrubbing ready ({took:.1f} s)")
        self._show(self.seek.value())  # Same frame, now from the proxy

    def _on_opened(self) -> None:
        if self.job is not None:
            self.job.stop()
        for obj in self.objects:  # Objects belong to the old video
            obj.remove()
        self.objects.clear()
        self.trim = None
        self.trim_row.hide()
        if self.axes is not None:  # Axes belong to the old video
            self.view.scene().removeItem(self.axes)
            self.axes = None
        self.seek.blockSignals(True)
        self.seek.setRange(0, max(0, self.source.fcount - 1))
        self.seek.setValue(0)
        self.seek.blockSignals(False)
        self.seek.setEnabled(True)
        for button in self.video_buttons:
            button.setEnabled(True)
        self._show(0)

    def _on_failed(self, message: str) -> None:
        self.progress.hide()
        self.statusBar().clearMessage()
        QMessageBox.warning(self, "Import Video", message)

    # --- Seeking ---------------------------------------------------------------------------------
    def _seek_to(self, index: int) -> None:
        """Seek bar moved. A drag sends many moves; only the latest is drawn, once they are handled."""
        first = self._pending_frame is None
        self._pending_frame = index
        if first:
            QTimer.singleShot(0, self._draw_pending)

    def _draw_pending(self) -> None:
        index, self._pending_frame = self._pending_frame, None
        if index is not None:
            self._show(index)

    def _show(self, index: int) -> None:
        t0 = time.perf_counter()
        frame = self.source.frame(index)
        t1 = time.perf_counter()
        self.view.show_frame(frame)
        t2 = time.perf_counter()
        for obj in self.objects:
            obj.show_frame(index)
        self.frame_label.setText(f"frame {index} / {self.source.fcount - 1}")
        self.draw_label.setText(f"decode {(t1 - t0) * 1000:.0f} ms, upload {(t2 - t1) * 1000:.0f} ms")

    def _step(self, delta: int) -> None:
        if self.seek.isEnabled():
            self.seek.setValue(self.seek.value() + delta)  # Clamped to the (trimmed) range

    # --- Trim ------------------------------------------------------------------------------------
    # The spike only limits the seek bar to a range. The Tk app also writes a trimmed (and cropped)
    # clip with media.trimvideo; that comes later.
    def toggle_trim(self) -> None:
        if self.trim_row.isVisible():
            self.trim_row.hide()
            return
        self._cancel_tool()
        self._trim_marks = list(self.trim) if self.trim else [0, self.source.fcount - 1]
        self.seek.setRange(0, self.source.fcount - 1)  # The whole video while choosing
        self._show_trim_marks()
        self.trim_row.show()
        self.statusBar().showMessage("Move to the first frame and press Start here, then the last "
                                     "and press End here")

    def _mark_trim(self, which: int) -> None:
        self._trim_marks[which] = self.seek.value()
        self._show_trim_marks()

    def _show_trim_marks(self) -> None:
        start, end = self._trim_marks
        bad = start >= end
        self.trim_label.setText(f"frames {start} to {end}" + ("  (start must be before end)"
                                                               if bad else ""))

    def _apply_trim(self) -> None:
        start, end = self._trim_marks
        if start >= end:
            return
        self.trim = (start, end)
        self.seek.setRange(start, end)
        self.trim_row.hide()
        self.statusBar().showMessage(f"Trimmed to frames {start} to {end} ({end - start + 1} frames)")

    def _clear_trim(self) -> None:
        self.trim = None
        self.seek.setRange(0, self.source.fcount - 1)
        self.trim_row.hide()
        self.statusBar().showMessage("Whole video")

    # --- Axes ------------------------------------------------------------------------------------
    def start_axis(self) -> None:
        self.trim_row.hide()
        self.view.set_tool(TOOL_CLICK)
        self.statusBar().showMessage("Click on the picture to place the origin (Escape to cancel)")

    def _place_axes(self, point) -> None:
        self.view.set_tool(TOOL_NONE)
        if self.axes is None:
            size = self.view.picture.boundingRect()
            self.axes = AxesItem(0.3 * min(size.width(), size.height()), self.view.picture)
            self.axes.changed.connect(self._report_axes)
        self.axes.setPos(point)
        self._report_axes()

    def _report_axes(self) -> None:
        p = self.axes.pos()
        self.statusBar().showMessage(f"Axes: origin ({p.x():.1f}, {p.y():.1f}) px, "
                                     f"angle {self.axes.angle():.1f}°. Drag a line to move, the "
                                     "handle to rotate")

    def _cancel_tool(self) -> None:
        if self.view.tool != TOOL_NONE:
            self.view.set_tool(TOOL_NONE)
            self.statusBar().clearMessage()

    # --- Mark and track --------------------------------------------------------------------------
    def start_mark(self) -> None:
        self.trim_row.hide()
        self.view.set_tool(TOOL_BOX)
        self.statusBar().showMessage("Drag a box around the object (Escape when done)")

    def _add_object(self, box) -> None:
        """A box was dragged out. The tool stays on, so several objects can be marked in a row."""
        frame = self.seek.value()
        obj = TrackedObject(len(self.objects) + 1, box, frame, self.view.picture)
        self.objects.append(obj)
        self.statusBar().showMessage(f"Object {obj.number} marked on frame {frame}. Drag another, or "
                                     "Escape when done, then Track")

    def toggle_track(self) -> None:
        if self.job is not None:
            self.job.stop()
            return
        self._cancel_tool()
        if not self.objects:
            QMessageBox.information(self, "Track", "Mark an object first.")
            return
        if not self.source.has_proxy:
            QMessageBox.information(self, "Track", "The fast copy of the video is still being made; "
                                    "Track can start once it is ready.")
            return
        start = self.seek.value()
        end = self.trim[1] if self.trim else self.source.fcount - 1
        if start >= end:
            QMessageBox.information(self, "Track", "Move to a frame before the end first.")
            return
        # Each object starts from its box as shown on this frame (the marked box, or where the last
        # run had it here), so tracking can be restarted from any frame
        boxes = []
        for obj in self.objects:
            r = obj.rect.rect()
            boxes.append((r.x(), r.y(), r.width(), r.height()))
            obj.clear_track()
        self.job = TrackJob(self.source.proxy_path, boxes, start, end)
        self.job.tracked.connect(self._on_tracked)
        self.job.finished.connect(self._on_track_done)
        self.job.failed.connect(self._on_track_failed)
        self._track_t0, self._track_range = time.perf_counter(), (start, end)
        self.track_btn.setText("Stop")
        self.mark_btn.setEnabled(False)
        self._live_timer.start()
        self.job.run()

    def _on_tracked(self, index: int, boxes: list) -> None:
        for obj, box in zip(self.objects, boxes):
            obj.add(index, box)
        self._live_frame = index

    def _follow_tracker(self) -> None:
        """Shows the tracker's latest frame, at most 30 times a second."""
        if self._live_frame is None:
            return
        index, self._live_frame = self._live_frame, None
        for obj in self.objects:
            obj.draw_trail()
        self.seek.setValue(index)
        start, end = self._track_range
        self.statusBar().showMessage(f"Tracking: frame {index} of {start} to {end}")

    def _on_track_done(self, completed: bool) -> None:
        self._end_track()
        start, end = self._track_range
        done = max((max(o.boxes) for o in self.objects if o.boxes), default=start) - start
        took = time.perf_counter() - self._track_t0
        lost = sum(1 for o in self.objects for b in o.boxes.values() if b is None)
        self.statusBar().showMessage(
            f"{'Tracked' if completed else 'Stopped after'} {done} frames in {took:.1f} s "
            f"({done / max(took, 1e-6):.0f} frames/s), {len(self.objects)} object(s), "
            f"{lost} lost frame(s)")

    def _on_track_failed(self, message: str) -> None:
        self._end_track()
        QMessageBox.warning(self, "Track", message)

    def _end_track(self) -> None:
        self._live_timer.stop()
        self._follow_tracker()  # The last frame and the whole trail
        self.job = None
        self.track_btn.setText("Track")
        self.mark_btn.setEnabled(True)
