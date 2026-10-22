"""
Main window of the Qt spike: a left sidebar of tool buttons, the video view, and a seek bar with
trim controls under it. Experimental; the real app is the Tk one in the parent folder.
"""

import os
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QFileDialog, QFrame, QHBoxLayout, QLabel, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton, QSlider, QVBoxLayout,
                               QWidget)

from .axes_item import AxesItem
from .video_source import VideoSource
from .video_view import VideoView

SIDEBAR_WIDTH = 170


class MainWindow(QMainWindow):
    def __init__(self, cache_dir: str) -> None:
        super().__init__()
        self.setWindowTitle("PhysTrackX (Qt spike)")
        self.source = VideoSource(cache_dir)
        self.source.progress.connect(self._on_progress)
        self.source.loaded.connect(self._on_loaded)
        self.source.failed.connect(self._on_failed)
        self.axes: AxesItem | None = None
        self.trim = None            # (start, end) frames once a trim is applied
        self._pending_frame = None  # Latest frame asked for by the seek bar, drawn once per batch

        # Sidebar
        sidebar = QFrame()
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        side = QVBoxLayout(sidebar)
        self.import_btn = QPushButton("Import Video")
        self.trim_btn = QPushButton("Trim")
        self.axis_btn = QPushButton("Place Axis")
        for button, action in ((self.import_btn, self.choose_video),
                               (self.trim_btn, self.toggle_trim),
                               (self.axis_btn, self.start_axis)):
            button.setMinimumHeight(40)
            button.clicked.connect(action)
            side.addWidget(button)
        side.addStretch()
        self.trim_btn.setEnabled(False)
        self.axis_btn.setEnabled(False)

        # Video view
        self.view = VideoView()
        self.view.clicked.connect(self._place_axes)
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
        self._loading_since = time.perf_counter()
        self.import_btn.setEnabled(False)
        self.progress.setValue(0)
        self.progress.show()
        self.statusBar().showMessage(f"Preparing {os.path.basename(path)}…")
        self.source.load(path)

    def _on_progress(self, fraction: float) -> None:
        self.progress.setValue(round(fraction * 100))

    def _on_loaded(self, path: str) -> None:
        self.source.open_proxy(path)
        self.progress.hide()
        self.import_btn.setEnabled(True)
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
        self.trim_btn.setEnabled(True)
        self.axis_btn.setEnabled(True)
        self._show(0)
        took = time.perf_counter() - self._loading_since
        self.statusBar().showMessage(f"{os.path.basename(path)}: {self.source.fcount} frames, "
                                     f"{self.source.fps} fps (ready in {took:.1f} s)")

    def _on_failed(self, message: str) -> None:
        self.progress.hide()
        self.import_btn.setEnabled(True)
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
        self.view.set_wants_click(True)
        self.statusBar().showMessage("Click on the picture to place the origin (Escape to cancel)")

    def _place_axes(self, point) -> None:
        self.view.set_wants_click(False)
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
        if self.view.wants_click:
            self.view.set_wants_click(False)
            self.statusBar().clearMessage()
