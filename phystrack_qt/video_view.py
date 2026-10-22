"""
The video view of the Qt spike: a QGraphicsView drawn with OpenGL.

Scene coordinates are frame pixels: the picture item sits at (0, 0) unscaled, and the view's
transform does all zooming and panning on the GPU. Overlays are children of the picture item, so
they stay on the same frame pixels at any zoom without coordinate code of their own.
"""

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

MAX_ZOOM = 8.0  # Times the fitted size, as in the Tk app


class VideoView(QGraphicsView):
    clicked = Signal(QPointF)     # A click on the picture while a tool wants one, in frame pixels
    zoom_changed = Signal(float)  # Zoom relative to the fitted size

    def __init__(self) -> None:
        super().__init__()
        self.setViewport(QOpenGLWidget())  # GPU drawing: about 9 ms a frame, vs about 26 ms on CPU
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#4d535c"))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)

        self.picture = QGraphicsPixmapItem()
        self.picture.setShapeMode(QGraphicsPixmapItem.ShapeMode.BoundingRectShape)
        self.scene().addItem(self.picture)
        self._fitted = True    # Follows the window's size until the user zooms
        self.wants_click = False

    # --- Picture ---------------------------------------------------------------------------------
    def show_frame(self, frame: np.ndarray) -> None:
        """Shows a BGR frame. The first one (or a new size) also fits the view to it."""
        h, w = frame.shape[:2]
        image = QImage(frame.data, w, h, frame.strides[0], QImage.Format.Format_BGR888)
        new_size = self.picture.pixmap().size().toTuple() != (w, h)
        self.picture.setPixmap(QPixmap.fromImage(image))  # Copies, so frame may be reused after
        if new_size:
            self.scene().setSceneRect(self.picture.boundingRect())
            self.fit()

    # --- Zoom and pan ----------------------------------------------------------------------------
    def fit(self) -> None:
        """The whole picture in the view (double-click, or a new video)."""
        if self.picture.pixmap().isNull():
            return
        self.fitInView(self.picture, Qt.AspectRatioMode.KeepAspectRatio)
        self._fitted = True
        self._update_sharpness()

    def _fit_scale(self) -> float:
        rect = self.picture.boundingRect()
        if rect.isEmpty():
            return 1.0
        return min(self.viewport().width() / rect.width(), self.viewport().height() / rect.height())

    def zoom(self) -> float:
        return self.transform().m11() / self._fit_scale()

    def zoom_by(self, factor: float) -> None:
        current = self.zoom()
        factor = max(1.0, min(MAX_ZOOM, current * factor)) / current
        if abs(factor - 1) < 1e-6:
            return
        self.scale(factor, factor)
        self._fitted = self.zoom() <= 1.0001
        self._update_sharpness()

    def _update_sharpness(self) -> None:
        """Enlarged pixels stay sharp (as in the Tk app); a shrunk picture is smoothed."""
        enlarged = self.transform().m11() * self.devicePixelRatioF() > 1
        self.picture.setTransformationMode(Qt.TransformationMode.FastTransformation if enlarged
                                           else Qt.TransformationMode.SmoothTransformation)
        self.zoom_changed.emit(self.zoom())

    def wheelEvent(self, event) -> None:
        # Cmd (Ctrl on other systems) + scroll zooms; plain scroll pans, as on a trackpad
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            self.zoom_by(1.0015 ** event.angleDelta().y())
        else:
            super().wheelEvent(event)

    def viewportEvent(self, event) -> bool:
        if event.type() == QEvent.Type.NativeGesture and \
                event.gestureType() == Qt.NativeGestureType.ZoomNativeGesture:
            self.zoom_by(1 + event.value())  # Trackpad pinch
            return True
        return super().viewportEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._fitted:
            self.fit()

    def mouseDoubleClickEvent(self, event) -> None:
        if self.itemAt(event.position().toPoint()) in (self.picture, None):
            self.fit()
        else:
            super().mouseDoubleClickEvent(event)

    # --- Clicks for tools ------------------------------------------------------------------------
    def set_wants_click(self, on: bool) -> None:
        """A tool waits for a click on the picture: no hand-drag panning meanwhile."""
        self.wants_click = on
        self.setDragMode(QGraphicsView.DragMode.NoDrag if on else
                         QGraphicsView.DragMode.ScrollHandDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor if on else Qt.CursorShape.OpenHandCursor)

    def mousePressEvent(self, event) -> None:
        if self.wants_click and event.button() == Qt.MouseButton.LeftButton:
            point = self.picture.mapFromScene(self.mapToScene(event.position().toPoint()))
            if self.picture.contains(point):
                self.clicked.emit(point)
                return
        super().mousePressEvent(event)
