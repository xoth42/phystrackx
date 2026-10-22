"""
A marked object for the Qt spike: its box on the picture, its tracked boxes by frame, and the trail
of its centre. All in proxy pixels, as children of the picture item, so they follow zoom and pan.
"""

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainterPath, QPen
from PySide6.QtWidgets import (QGraphicsItem, QGraphicsPathItem, QGraphicsRectItem,
                               QGraphicsSimpleTextItem)

COLORS = ["#ffd400", "#00e5ff", "#ff6bd6", "#7cff4d", "#ff9f1a"]


def _pen(color: str, width: float, style=Qt.PenStyle.SolidLine) -> QPen:
    pen = QPen(QColor(color), width, style)
    pen.setCosmetic(True)  # The same width on screen at any zoom
    return pen


class TrackedObject:
    def __init__(self, number: int, box: QRectF, frame: int, picture: QGraphicsItem) -> None:
        self.number = number
        self.color = COLORS[(number - 1) % len(COLORS)]
        self.marked = (box.x(), box.y(), box.width(), box.height())  # Where tracking starts
        self.marked_frame = frame
        self.boxes: dict[int, tuple | None] = {}  # Frame -> tracked box, or None where lost

        self.rect = QGraphicsRectItem(box, picture)
        self.rect.setPen(_pen(self.color, 2))
        self.rect.setZValue(2)
        self.label = QGraphicsSimpleTextItem(str(number), self.rect)
        self.label.setBrush(QColor(self.color))
        self.label.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.trail = QGraphicsPathItem(picture)
        self.trail.setPen(_pen(self.color, 1.5))
        self.trail.setZValue(1)
        self._place_label()

    def _place_label(self) -> None:
        """Inside the box's top-left corner (the label keeps its size on screen)."""
        r = self.rect.rect()
        self.label.setPos(r.x() + 1, r.y() + 1)

    def clear_track(self) -> None:
        self.boxes.clear()
        self.trail.setPath(QPainterPath())

    def add(self, frame: int, box) -> None:
        """A tracked result. The trail is rebuilt from all frames, in order, by draw_trail."""
        self.boxes[frame] = box

    def draw_trail(self) -> None:
        path = QPainterPath()
        pen_down = False
        for frame in sorted(self.boxes):
            box = self.boxes[frame]
            if box is None:
                pen_down = False  # A gap where the object was lost
                continue
            centre = QPointF(box[0] + box[2] / 2, box[1] + box[3] / 2)
            if pen_down:
                path.lineTo(centre)
            else:
                path.moveTo(centre)
                pen_down = True
        self.trail.setPath(path)

    def show_frame(self, frame: int) -> None:
        """The box where the object is in this frame: tracked, or as marked before tracking."""
        if self.boxes:
            box = self.boxes.get(frame)
            lost = frame in self.boxes and box is None
            self.rect.setVisible(box is not None)
            if box is not None:
                self.rect.setRect(QRectF(*box))
            self.rect.setPen(_pen(self.color, 2, Qt.PenStyle.DotLine if lost else
                                  Qt.PenStyle.SolidLine))
        else:
            self.rect.setVisible(True)
            self.rect.setRect(QRectF(*self.marked))
        self._place_label()

    def centre(self, frame: int) -> tuple[float, float] | None:
        box = self.boxes.get(frame)
        return None if box is None else (box[0] + box[2] / 2, box[1] + box[3] / 2)

    def remove(self) -> None:
        scene = self.rect.scene()
        if scene is not None:
            scene.removeItem(self.rect)
            scene.removeItem(self.trail)
