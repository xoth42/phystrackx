"""
Coordinate axes overlay for the Qt spike. A child of the picture item, so its position is the
origin in frame pixels and it follows zoom and pan by itself.

Drag either line to move the axes; drag the round handle at the end of the x axis to rotate them.
Lines are cosmetic (a fixed width on screen at any zoom).
"""

import math

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsItem, QGraphicsObject

X_COLOR = QColor("#ff4d4d")
Y_COLOR = QColor("#4dff88")
GRAB_WIDTH = 14  # Frame pixels either side of a line that pick it up


class AxesItem(QGraphicsObject):
    changed = Signal()  # Moved or rotated

    def __init__(self, length: float, parent: QGraphicsItem) -> None:
        super().__init__(parent)
        self.length = length
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.handle = _RotateHandle(self)
        self.handle.setPos(length, 0)

    # Screen convention: x to the right, y up (so drawn towards negative item y)
    def _lines(self) -> tuple[QLineF, QLineF]:
        L = self.length
        return QLineF(-L * 0.25, 0, L, 0), QLineF(0, L * 0.25, 0, -L)

    def boundingRect(self) -> QRectF:
        L = self.length
        return QRectF(-L * 0.25, -L, L * 1.25, L * 1.25).adjusted(-40, -40, 40, 40)

    def shape(self) -> QPainterPath:
        """Only the lines pick up the mouse, so a click beside them still pans the picture."""
        path = QPainterPath()
        for line in self._lines():
            path.moveTo(line.p1())
            path.lineTo(line.p2())
        stroker = QPainterPathStroker()
        stroker.setWidth(GRAB_WIDTH * 2)
        return stroker.createStroke(path)

    def paint(self, painter, option, widget=None) -> None:
        x_line, y_line = self._lines()
        for line, color, label in ((x_line, X_COLOR, "x"), (y_line, Y_COLOR, "y")):
            pen = QPen(color, 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLine(line)
            painter.setFont(QFont("Helvetica", max(12, int(self.length / 8)), QFont.Weight.Bold))
            end = line.p2()
            painter.drawText(QRectF(end.x() + 10, end.y() - 60, 80, 60), label)
        pen = QPen(QColor("white"), 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawEllipse(QPointF(0, 0), 5, 5)

    def angle(self) -> float:
        """The x axis' angle in degrees, counter-clockwise from the picture's x (physics sense)."""
        return -self.rotation()

    def itemChange(self, change, value):
        if change in (QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged,
                      QGraphicsItem.GraphicsItemChange.ItemRotationHasChanged):
            self.changed.emit()
        return super().itemChange(change, value)


class _RotateHandle(QGraphicsEllipseItem):
    """The round handle at the end of the x axis. Constant size on screen."""

    RADIUS = 7

    def __init__(self, axes: AxesItem) -> None:
        r = self.RADIUS
        super().__init__(-r, -r, 2 * r, 2 * r, axes)
        self.axes = axes
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setBrush(QBrush(X_COLOR))
        self.setPen(QPen(QColor("white"), 1.5))
        self.setCursor(Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, event) -> None:
        event.accept()  # Without this the press goes to the axes, which would move instead

    def mouseMoveEvent(self, event) -> None:
        # The angle from the origin to the pointer, in the picture's (parent's) pixels
        parent = self.axes.parentItem()
        p = parent.mapFromScene(event.scenePos())
        o = self.axes.pos()
        self.axes.setRotation(math.degrees(math.atan2(p.y() - o.y(), p.x() - o.x())))
