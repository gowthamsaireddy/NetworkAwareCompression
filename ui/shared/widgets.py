"""Reusable custom PyQt6 widgets adhering to Zero-Dummy-Data policy."""

from pathlib import Path
from typing import Optional
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class SectionHeader(QWidget):
    """Header widget with a title and descriptive subtitle."""

    def __init__(self, title: str, subtitle: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 10)
        layout.setSpacing(2)

        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("ViewTitle")

        self.subtitle_lbl = QLabel(subtitle)
        self.subtitle_lbl.setObjectName("ViewSubtitle")

        layout.addWidget(self.title_lbl)
        layout.addWidget(self.subtitle_lbl)


class MetricBox(QFrame):
    """Card widget displaying a single measured metric value and label.

    Strict zero-dummy policy: default text is 'No data available' or 'Not calculated'.
    """

    def __init__(self, label: str, default_value: str = "No data available", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("MetricBox")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        self.label_lbl = QLabel(label.upper())
        self.label_lbl.setObjectName("MetricLabel")

        self.val_lbl = QLabel(default_value)
        self.val_lbl.setObjectName("MetricValue")

        layout.addWidget(self.label_lbl)
        layout.addWidget(self.val_lbl)

    def set_value(self, value: str) -> None:
        self.val_lbl.setText(value)


class ResizableImageLabel(QLabel):
    """Custom QLabel that retains the original uncompressed QPixmap and automatically rescales

    with KeepAspectRatio and SmoothTransformation on every resize event, preventing distortion,
    clipping, or accidental zooming.
    """

    def __init__(self, placeholder: str = "", parent: Optional[QWidget] = None):
        super().__init__(placeholder, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._orig_pixmap: Optional[QPixmap] = None
        self._placeholder: str = placeholder

    def set_original_pixmap(self, pixmap: Optional[QPixmap]) -> None:
        self._orig_pixmap = pixmap
        if pixmap is None or pixmap.isNull():
            self.clear()
            self.setText(self._placeholder)
        else:
            self.setText("")
            self._rescale()

    def clear_pixmap(self, message: Optional[str] = None) -> None:
        self._orig_pixmap = None
        self.clear()
        self.setText(message or self._placeholder)

    def _rescale(self) -> None:
        if self._orig_pixmap and not self._orig_pixmap.isNull():
            sz = self.size()
            # If label geometry has settled, scale to fit preserving aspect ratio
            if sz.width() > 10 and sz.height() > 10:
                scaled = self._orig_pixmap.scaled(
                    sz,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                super().setPixmap(scaled)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rescale()

    def sizeHint(self) -> QSize:
        return QSize(256, 256)

    def minimumSizeHint(self) -> QSize:
        return QSize(160, 160)


class ImagePreviewBox(QFrame):
    """Dedicated widget for displaying ultrasound images or placeholders with responsive scaling."""

    def __init__(self, title: str, placeholder: str = "No data available", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("CardFrame")
        self.setMinimumHeight(260)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("CardTitle")
        self.title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_lbl = ResizableImageLabel(placeholder)
        self.image_lbl.setStyleSheet(
            "background-color: #0b1120; border: 1px dashed #334155; border-radius: 6px; color: #64748b; font-style: italic;"
        )
        self.image_lbl.setMinimumSize(180, 180)
        self.image_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.caption_lbl = QLabel("")
        self.caption_lbl.setObjectName("MetricLabel")
        self.caption_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout.addWidget(self.title_lbl)
        layout.addWidget(self.image_lbl, stretch=1)
        layout.addWidget(self.caption_lbl)

    def set_pixmap(self, pixmap: QPixmap, caption: str = "") -> None:
        self.image_lbl.set_original_pixmap(pixmap)
        self.caption_lbl.setText(caption)

    def clear_image(self, message: str = "No data available") -> None:
        self.image_lbl.clear_pixmap(message)
        self.caption_lbl.setText("")

