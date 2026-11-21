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
    """Card widget displaying measured metric values, with primary and secondary details.

    Strict zero-dummy policy: default text is 'No data available' or 'Not calculated'.
    Auto-adjusts font size and wraps text to prevent truncation on any display resolution.
    """

    def __init__(
        self,
        label: str,
        default_value: str = "No data available",
        subtitle: str = "",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setObjectName("MetricBox")
        self.setMinimumHeight(78)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)

        self.label_lbl = QLabel(label.upper())
        self.label_lbl.setObjectName("MetricLabel")

        self.val_lbl = QLabel(default_value)
        self.val_lbl.setObjectName("MetricValue")
        self.val_lbl.setWordWrap(True)
        self._apply_val_font(default_value)

        self.sub_lbl = QLabel(subtitle)
        self.sub_lbl.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: 500;")
        self.sub_lbl.setWordWrap(True)
        if not subtitle:
            self.sub_lbl.hide()

        layout.addWidget(self.label_lbl)
        layout.addWidget(self.val_lbl)
        layout.addWidget(self.sub_lbl)

    def _apply_val_font(self, text: str, color: str = "#38bdf8") -> None:
        if len(text) > 22:
            self.val_lbl.setStyleSheet(f"font-size: 14px; font-weight: 700; color: {color};")
        elif len(text) > 13:
            self.val_lbl.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {color};")
        else:
            self.val_lbl.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {color};")

    def set_value(self, value: str, subtitle: Optional[str] = None, color: str = "#38bdf8") -> None:
        self.val_lbl.setText(value)
        self._apply_val_font(value, color=color)
        if subtitle is not None:
            self.sub_lbl.setText(subtitle)
            self.sub_lbl.setVisible(bool(subtitle))


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
        return QSize(140, 140)


class ImagePreviewBox(QFrame):
    """Dedicated widget for displaying ultrasound images or placeholders with responsive scaling.

    Guarantees that the caption label has its own dedicated physical footer container separated from
    the image canvas, preventing any text-over-image overlap.
    """

    def __init__(self, title: str, placeholder: str = "No data available", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("CardFrame")
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        self.title_lbl = QLabel(title)
        self.title_lbl.setObjectName("CardTitle")
        self.title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.image_lbl = ResizableImageLabel(placeholder)
        self.image_lbl.setStyleSheet(
            "background-color: #0b1120; border: 1px dashed #334155; border-radius: 6px; color: #64748b; font-style: italic;"
        )
        self.image_lbl.setMinimumSize(160, 160)
        self.image_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Dedicated footer badge container so caption NEVER overlays the image canvas
        self.caption_lbl = QLabel("")
        self.caption_lbl.setObjectName("ImageCaptionLabel")
        self.caption_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption_lbl.setWordWrap(True)
        self.caption_lbl.setStyleSheet(
            "QLabel#ImageCaptionLabel {"
            "    background-color: #0b1120;"
            "    border: 1px solid #1e293b;"
            "    border-radius: 4px;"
            "    padding: 3px 8px;"
            "    color: #38bdf8;"
            "    font-size: 11px;"
            "    font-weight: 600;"
            "    min-height: 22px;"
            "}"
        )

        layout.addWidget(self.title_lbl)
        layout.addWidget(self.image_lbl, stretch=1)
        layout.addWidget(self.caption_lbl)

    def set_pixmap(self, pixmap: QPixmap, caption: str = "") -> None:
        self.image_lbl.set_original_pixmap(pixmap)
        self.caption_lbl.setText(caption)

    def clear_image(self, message: str = "No data available") -> None:
        self.image_lbl.clear_pixmap(message)
        self.caption_lbl.setText("")
