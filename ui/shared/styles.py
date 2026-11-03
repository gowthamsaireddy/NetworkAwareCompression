"""Professional styling and color palettes for PyQt6 applications."""

APP_STYLESHEET = """
QMainWindow {
    background-color: #0f172a;
}

QWidget {
    color: #e2e8f0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}

/* Sidebar navigation */
QListWidget#SidebarList {
    background-color: #1e293b;
    border: none;
    border-right: 1px solid #334155;
    outline: none;
    padding-top: 10px;
}

QListWidget#SidebarList::item {
    height: 44px;
    padding-left: 16px;
    border-radius: 6px;
    margin: 3px 8px;
    color: #94a3b8;
    font-weight: 500;
}

QListWidget#SidebarList::item:hover {
    background-color: #334155;
    color: #f8fafc;
}

QListWidget#SidebarList::item:selected {
    background-color: #2563eb;
    color: #ffffff;
    font-weight: 600;
}

/* Card / Box containers */
QFrame#CardFrame {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 16px;
}

QFrame#MetricBox {
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 12px;
}

/* Headings */
QLabel#ViewTitle {
    font-size: 20px;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 4px;
}

QLabel#ViewSubtitle {
    font-size: 13px;
    color: #94a3b8;
    margin-bottom: 12px;
}

QLabel#CardTitle {
    font-size: 15px;
    font-weight: 600;
    color: #cbd5e1;
    margin-bottom: 8px;
}

QLabel#MetricValue {
    font-size: 22px;
    font-weight: 700;
    color: #38bdf8;
}

QLabel#MetricLabel {
    font-size: 12px;
    font-weight: 500;
    color: #94a3b8;
    text-transform: uppercase;
}

QLabel#StatusBadge {
    padding: 4px 10px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 11px;
}

/* Buttons */
QPushButton {
    background-color: #2563eb;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 600;
    min-height: 20px;
}

QPushButton:hover {
    background-color: #1d4ed8;
}

QPushButton:pressed {
    background-color: #1e40af;
}

QPushButton:disabled {
    background-color: #334155;
    color: #64748b;
}

QPushButton#SecondaryBtn {
    background-color: #334155;
    color: #e2e8f0;
}

QPushButton#SecondaryBtn:hover {
    background-color: #475569;
}

QPushButton#DangerBtn {
    background-color: #dc2626;
    color: #ffffff;
}

QPushButton#DangerBtn:hover {
    background-color: #b91c1c;
}

/* Inputs */
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 6px 10px;
    color: #f8fafc;
    min-height: 22px;
}

QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #38bdf8;
}

/* Tables */
QTableWidget {
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    gridline-color: #1e293b;
    color: #e2e8f0;
}

QTableWidget::item {
    padding: 6px;
}

QTableWidget::item:selected {
    background-color: #1d4ed8;
    color: #ffffff;
}

QHeaderView::section {
    background-color: #1e293b;
    color: #94a3b8;
    padding: 8px;
    font-weight: 600;
    font-size: 12px;
    border: none;
    border-bottom: 1px solid #334155;
}

/* Progress bar */
QProgressBar {
    background-color: #0f172a;
    border: 1px solid #334155;
    border-radius: 4px;
    text-align: center;
    color: #f8fafc;
    height: 18px;
}

QProgressBar::chunk {
    background-color: #0284c7;
    border-radius: 3px;
}

/* Scrollbars */
QScrollBar:vertical {
    border: none;
    background: #0f172a;
    width: 10px;
    margin: 0;
}

QScrollBar::handle:vertical {
    background: #334155;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}
"""

