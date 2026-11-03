"""Sender Application Entrypoint."""

import sys
from pathlib import Path

# Ensure root repository directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from PyQt6.QtWidgets import QApplication
from common.config import ensure_directories
from common.logging_config import setup_logger
from ui.sender.main_window import SenderMainWindow

logger = setup_logger("SenderApp", log_file="sender.log")


def main():
    """Starts the Sender Node GUI application."""
    ensure_directories()
    logger.info("Initializing Sender Application...")

    app = QApplication(sys.argv)
    app.setApplicationName("NetworkAwareCompression-Sender")

    window = SenderMainWindow()
    window.show()

    logger.info("Sender GUI launched.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

