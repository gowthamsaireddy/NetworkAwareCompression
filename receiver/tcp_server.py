"""TCP Server for accepting connections from the sender and receiving images."""

import socket
import threading
from typing import Callable, Optional, Tuple

from common.logging_config import setup_logger
from shared.constants import DEFAULT_HOST, DEFAULT_PORT
from shared.metadata import AcknowledgmentMessage, TransmissionMetadata

logger = setup_logger("TCPReceiverServer")


class TCPReceiverServer:
    """Manages listening TCP server and incoming client sessions."""

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.host = host
        self.port = port
        self.server_socket: Optional[socket.socket] = None
        self.is_running: bool = False
        self._listen_thread: Optional[threading.Thread] = None
        self.on_image_received: Optional[Callable[[TransmissionMetadata, bytes], None]] = None

    def start(self) -> Tuple[bool, str]:
        """Starts listening on specified host and port."""
        if self.is_running:
            return False, "Server is already running."

        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.is_running = True
            logger.info(f"Server started listening on {self.host}:{self.port}")
            return True, f"Listening on {self.host}:{self.port}"
        except Exception as e:
            self.is_running = False
            self.server_socket = None
            logger.error(f"Failed to start server on {self.host}:{self.port}: {e}")
            return False, f"Failed to bind: {e}"

    def stop(self) -> Tuple[bool, str]:
        """Stops the listening server."""
        if not self.is_running:
            return False, "Server is not running."

        self.is_running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
            self.server_socket = None

        logger.info("Server stopped.")
        return True, "Server stopped."

