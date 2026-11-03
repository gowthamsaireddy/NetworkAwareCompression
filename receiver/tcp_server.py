"""TCP Server implementation for receiving ultrasound images and validating checksums."""

from datetime import datetime
from pathlib import Path
import socket
import threading
import time
from typing import Callable, Optional, Tuple

from common.config import DATA_DIR
from common.logging_config import setup_logger
from receiver.image_receiver import ImageReceiverManager
from receiver.integrity import IntegrityChecker
from shared.constants import (
    ACK_CHECKSUM_MISMATCH,
    ACK_CORRUPT_PAYLOAD,
    ACK_OK,
    ACK_SERVER_ERROR,
    DEFAULT_HOST,
    DEFAULT_PORT,
)
from shared.metadata import AcknowledgmentMessage, TransmissionMetadata
from shared.protocol import recv_image_frame, send_ack_frame

logger = setup_logger("TCPReceiverServer")


class TCPReceiverServer:
    """Multi-threaded TCP server daemon for receiving framed image streams."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        save_dir: Optional[Path] = None,
        client_timeout: float = 30.0,
    ):
        self.host = host
        self.port = port
        self.client_timeout = client_timeout
        self.save_dir = save_dir or (DATA_DIR / "received")
        self.image_manager = ImageReceiverManager(self.save_dir)

        self.server_socket: Optional[socket.socket] = None
        self.is_running: bool = False
        self._listen_thread: Optional[threading.Thread] = None

        # Callbacks for GUI integration
        self.on_status_changed: Optional[Callable[[bool, str], None]] = None
        self.on_client_connected: Optional[Callable[[str], None]] = None
        self.on_client_disconnected: Optional[Callable[[str], None]] = None
        self.on_image_received: Optional[
            Callable[[TransmissionMetadata, bytes, bool, Optional[Path], str], None]
        ] = None
        self.on_progress: Optional[Callable[[int, int], None]] = None

    def start(self) -> Tuple[bool, str]:
        """Starts listening server in a background thread."""
        if self.is_running:
            return False, "Server is already running."

        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            # Short timeout on listen socket so accept() doesn't block server shutdown
            self.server_socket.settimeout(1.0)
            self.is_running = True

            self._listen_thread = threading.Thread(
                target=self._listen_loop,
                name="TCPReceiver-Listener",
                daemon=True,
            )
            self._listen_thread.start()

            msg = f"Server listening on {self.host}:{self.port}"
            logger.info(msg)
            if self.on_status_changed:
                self.on_status_changed(True, msg)
            return True, msg
        except Exception as e:
            self.is_running = False
            if self.server_socket:
                try:
                    self.server_socket.close()
                except Exception:
                    pass
                self.server_socket = None
            err = f"Failed to start server on {self.host}:{self.port} - {e}"
            logger.error(err)
            if self.on_status_changed:
                self.on_status_changed(False, err)
            return False, err

    def stop(self) -> Tuple[bool, str]:
        """Stops the listening server and releases socket resources."""
        if not self.is_running:
            return False, "Server is not running."

        self.is_running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
            self.server_socket = None

        if self._listen_thread and self._listen_thread.is_alive():
            self._listen_thread.join(timeout=2.0)

        msg = "Server stopped."
        logger.info(msg)
        if self.on_status_changed:
            self.on_status_changed(False, msg)
        return True, msg

    def _listen_loop(self) -> None:
        """Background listener accepting connections."""
        while self.is_running and self.server_socket:
            try:
                client_sock, client_addr = self.server_socket.accept()
            except socket.timeout:
                continue
            except Exception as e:
                if not self.is_running:
                    break
                logger.warning(f"Accept exception: {e}")
                continue

            addr_str = f"{client_addr[0]}:{client_addr[1]}"
            logger.info(f"Accepted connection from {addr_str}")
            if self.on_client_connected:
                self.on_client_connected(addr_str)

            # Handle client in dedicated worker thread
            client_thread = threading.Thread(
                target=self._handle_client,
                args=(client_sock, addr_str),
                name=f"TCPReceiver-Client-{addr_str}",
                daemon=True,
            )
            client_thread.start()

    def _handle_client(self, client_sock: socket.socket, addr_str: str) -> None:
        """Handles framed transfers from a connected client session."""
        try:
            while self.is_running:
                try:
                    metadata, image_bytes = recv_image_frame(
                        client_sock,
                        timeout=self.client_timeout,
                        progress_callback=self.on_progress,
                    )
                except (ConnectionError, TimeoutError) as e:
                    logger.debug(f"Client session ended ({addr_str}): {e}")
                    break
                except ValueError as e:
                    logger.error(f"Frame parsing error from {addr_str}: {e}")
                    # Try to send error ACK
                    err_ack = AcknowledgmentMessage(
                        status_code=ACK_CORRUPT_PAYLOAD,
                        image_id="unknown",
                        sha256_verified=False,
                        error_message=str(e),
                    )
                    try:
                        send_ack_frame(client_sock, err_ack, timeout=5.0)
                    except Exception:
                        pass
                    break

                # 2. Checksum and length verification
                is_valid, reason = IntegrityChecker.verify_all(image_bytes, metadata)
                now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if is_valid:
                    # Save image to storage
                    save_path = self.image_manager.save_image(metadata, image_bytes, now_str)
                    ack = AcknowledgmentMessage(
                        status_code=ACK_OK,
                        image_id=metadata.image_id,
                        sha256_verified=True,
                    )
                    send_ack_frame(client_sock, ack, timeout=10.0)
                    logger.info(
                        f"Successfully received and verified image '{metadata.image_id}' from {addr_str}. Saved to {save_path}"
                    )
                    if self.on_image_received:
                        self.on_image_received(metadata, image_bytes, True, save_path, now_str)
                else:
                    logger.error(
                        f"Integrity check failed for image '{metadata.image_id}' from {addr_str}: {reason}"
                    )
                    ack = AcknowledgmentMessage(
                        status_code=ACK_CHECKSUM_MISMATCH,
                        image_id=metadata.image_id,
                        sha256_verified=False,
                        error_message=reason,
                    )
                    send_ack_frame(client_sock, ack, timeout=10.0)
                    if self.on_image_received:
                        self.on_image_received(metadata, image_bytes, False, None, now_str)

        except Exception as e:
            logger.error(f"Unexpected error in client session {addr_str}: {e}")
        finally:
            try:
                client_sock.close()
            except Exception:
                pass
            logger.info(f"Closed connection from {addr_str}")
            if self.on_client_disconnected:
                self.on_client_disconnected(addr_str)
