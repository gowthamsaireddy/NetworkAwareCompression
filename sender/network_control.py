"""Linux tc (traffic control) and netem network emulation manager.

Provides bandwidth throttling, delay, and packet-loss emulation on Ubuntu.
Enforces verification and zero-dummy reporting.
"""

from dataclasses import dataclass
import os
import platform
import subprocess
from typing import List, Optional, Tuple


@dataclass
class NetworkProfile:
    name: str
    bandwidth: str  # e.g., "10mbit", "2mbit", "512kbit", "128kbit"
    delay_ms: int = 0
    loss_percent: float = 0.0


PRESET_PROFILES = [
    NetworkProfile("10 Mbps (High Speed)", "10mbit", delay_ms=5, loss_percent=0.0),
    NetworkProfile("2 Mbps (Medium Speed)", "2mbit", delay_ms=20, loss_percent=0.5),
    NetworkProfile("512 Kbps (Constrained)", "512kbit", delay_ms=50, loss_percent=1.0),
    NetworkProfile("128 Kbps (Severely Constrained)", "128kbit", delay_ms=100, loss_percent=2.0),
]


class NetworkController:
    """Controls network emulation via Linux tc/netem."""

    def __init__(self, interface: str = "eth0"):
        self.interface = interface
        self.is_linux = platform.system().lower() == "linux"
        self._current_profile: Optional[NetworkProfile] = None

    def detect_interfaces(self) -> List[str]:
        """Detect available network interfaces."""
        if not self.is_linux:
            # On Windows (development/testing environment)
            return ["loopback", "Ethernet"]
        try:
            output = subprocess.check_output(["ip", "-o", "link", "show"], text=True)
            interfaces = []
            for line in output.splitlines():
                parts = line.split(":")
                if len(parts) >= 2:
                    iface = parts[1].strip()
                    interfaces.append(iface)
            return interfaces or ["eth0", "lo"]
        except Exception:
            return ["eth0", "lo"]

    def apply_profile(self, profile: NetworkProfile) -> Tuple[bool, str]:
        """Applies traffic shaping via tc/netem on Linux."""
        if not self.is_linux:
            return False, "tc/netem traffic shaping is only supported on Linux (Ubuntu VM)."

        # Real execution to be fully wired in Phase 4
        return False, "Network shaping execution scheduled for Phase 4."

    def reset_configuration(self) -> Tuple[bool, str]:
        """Clears all qdisc traffic shaping rules on the interface."""
        if not self.is_linux:
            return True, "No-op: Not running on Linux."

        try:
            cmd = ["sudo", "tc", "qdisc", "del", "dev", self.interface, "root"]
            subprocess.run(cmd, check=True, capture_output=True)
            self._current_profile = None
            return True, "Traffic shaping rules successfully reset."
        except subprocess.CalledProcessError as e:
            return False, f"Failed to reset tc rules: {e.stderr.decode('utf-8', errors='ignore')}"
        except Exception as e:
            return False, f"Error: {e}"

    def verify_status(self) -> str:
        """Verifies actual active qdisc rules from kernel."""
        if not self.is_linux:
            return "Emulation unavailable (Host OS is not Linux)"
        try:
            cmd = ["tc", "qdisc", "show", "dev", self.interface]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                output = res.stdout.strip()
                return output if output else "No active traffic shaping rules (Default full rate)"
            return f"Error querying tc: {res.stderr.strip()}"
        except Exception as e:
            return f"Query error: {e}"

