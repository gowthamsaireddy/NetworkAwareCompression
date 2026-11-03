"""Linux tc (traffic control) and netem network emulation manager.

Provides bandwidth throttling, delay, and packet-loss emulation on Ubuntu.
Enforces kernel-level verification and strict Zero-Dummy reporting.
"""

from dataclasses import dataclass
import os
import platform
import re
import subprocess
from typing import Dict, List, Optional, Tuple

from common.logging_config import setup_logger

logger = setup_logger("NetworkController")


@dataclass
class NetworkProfile:
    name: str
    bandwidth: str  # e.g., "10mbit", "2mbit", "512kbit", "128kbit"
    delay_ms: int = 0
    loss_percent: float = 0.0

    @property
    def bandwidth_bps(self) -> float:
        """Converts bandwidth string (e.g. '10mbit', '512kbit') to bits per second."""
        return parse_bandwidth_to_bps(self.bandwidth)


PRESET_PROFILES = [
    NetworkProfile("10 Mbps (High Speed)", "10mbit", delay_ms=5, loss_percent=0.0),
    NetworkProfile("2 Mbps (Medium Speed)", "2mbit", delay_ms=20, loss_percent=0.5),
    NetworkProfile("512 Kbps (Constrained)", "512kbit", delay_ms=50, loss_percent=1.0),
    NetworkProfile("128 Kbps (Severely Constrained)", "128kbit", delay_ms=100, loss_percent=2.0),
]


def parse_bandwidth_to_bps(bandwidth_str: str) -> float:
    """Parses standard Linux rate strings (e.g. '10mbit', '2mbit', '512kbit') to bits per second."""
    bw = bandwidth_str.strip().lower()
    match = re.match(r"^([0-9.]+)\s*([a-z]+)?$", bw)
    if not match:
        return 0.0

    val = float(match.group(1))
    unit = match.group(2) or "bps"

    if "mbit" in unit or "mbps" in unit or "m" == unit:
        return val * 1_000_000.0
    elif "kbit" in unit or "kbps" in unit or "k" == unit:
        return val * 1_000.0
    elif "gbit" in unit or "gbps" in unit or "g" == unit:
        return val * 1_000_000_000.0
    return val


def compute_theoretical_transmission_time(payload_bytes: int, bandwidth_bps: float) -> Optional[float]:
    """Calculates theoretical wire transmission time: (payload_bytes * 8) / bandwidth_bps.

    Returns None if bandwidth is zero or unlimited.
    """
    if bandwidth_bps <= 0:
        return None
    return float((payload_bytes * 8.0) / bandwidth_bps)


class NetworkController:
    """Controls network emulation via Linux tc/netem with kernel status verification."""

    def __init__(self, interface: str = "eth0"):
        self.interface = interface
        self.is_linux = platform.system().lower() == "linux"
        self._current_profile: Optional[NetworkProfile] = None

    def detect_interfaces(self) -> List[str]:
        """Detect available network interfaces on the operating system."""
        if not self.is_linux:
            # On Windows: Query real interface names via netsh or ipconfig
            try:
                out = subprocess.check_output(["netsh", "interface", "show", "interface"], text=True)
                interfaces = []
                for line in out.splitlines():
                    if "Connected" in line or "Disconnected" in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            name = " ".join(parts[3:])
                            interfaces.append(name)
                return interfaces or ["Ethernet", "Wi-Fi", "loopback"]
            except Exception:
                return ["Ethernet", "Wi-Fi", "loopback"]

        # On Linux: Query iproute2
        try:
            output = subprocess.check_output(["ip", "-o", "link", "show"], text=True)
            interfaces = []
            for line in output.splitlines():
                parts = line.split(":")
                if len(parts) >= 2:
                    iface = parts[1].strip()
                    if "@" in iface:
                        iface = iface.split("@")[0]
                    interfaces.append(iface)
            return interfaces or ["eth0", "lo"]
        except Exception as e:
            logger.warning(f"Failed to query network interfaces: {e}")
            return ["eth0", "lo"]

    def apply_profile(self, profile: NetworkProfile) -> Tuple[bool, str]:
        """Applies traffic shaping via tc/netem on Linux.

        Enforces verification: does not claim shaping is active without kernel confirmation.
        """
        if not self.is_linux:
            return False, "tc/netem traffic shaping is only supported on Linux (Ubuntu VM). Local emulation unavailable."

        # Step 1: Clean any existing root qdisc rules
        self.reset_configuration()

        # Step 2: Assemble tc netem command
        # Modern iproute2 syntax: tc qdisc add dev <iface> root netem rate <rate> delay <delay> loss <loss>
        cmd = ["sudo", "tc", "qdisc", "add", "dev", self.interface, "root", "netem"]

        if profile.bandwidth:
            cmd.extend(["rate", profile.bandwidth])
        if profile.delay_ms > 0:
            cmd.extend(["delay", f"{profile.delay_ms}ms"])
        if profile.loss_percent > 0:
            cmd.extend(["loss", f"{profile.loss_percent}%"])

        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Executed: {' '.join(cmd)}")
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.strip() or str(e)
            logger.error(f"tc command failed: {err_msg}")
            return False, f"Failed to apply tc rule: {err_msg}"
        except Exception as e:
            return False, f"Execution error: {e}"

        # Step 3: Kernel verification
        status = self.verify_status()
        if "netem" in status.lower():
            self._current_profile = profile
            msg = f"Traffic shaping active on {self.interface} ({profile.name}). Kernel: {status}"
            logger.info(msg)
            return True, msg
        else:
            self._current_profile = None
            msg = f"Verification failed: kernel did not confirm active netem qdisc ({status})"
            logger.warning(msg)
            return False, msg

    def reset_configuration(self) -> Tuple[bool, str]:
        """Clears all qdisc traffic shaping rules on the interface."""
        if not self.is_linux:
            return True, "No-op: Host is not Linux."

        try:
            cmd = ["sudo", "tc", "qdisc", "del", "dev", self.interface, "root"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self._current_profile = None
            if res.returncode == 0 or "No such file or directory" in res.stderr or "Cannot delete qdisc" in res.stderr:
                return True, f"Traffic shaping cleared on {self.interface} (Default unshaped rate)."
            return False, f"Reset notice: {res.stderr.strip()}"
        except Exception as e:
            return False, f"Error resetting tc rules: {e}"

    def verify_status(self) -> str:
        """Queries the Linux kernel for actual active qdisc rules on the interface."""
        if not self.is_linux:
            return "Emulation unavailable (Host OS is not Linux)"

        try:
            cmd = ["tc", "qdisc", "show", "dev", self.interface]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                output = res.stdout.strip()
                if not output:
                    return "No active traffic shaping rules (Full rate / Unshaped)"
                return output
            return f"Error querying tc: {res.stderr.strip()}"
        except Exception as e:
            return f"Query error: {e}"

    @property
    def current_profile(self) -> Optional[NetworkProfile]:
        return self._current_profile
