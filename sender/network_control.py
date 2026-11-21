"""Linux tc (traffic control) and netem network emulation manager.

Provides bandwidth throttling, delay, jitter, and packet-loss emulation on Ubuntu.
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
    bandwidth: str  # e.g., "100mbit", "100 Mbps", "512kbit"
    delay_ms: float = 0.0
    jitter_ms: float = 0.0
    loss_percent: float = 0.0
    description: str = ""

    @property
    def bandwidth_bps(self) -> float:
        """Converts bandwidth string (e.g. '100mbit', '100 Mbps', '512kbit') to bits per second."""
        return parse_bandwidth_to_bps(self.bandwidth)

    @property
    def bandwidth_display(self) -> str:
        """Returns clean human-readable rate representation, e.g. '100 Mbps', '512 Kbps'."""
        bps = self.bandwidth_bps
        if bps >= 1_000_000_000:
            val = bps / 1_000_000_000.0
            return f"{int(val) if val.is_integer() else val} Gbps"
        elif bps >= 1_000_000:
            val = bps / 1_000_000.0
            return f"{int(val) if val.is_integer() else val} Mbps"
        elif bps >= 1_000:
            val = bps / 1_000.0
            return f"{int(val) if val.is_integer() else val} Kbps"
        return f"{bps:.0f} bps"

    @property
    def tc_bandwidth_str(self) -> str:
        """Returns rate formatted for Linux tc netem (e.g. '100mbit', '512kbit')."""
        bps = self.bandwidth_bps
        if bps >= 1_000_000:
            mb = bps / 1_000_000.0
            return f"{int(mb)}mbit" if mb.is_integer() else f"{mb:.2f}mbit"
        elif bps >= 1_000:
            kb = bps / 1_000.0
            return f"{int(kb)}kbit" if kb.is_integer() else f"{kb:.2f}kbit"
        return f"{int(bps)}bit"


# Six Predefined Experimental Network Emulation Profiles
PRESET_PROFILES = [
    NetworkProfile(
        name="Hospital LAN",
        bandwidth="100mbit",
        delay_ms=2.0,
        jitter_ms=0.5,
        loss_percent=0.01,
        description="High-speed hospital intranet with minimal latency and negligible loss.",
    ),
    NetworkProfile(
        name="Broadband",
        bandwidth="50mbit",
        delay_ms=20.0,
        jitter_ms=5.0,
        loss_percent=0.1,
        description="Standard medical center fiber/broadband connection.",
    ),
    NetworkProfile(
        name="4G LTE",
        bandwidth="10mbit",
        delay_ms=40.0,
        jitter_ms=15.0,
        loss_percent=0.5,
        description="Modern commercial mobile 4G LTE cellular data link.",
    ),
    NetworkProfile(
        name="3G Cellular",
        bandwidth="2mbit",
        delay_ms=100.0,
        jitter_ms=30.0,
        loss_percent=1.0,
        description="Constrained legacy 3G mobile cellular link with moderate delay and jitter.",
    ),
    NetworkProfile(
        name="Rural Link",
        bandwidth="512kbit",
        delay_ms=150.0,
        jitter_ms=40.0,
        loss_percent=2.0,
        description="Severely bandwidth-constrained remote clinic/telemedicine connection.",
    ),
    NetworkProfile(
        name="Satellite",
        bandwidth="10mbit",
        delay_ms=300.0,
        jitter_ms=30.0,
        loss_percent=0.5,
        description="Long-range geostationary satellite uplink with high propagation latency.",
    ),
]


def parse_bandwidth_to_bps(bandwidth_str: str) -> float:
    """Parses standard Linux rate strings (e.g. '100mbit', '50 Mbps', '512kbit') to bits per second."""
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
        if not self.interface or not self.interface.strip():
            return False, "No network interface specified. Please select or enter a valid network interface."

        if not self.is_linux:
            # Safe reporting for non-Linux hosts: do not falsely claim emulation
            return False, "tc/netem traffic shaping is only supported on Linux (Ubuntu). Local host emulation unavailable."

        # Step 1: Clean any existing root qdisc rules to prevent duplicate or conflicting rules
        self.reset_configuration()

        # Step 2: Assemble tc netem command
        # Syntax: tc qdisc add dev <iface> root netem rate <rate> delay <delay> [jitter] loss <loss>
        cmd = ["sudo", "tc", "qdisc", "add", "dev", self.interface.strip(), "root", "netem"]

        if profile.bandwidth:
            cmd.extend(["rate", profile.tc_bandwidth_str])

        if profile.delay_ms > 0:
            d_str = f"{int(profile.delay_ms)}ms" if profile.delay_ms.is_integer() else f"{profile.delay_ms}ms"
            if profile.jitter_ms > 0:
                j_str = f"{int(profile.jitter_ms)}ms" if profile.jitter_ms.is_integer() else f"{profile.jitter_ms}ms"
                cmd.extend(["delay", d_str, j_str])
            else:
                cmd.extend(["delay", d_str])

        if profile.loss_percent > 0:
            l_str = f"{int(profile.loss_percent)}%" if profile.loss_percent.is_integer() else f"{profile.loss_percent}%"
            cmd.extend(["loss", l_str])

        try:
            res = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Executed: {' '.join(cmd)}")
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.strip() or str(e)
            logger.error(f"tc command failed on interface '{self.interface}': {err_msg}")
            return False, f"Failed to apply tc rule on {self.interface}: {err_msg}"
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
        """Clears all qdisc traffic shaping rules on the target interface."""
        if not self.interface or not self.interface.strip():
            return False, "No network interface specified."

        if not self.is_linux:
            self._current_profile = None
            return True, "No-op: Host is not Linux."

        try:
            cmd = ["sudo", "tc", "qdisc", "del", "dev", self.interface.strip(), "root"]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self._current_profile = None
            if res.returncode == 0 or "No such file or directory" in res.stderr or "Cannot delete qdisc" in res.stderr:
                return True, f"Traffic shaping cleared on {self.interface} (Default unshaped rate)."
            return False, f"Reset notice: {res.stderr.strip()}"
        except Exception as e:
            return False, f"Error resetting tc rules: {e}"

    def reset(self) -> Tuple[bool, str]:
        """Clears all traffic shaping rules (convenience alias for reset_configuration)."""
        return self.reset_configuration()

    def verify_status(self) -> str:
        """Queries the Linux kernel for actual active qdisc rules on the interface."""
        if not self.is_linux:
            return "Emulation unavailable (Host OS is not Linux)"

        if not self.interface:
            return "No interface specified"

        try:
            cmd = ["tc", "qdisc", "show", "dev", self.interface.strip()]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                output = res.stdout.strip()
                if not output:
                    return f"No active traffic shaping rules on {self.interface} (Full rate / Unshaped)"
                return output
            return f"Error querying tc: {res.stderr.strip()}"
        except Exception as e:
            return f"Query error: {e}"

    @property
    def current_profile(self) -> Optional[NetworkProfile]:
        return self._current_profile
