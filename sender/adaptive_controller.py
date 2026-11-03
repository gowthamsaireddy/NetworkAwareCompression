"""Provisional rule-based adaptive compression quality controller.

Maps estimated/measured available bandwidth to recommended JPEG quality levels:
- >= 10 Mbps: Q90
- >= 2 Mbps and < 10 Mbps: Q70
- >= 512 Kbps and < 2 Mbps: Q50
- < 512 Kbps: Q30

Thresholds are provisional and configurable for research experimentation.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class AdaptiveThresholds:
    q90_min_kbps: float = 10000.0  # 10 Mbps
    q70_min_kbps: float = 2000.0   # 2 Mbps
    q50_min_kbps: float = 512.0    # 512 Kbps


class AdaptiveController:
    """Calculates target compression quality based on network bandwidth conditions."""

    def __init__(self, thresholds: AdaptiveThresholds = AdaptiveThresholds()):
        self.thresholds = thresholds

    def determine_quality(self, bandwidth_kbps: float) -> Tuple[int, str]:
        """Returns recommended (quality, explanation)."""
        if bandwidth_kbps >= self.thresholds.q90_min_kbps:
            return 90, f"Bandwidth {bandwidth_kbps:.1f} Kbps >= 10 Mbps -> Q90 (High Quality)"
        elif bandwidth_kbps >= self.thresholds.q70_min_kbps:
            return 70, f"Bandwidth {bandwidth_kbps:.1f} Kbps >= 2 Mbps -> Q70 (Balanced)"
        elif bandwidth_kbps >= self.thresholds.q50_min_kbps:
            return 50, f"Bandwidth {bandwidth_kbps:.1f} Kbps >= 512 Kbps -> Q50 (Constrained)"
        else:
            return 30, f"Bandwidth {bandwidth_kbps:.1f} Kbps < 512 Kbps -> Q30 (Severely Constrained)"

