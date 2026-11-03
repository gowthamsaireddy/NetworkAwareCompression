"""Provisional rule-based adaptive compression quality controller.

Maps estimated/measured available bandwidth to recommended JPEG quality levels:
- >= 10 Mbps: Q90 (High Quality)
- >= 2 Mbps and < 10 Mbps: Q70 (Balanced)
- >= 512 Kbps and < 2 Mbps: Q50 (Constrained)
- < 512 Kbps: Q30 (Severely Constrained)

Thresholds are provisional and configurable for academic research experiments.
Maintains strict separation between adaptive compression and fixed-quality experiments.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple


class BandwidthEstimationMode(Enum):
    STATIC_PROFILE = "Static Profile"
    MEASURED_EWMA = "Measured EWMA Throughput"
    MANUAL_INPUT = "Manual Input"


@dataclass
class AdaptiveThresholds:
    q90_min_kbps: float = 10000.0  # 10 Mbps
    q70_min_kbps: float = 2000.0   # 2 Mbps
    q50_min_kbps: float = 512.0    # 512 Kbps


@dataclass
class AdaptiveDecision:
    """Stores controller quality decision rationale and channel state."""
    quality: int
    bandwidth_kbps: float
    mode: str
    explanation: str
    timestamp: str

    def __iter__(self):
        """Allows unpacking as (quality, explanation) for backwards compatibility."""
        yield self.quality
        yield self.explanation


class AdaptiveController:
    """Calculates target compression quality based on channel bandwidth conditions."""

    def __init__(
        self,
        thresholds: AdaptiveThresholds = AdaptiveThresholds(),
        ewma_alpha: float = 0.3,
    ):
        self.thresholds = thresholds
        self.ewma_alpha = ewma_alpha  # Smoothing factor: 0 < alpha <= 1.0
        self.estimated_bandwidth_kbps: Optional[float] = None
        self.history: List[AdaptiveDecision] = []

    def update_measured_throughput(self, measured_kbps: float) -> float:
        """Updates channel bandwidth estimate using Exponentially Weighted Moving Average (EWMA)."""
        if measured_kbps <= 0:
            return self.estimated_bandwidth_kbps or 0.0

        if self.estimated_bandwidth_kbps is None:
            self.estimated_bandwidth_kbps = measured_kbps
        else:
            self.estimated_bandwidth_kbps = (
                self.ewma_alpha * measured_kbps + (1.0 - self.ewma_alpha) * self.estimated_bandwidth_kbps
            )
        return self.estimated_bandwidth_kbps

    def reset_estimation(self) -> None:
        """Resets the dynamic throughput estimator."""
        self.estimated_bandwidth_kbps = None

    def determine_quality(self, bandwidth_kbps: float, source_mode: str = "Manual") -> AdaptiveDecision:
        """Evaluates bandwidth against provisional thresholds and returns an AdaptiveDecision."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if bandwidth_kbps >= self.thresholds.q90_min_kbps:
            quality = 90
            desc = f"Bandwidth {bandwidth_kbps:,.1f} Kbps >= 10 Mbps -> Selected Q90 (High Quality)"
        elif bandwidth_kbps >= self.thresholds.q70_min_kbps:
            quality = 70
            desc = f"Bandwidth {bandwidth_kbps:,.1f} Kbps in [2 Mbps, 10 Mbps) -> Selected Q70 (Balanced)"
        elif bandwidth_kbps >= self.thresholds.q50_min_kbps:
            quality = 50
            desc = f"Bandwidth {bandwidth_kbps:,.1f} Kbps in [512 Kbps, 2 Mbps) -> Selected Q50 (Constrained)"
        else:
            quality = 30
            desc = f"Bandwidth {bandwidth_kbps:,.1f} Kbps < 512 Kbps -> Selected Q30 (Severely Constrained)"

        decision = AdaptiveDecision(
            quality=quality,
            bandwidth_kbps=bandwidth_kbps,
            mode=source_mode,
            explanation=desc,
            timestamp=now_str,
        )
        self.history.append(decision)
        return decision

    def evaluate_with_fallback(
        self,
        manual_kbps: Optional[float] = None,
        profile_bw_bps: Optional[float] = None,
        use_ewma: bool = False,
    ) -> AdaptiveDecision:
        """Determines quality based on prioritized bandwidth inputs."""
        if use_ewma and self.estimated_bandwidth_kbps is not None:
            return self.determine_quality(self.estimated_bandwidth_kbps, source_mode="Measured EWMA")
        elif profile_bw_bps is not None and profile_bw_bps > 0:
            profile_kbps = profile_bw_bps / 1000.0
            return self.determine_quality(profile_kbps, source_mode="Network Profile")
        elif manual_kbps is not None and manual_kbps > 0:
            return self.determine_quality(manual_kbps, source_mode="Manual Configuration")
        else:
            # Default unshaped fallback (assumes >= 10 Mbps)
            return self.determine_quality(10000.0, source_mode="Default Fallback (10 Mbps)")
