"""Experiment results manager with CSV and SQLite persistence.

Follows strict Zero-Dummy-Data policy: only records genuine, measured experimental values.
"""

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
import sqlite3
from typing import Any, List, Optional
from common.config import EXPERIMENTS_DIR


@dataclass
class ExperimentRecord:
    """Record representing a single transmission and evaluation experiment."""
    experiment_id: str
    timestamp: str
    image_id: str
    compression_quality: int
    original_file_size: int  # bytes
    compressed_file_size: int  # bytes
    compression_ratio: float
    psnr: Optional[float]  # dB or None if uncalculated
    ssim: Optional[float]  # 0..1 or None if uncalculated
    bandwidth_limit: Optional[str]  # e.g., "10Mbps" or "unlimited"
    packet_loss_rate: Optional[float]  # %
    network_delay_ms: Optional[float]  # ms
    transmission_time_sec: Optional[float]  # measured seconds
    measured_throughput_kbps: Optional[float]  # measured kbps
    checksum_verified: bool
    dice_coefficient: Optional[float]  # None if no GT available
    iou: Optional[float]  # None if no GT available
    notes: Optional[str] = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResultsManager:
    """Manages persistent experiment logs in CSV and SQLite formats."""

    CSV_HEADER = [f.name for f in fields(ExperimentRecord)]

    def __init__(self, db_path: Optional[Path] = None, csv_path: Optional[Path] = None):
        EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path or (EXPERIMENTS_DIR / "experiments.sqlite")
        self.csv_path = csv_path or (EXPERIMENTS_DIR / "experiments.csv")
        self._init_sqlite()
        self._init_csv()

    def _init_sqlite(self) -> None:
        """Initializes SQLite database table if not present."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    image_id TEXT NOT NULL,
                    compression_quality INTEGER NOT NULL,
                    original_file_size INTEGER NOT NULL,
                    compressed_file_size INTEGER NOT NULL,
                    compression_ratio REAL NOT NULL,
                    psnr REAL,
                    ssim REAL,
                    bandwidth_limit TEXT,
                    packet_loss_rate REAL,
                    network_delay_ms REAL,
                    transmission_time_sec REAL,
                    measured_throughput_kbps REAL,
                    checksum_verified INTEGER NOT NULL,
                    dice_coefficient REAL,
                    iou REAL,
                    notes TEXT
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def _init_csv(self) -> None:
        """Ensures CSV file exists with proper headers."""
        if not self.csv_path.exists():
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.CSV_HEADER)

    def record_experiment(self, record: ExperimentRecord) -> None:
        """Persist a genuine experimental record to both SQLite and CSV."""
        # 1. Insert into SQLite
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO experiments (
                    experiment_id, timestamp, image_id, compression_quality,
                    original_file_size, compressed_file_size, compression_ratio,
                    psnr, ssim, bandwidth_limit, packet_loss_rate,
                    network_delay_ms, transmission_time_sec, measured_throughput_kbps,
                    checksum_verified, dice_coefficient, iou, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.experiment_id,
                    record.timestamp,
                    record.image_id,
                    record.compression_quality,
                    record.original_file_size,
                    record.compressed_file_size,
                    record.compression_ratio,
                    record.psnr,
                    record.ssim,
                    record.bandwidth_limit,
                    record.packet_loss_rate,
                    record.network_delay_ms,
                    record.transmission_time_sec,
                    record.measured_throughput_kbps,
                    1 if record.checksum_verified else 0,
                    record.dice_coefficient,
                    record.iou,
                    record.notes,
                ),
            )
            conn.commit()
        finally:
            conn.close()

        # 2. Append to CSV
        with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_HEADER)
            writer.writerow(record.to_dict())

    def fetch_all(self) -> List[dict[str, Any]]:
        """Fetch all recorded experiments from SQLite."""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM experiments ORDER BY id DESC")
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()
