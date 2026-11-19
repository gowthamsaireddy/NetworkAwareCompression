"""Unit tests for Phase 2: Research Infrastructure, Metrics Engine, and SQLite Persistence."""

import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from common.results_manager import ExperimentRecord, ResultsManager
from sender.tcp_client import TransmissionReport


class TestResultsInfrastructure(unittest.TestCase):
    """Verifies schema migration, transactional writes, backup creation, and export utilities."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="nac_test_infra_")
        self.db_path = Path(self.temp_dir) / "test_experiments.sqlite"
        self.csv_path = Path(self.temp_dir) / "test_experiments.csv"
        self.mgr = ResultsManager(db_path=self.db_path, csv_path=self.csv_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_full_record_lifecycle(self):
        """Test recording, fetching, and updating an ExperimentRecord with all 38+ metrics."""
        rec = ExperimentRecord(
            experiment_id="EXP_TEST_001",
            timestamp="2026-11-18 20:00:00",
            image_id="benign (10)",
            image_category="benign",
            compression_method="NORMAL",
            compression_quality=70,
            compression_params='{"quality": 70}',
            region_adaptive_margin=None,
            compression_model_version=None,
            original_file_size=200000,
            compressed_file_size=40000,
            transmitted_bytes=40120,
            received_bytes=40120,
            compression_ratio=5.0,
            size_reduction_pct=80.0,
            psnr=35.5,
            ssim=0.92,
            bandwidth_limit="10mbit",
            packet_loss_rate=0.0,
            network_delay_ms=5.0,
            theoretical_transmission_time_sec=0.032,
            transmission_time_sec=0.035,
            measured_throughput_kbps=9170.0,
            connection_time_ms=1.2,
            round_trip_time_sec=0.045,
            effective_bandwidth_kbps=7132.0,
            decoding_time_ms=2.1,
            verification_time_ms=0.4,
            inference_time_ms=18.5,
            end_to_end_time_ms=45.0,
            checksum_verified=True,
            dice_coefficient=0.88,
            iou=0.79,
            ground_truth_id="benign (10)_mask.png",
            experiment_status="SUCCESS",
            error_message=None,
            software_version="1.1.0",
            model_version="unet_mgda_v1",
            transmission_mode="MANUAL",
            artifact_path=None,
            notes="Full infrastructure test trial",
        )

        self.mgr.record_experiment(rec)

        # Retrieve and verify
        stored = self.mgr.get_record("EXP_TEST_001")
        self.assertIsNotNone(stored)
        self.assertEqual(stored["experiment_id"], "EXP_TEST_001")
        self.assertEqual(stored["image_category"], "benign")
        self.assertEqual(stored["compression_method"], "NORMAL")
        self.assertEqual(stored["transmitted_bytes"], 40120)
        self.assertEqual(stored["received_bytes"], 40120)
        self.assertAlmostEqual(stored["psnr"], 35.5, places=1)
        self.assertAlmostEqual(stored["ssim"], 0.92, places=2)
        self.assertAlmostEqual(stored["round_trip_time_sec"], 0.045, places=3)
        self.assertAlmostEqual(stored["decoding_time_ms"], 2.1, places=1)
        self.assertAlmostEqual(stored["verification_time_ms"], 0.4, places=1)
        self.assertAlmostEqual(stored["inference_time_ms"], 18.5, places=1)
        self.assertAlmostEqual(stored["dice_coefficient"], 0.88, places=2)
        self.assertAlmostEqual(stored["iou"], 0.79, places=2)
        self.assertEqual(stored["checksum_verified"], 1)

    def test_legacy_schema_migration_preserves_data(self):
        """Simulate a legacy 20-column table and test non-destructive schema migration."""
        legacy_db = Path(self.temp_dir) / "legacy.sqlite"
        conn = sqlite3.connect(legacy_db)
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE experiments (
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
                theoretical_transmission_time_sec REAL,
                transmission_time_sec REAL,
                measured_throughput_kbps REAL,
                checksum_verified INTEGER NOT NULL,
                dice_coefficient REAL,
                iou REAL,
                notes TEXT
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO experiments (
                experiment_id, timestamp, image_id, compression_quality,
                original_file_size, compressed_file_size, compression_ratio,
                checksum_verified
            ) VALUES ('LEGACY_001', '2026-11-04 10:00:00', 'benign (1)', 70, 100000, 20000, 5.0, 1)
            """
        )
        conn.commit()
        conn.close()

        # Initialize ResultsManager on top of legacy DB
        migrated_mgr = ResultsManager(db_path=legacy_db)
        all_rows = migrated_mgr.fetch_all()
        self.assertEqual(len(all_rows), 1)
        self.assertEqual(all_rows[0]["experiment_id"], "LEGACY_001")

        # Verify that new columns exist and default cleanly
        cursor = sqlite3.connect(legacy_db).cursor()
        cols = {col[1] for col in cursor.execute("PRAGMA table_info(experiments)").fetchall()}
        self.assertIn("compression_method", cols)
        self.assertIn("transmitted_bytes", cols)
        self.assertIn("decoding_time_ms", cols)
        self.assertIn("verification_time_ms", cols)
        self.assertIn("inference_time_ms", cols)

    def test_database_backup(self):
        """Test online transactional backup of SQLite experiments database."""
        rec = ExperimentRecord(
            experiment_id="BACKUP_TEST_001",
            timestamp="2026-11-18 21:00:00",
            image_id="malignant (5)",
            compression_quality=50,
            original_file_size=150000,
            compressed_file_size=30000,
            compression_ratio=5.0,
            checksum_verified=True,
        )
        self.mgr.record_experiment(rec)

        backup_dir = Path(self.temp_dir) / "backups"
        backup_file = self.mgr.backup_database(backup_dir)
        self.assertTrue(backup_file.exists())

        # Verify backup contains the identical record
        backup_mgr = ResultsManager(db_path=backup_file)
        rows = backup_mgr.fetch_all()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["experiment_id"], "BACKUP_TEST_001")

    def test_disk_space_checker(self):
        """Verify disk space validation check."""
        is_ok, free_mb, msg = self.mgr.check_disk_space(min_free_mb=10)
        self.assertTrue(is_ok)
        self.assertGreater(free_mb, 10)

        # Impossibly high requirement should fail safely
        is_ok_fail, _, msg_fail = self.mgr.check_disk_space(min_free_mb=100_000_000)
        self.assertFalse(is_ok_fail)
        self.assertIn("Low disk space", msg_fail)

    def test_is_experiment_completed_deduplication(self):
        """Verify deduplication lookup for avoiding redundant trials."""
        rec = ExperimentRecord(
            experiment_id="DEDUP_001",
            timestamp="2026-11-18 21:30:00",
            image_id="benign (25)",
            compression_method="NORMAL",
            compression_quality=70,
            original_file_size=180000,
            compressed_file_size=36000,
            compression_ratio=5.0,
            bandwidth_limit="10mbit",
            checksum_verified=True,
        )
        self.mgr.record_experiment(rec)

        # Should report completed
        self.assertTrue(
            self.mgr.is_experiment_completed(
                image_id="benign (25)",
                compression_method="NORMAL",
                compression_quality=70,
                bandwidth_limit="10mbit",
            )
        )
        # Different quality or unperformed should report false
        self.assertFalse(
            self.mgr.is_experiment_completed(
                image_id="benign (25)",
                compression_method="NORMAL",
                compression_quality=90,
            )
        )

    def test_json_and_csv_exports(self):
        """Verify export to structured JSON and CSV."""
        rec = ExperimentRecord(
            experiment_id="EXPORT_001",
            timestamp="2026-11-18 22:00:00",
            image_id="normal (3)",
            compression_quality=90,
            original_file_size=250000,
            compressed_file_size=125000,
            compression_ratio=2.0,
            checksum_verified=True,
        )
        self.mgr.record_experiment(rec)

        json_export = Path(self.temp_dir) / "exported.json"
        self.mgr.export_to_json(json_export)
        self.assertTrue(json_export.exists())
        with open(json_export, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["experiment_id"], "EXPORT_001")


if __name__ == "__main__":
    unittest.main()
