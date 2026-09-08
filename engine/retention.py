"""Data retention and auto-purge management for database records and snapshot files."""

import datetime
import os
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from engine.logger import logger
from storage.db import DEFAULT_DB_PATH, get_connection


def cleanup_old_records(
    retention_days: int = 30,
    db_path: Optional[Path] = None,
) -> Dict[str, int]:
    """Purge incident records and snapshot files older than retention_days.

    Returns summary: {'deleted_records': int, 'deleted_snapshots': int, 'failed_snapshots': int}
    """
    summary: Dict[str, int] = {
        "deleted_records": 0,
        "deleted_snapshots": 0,
        "failed_snapshots": 0,
    }

    target_db: Path = db_path if db_path is not None else DEFAULT_DB_PATH
    if not target_db.exists():
        return summary

    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=retention_days)
    cutoff_iso = cutoff_date.isoformat()

    conn = None
    try:
        conn = get_connection(target_db)
        cursor = conn.cursor()

        # Determine table name (event_logs or incident_logs)
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('event_logs', 'incident_logs');"
        )
        tables = [row[0] for row in cursor.fetchall()]
        if not tables:
            return summary

        target_table = "event_logs" if "event_logs" in tables else tables[0]

        # Fetch old records and their snapshot paths
        cursor.execute(
            f"""
            SELECT id, snapshot_path FROM {target_table}
            WHERE created_at < ? OR trigger_time < ?;
            """,
            (cutoff_iso, cutoff_iso),
        )
        old_records = cursor.fetchall()

        if not old_records:
            return summary

        record_ids = [row["id"] for row in old_records]

        # Delete physical snapshot files
        for row in old_records:
            snap_path_str = row["snapshot_path"]
            if snap_path_str:
                snap_file = Path(snap_path_str)
                if snap_file.exists():
                    try:
                        snap_file.unlink()
                        summary["deleted_snapshots"] += 1
                    except OSError as e:
                        logger.warning(f"Failed to delete expired snapshot {snap_file}: {e}")
                        summary["failed_snapshots"] += 1

        # Delete database records in bulk
        placeholders = ",".join("?" for _ in record_ids)
        cursor.execute(
            f"DELETE FROM {target_table} WHERE id IN ({placeholders});",
            record_ids,
        )
        conn.commit()
        summary["deleted_records"] = cursor.rowcount

        # Reclaim SQLite disk space
        try:
            conn.execute("VACUUM;")
        except sqlite3.OperationalError as e:
            # VACUUM can fail if another transaction is pending in WAL mode
            logger.info(f"SQLite VACUUM deferred: {e}")

    except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
        logger.error(f"Retention cleanup failed due to database error: {e}")
    finally:
        if conn is not None:
            conn.close()

    if summary["deleted_records"] > 0 or summary["deleted_snapshots"] > 0:
        logger.info(
            f"Retention auto-purge executed: {summary['deleted_records']} record(s) "
            f"and {summary['deleted_snapshots']} snapshot(s) deleted."
        )

    return summary


def get_disk_usage(path: Optional[Path] = None) -> Dict[str, float]:
    """Return disk storage telemetry in GB and percentage."""
    import shutil
    target = path if path is not None else Path(__file__).resolve().parent.parent
    try:
        usage = shutil.disk_usage(str(target))
        total_gb = usage.total / (1024 ** 3)
        used_gb = usage.used / (1024 ** 3)
        free_gb = usage.free / (1024 ** 3)
        pct = (usage.used / float(max(1, usage.total))) * 100.0
        return {
            "total_gb": round(total_gb, 2),
            "used_gb": round(used_gb, 2),
            "free_gb": round(free_gb, 2),
            "percent_used": round(pct, 1),
        }
    except Exception as e:
        logger.warning(f"[DiskGuard] Failed to check disk usage: {e}")
        return {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 999.0, "percent_used": 0.0}


def emergency_fifo_purge(
    min_free_gb: float = 5.0,
    max_usage_percent: float = 90.0,
    target_dirs: Optional[list] = None,
) -> Dict[str, Any]:
    """Emergency FIFO deletion of oldest snapshots if free disk space < min_free_gb or usage > max_usage_percent."""
    summary: Dict[str, Any] = {"deleted_files": 0, "freed_bytes": 0, "triggered": False}
    root = Path(__file__).resolve().parent.parent
    usage = get_disk_usage(root)

    if usage["free_gb"] >= min_free_gb and usage["percent_used"] <= max_usage_percent:
        return summary

    summary["triggered"] = True
    logger.warning(
        f"[DiskGuard] Emergency Low-Disk Failsafe triggered! Free space: {usage['free_gb']} GB "
        f"(Threshold: {min_free_gb} GB), Usage: {usage['percent_used']}% (Threshold: {max_usage_percent}%)."
    )

    # Collect directories to scan
    scan_dirs = []
    if target_dirs is not None:
        scan_dirs.extend(target_dirs)
    else:
        cameras_dir = root / "cameras"
        if cameras_dir.is_dir():
            for cam in cameras_dir.iterdir():
                if cam.is_dir():
                    sdir = cam / "snapshots"
                    if sdir.is_dir():
                        scan_dirs.append(sdir)
        reports_snap = root / "reports" / "snapshots"
        if reports_snap.is_dir():
            scan_dirs.append(reports_snap)
        storage_snap = root / "storage" / "snapshots"
        if storage_snap.is_dir():
            scan_dirs.append(storage_snap)

    all_files = []
    for sdir in scan_dirs:
        for f in sdir.glob("*.*"):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                try:
                    all_files.append((f.stat().st_mtime, f))
                except OSError:
                    continue

    # Sort oldest first (FIFO)
    all_files.sort(key=lambda x: x[0])

    deleted_count = 0
    freed_bytes = 0

    for _, file_path in all_files:
        try:
            sz = file_path.stat().st_size
            file_path.unlink()
            deleted_count += 1
            freed_bytes += sz

            if deleted_count % 25 == 0:
                cur_usage = get_disk_usage(root)
                if cur_usage["free_gb"] >= min_free_gb and cur_usage["percent_used"] <= (max_usage_percent - 5.0):
                    break
        except OSError as e:
            logger.warning(f"[DiskGuard] Failed to delete file {file_path}: {e}")

    summary["deleted_files"] = deleted_count
    summary["freed_bytes"] = freed_bytes
    cur_usage = get_disk_usage(root)
    logger.info(
        f"[DiskGuard] Emergency FIFO purge complete: {deleted_count} file(s) removed, "
        f"{freed_bytes / (1024**2):.1f} MB freed. Current free: {cur_usage['free_gb']} GB."
    )
    return summary


class DiskGuardWorker:
    """Singleton background daemon that runs periodic 6-hour retention and low-disk failsafe."""

    _instance: Optional["DiskGuardWorker"] = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs) -> "DiskGuardWorker":
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DiskGuardWorker, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(
        self,
        retention_interval_sec: float = 21600.0,  # 6 hours
        check_disk_interval_sec: float = 300.0,   # 5 minutes
        retention_days: int = 30,
        min_free_gb: float = 5.0,
        max_usage_percent: float = 90.0,
    ) -> None:
        if getattr(self, "_initialized", False):
            return

        self.retention_interval_sec = retention_interval_sec
        self.check_disk_interval_sec = check_disk_interval_sec
        self.retention_days = retention_days
        self.min_free_gb = min_free_gb
        self.max_usage_percent = max_usage_percent

        self._stopped = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._initialized = True

    def start(self) -> None:
        """Start the background daemon worker."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stopped.clear()
        self._thread = threading.Thread(target=self._run, name="DiskGuardWorker", daemon=True)
        self._thread.start()
        logger.info(
            f"[DiskGuard] Background worker started (retention every {self.retention_interval_sec / 3600:.1f}h, "
            f"disk check every {self.check_disk_interval_sec:.0f}s, min_free={self.min_free_gb}GB)."
        )

    def _run(self) -> None:
        last_retention_time = 0.0
        while not self._stopped.is_set():
            now = time.time()
            try:
                # 1. Periodic Retention cleanup (every 6 hours)
                if (now - last_retention_time) >= self.retention_interval_sec:
                    cleanup_old_records(retention_days=self.retention_days)
                    last_retention_time = now

                # 2. Emergency Low-Disk Failsafe (every 5 minutes)
                emergency_fifo_purge(
                    min_free_gb=self.min_free_gb,
                    max_usage_percent=self.max_usage_percent,
                )
            except Exception as e:
                logger.error(f"[DiskGuard] Background worker loop error: {e}")

            # Sleep in short increments for responsive shutdown
            self._stopped.wait(timeout=self.check_disk_interval_sec)

    def stop(self) -> None:
        """Signal daemon to stop."""
        self._stopped.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            logger.info("[DiskGuard] Background worker terminated.")
