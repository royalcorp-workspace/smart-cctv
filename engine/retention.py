"""Data retention and auto-purge management for database records and snapshot files."""

import datetime
import os
import sqlite3
from pathlib import Path
from typing import Dict, Optional

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
