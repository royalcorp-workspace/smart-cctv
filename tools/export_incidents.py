"""Incident Log Exporter to CSV report."""

import argparse
import csv
import datetime
import sqlite3
import sys
from pathlib import Path
from typing import Optional, Tuple

# Ensure workspace root is in sys.path
_WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_DIR))

from storage.db import DEFAULT_DB_PATH, get_connection


def export_incidents_to_csv(
    output_path: Optional[Path] = None,
    camera_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> Tuple[Path, int]:
    """Export incident database records to a structured CSV file.

    Returns: (output_csv_path, row_count)
    """
    target_db: Path = db_path if db_path is not None else DEFAULT_DB_PATH
    if not target_db.exists():
        raise FileNotFoundError(f"Database file not found: {target_db}")

    reports_dir: Path = _WORKSPACE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        target_csv: Path = reports_dir / f"incident_report_{timestamp_str}.csv"
    else:
        target_csv = output_path
        target_csv.parent.mkdir(parents=True, exist_ok=True)

    conn = get_connection(target_db)
    try:
        cursor = conn.cursor()

        # Check existing table name (event_logs or incident_logs)
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('event_logs', 'incident_logs');"
        )
        tables = [r[0] for r in cursor.fetchall()]
        if not tables:
            target_table = "event_logs"
        else:
            target_table = "event_logs" if "event_logs" in tables else tables[0]

        query = f"SELECT * FROM {target_table}"
        params = []
        if camera_id:
            query += " WHERE camera_id = ?"
            params.append(camera_id)
        query += " ORDER BY id DESC;"

        cursor.execute(query, params)
        rows = cursor.fetchall()
    finally:
        conn.close()

    headers = [
        "ID",
        "Camera_ID",
        "Zone_ID",
        "Start_Time",
        "End_Time",
        "Duration_Seconds",
        "Snapshot_Path",
        "Status_Resolved",
    ]

    with open(target_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for row in rows:
            end_time = row["resolved_time"] if row["resolved_time"] else (row["trigger_time"] or "-")
            status_text = "Resolved" if row["is_resolved"] == 1 else "Active / Unresolved"
            duration = round(float(row["dwell_duration"]), 2) if row["dwell_duration"] is not None else 0.0

            writer.writerow([
                row["id"],
                row["camera_id"],
                row["zone_id"],
                row["start_time"] or "-",
                end_time,
                duration,
                row["snapshot_path"] or "-",
                status_text,
            ])

    return target_csv, len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart CCTV Incident CSV Exporter")
    parser.add_argument(
        "--cam",
        type=str,
        default=None,
        help="Optional camera_id filter (e.g. cam_01)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Custom output CSV file path",
    )
    args = parser.parse_args()

    custom_out = Path(args.out) if args.out else None

    print("==================================================")
    print("      Smart CCTV 2.0 - Incident Report Exporter   ")
    print("==================================================")

    try:
        csv_file, count = export_incidents_to_csv(output_path=custom_out, camera_id=args.cam)
        print(f"[SUCCESS] Exported {count} incident record(s) to:")
        print(f"  -> {csv_file}")
    except Exception as e:
        print(f"[ERROR] Failed to export incidents: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
