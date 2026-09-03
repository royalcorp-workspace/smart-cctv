"""SQLite persistence layer with WAL mode and thread-safe operations."""
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

DEFAULT_DB_PATH: Path = Path(__file__).resolve().parent.parent / "storage" / "events.db"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Create and configure a thread-isolated SQLite connection."""
    target_path: Path = db_path if db_path is not None else DEFAULT_DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn: sqlite3.Connection = sqlite3.connect(str(target_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    
    # Configure concurrency and integrity pragmas
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize database tables and indexes."""
    with get_connection(db_path) as conn:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS event_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                zone_id TEXT NOT NULL,
                track_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                dwell_duration REAL NOT NULL,
                start_time TIMESTAMP NOT NULL,
                trigger_time TIMESTAMP NOT NULL,
                snapshot_path TEXT,
                resolved_time TIMESTAMP,
                is_resolved INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_event_logs_camera ON event_logs (camera_id);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_event_logs_trigger ON event_logs (trigger_time);"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_event_logs_resolved ON event_logs (is_resolved);"
        )
        conn.commit()


def log_event(
    camera_id: str,
    zone_id: str,
    track_id: int,
    event_type: str,
    dwell_duration: float,
    start_time: str,
    trigger_time: str,
    snapshot_path: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Insert a violation event log and return its generated ID."""
    with get_connection(db_path) as conn:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO event_logs (
                camera_id, zone_id, track_id, event_type,
                dwell_duration, start_time, trigger_time,
                snapshot_path, is_resolved
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0);
            """,
            (
                camera_id,
                zone_id,
                track_id,
                event_type,
                dwell_duration,
                start_time,
                trigger_time,
                snapshot_path,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def resolve_event(
    event_id: int,
    resolved_time: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> bool:
    """Mark an event as resolved when object is removed."""
    res_time: str = resolved_time if resolved_time else datetime.now().isoformat()
    with get_connection(db_path) as conn:
        cursor: sqlite3.Cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE event_logs
            SET is_resolved = 1, resolved_time = ?
            WHERE id = ?;
            """,
            (res_time, event_id),
        )
        conn.commit()
        return cursor.rowcount > 0


def get_unresolved_events(
    camera_id: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Fetch active/unresolved incidents."""
    with get_connection(db_path) as conn:
        cursor: sqlite3.Cursor = conn.cursor()
        if camera_id:
            cursor.execute(
                """
                SELECT * FROM event_logs
                WHERE is_resolved = 0 AND camera_id = ?
                ORDER BY trigger_time DESC;
                """,
                (camera_id,),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM event_logs
                WHERE is_resolved = 0
                ORDER BY trigger_time DESC;
                """
            )
        rows: List[sqlite3.Row] = cursor.fetchall()
        return [dict(row) for row in rows]
