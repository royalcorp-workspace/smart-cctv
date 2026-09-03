"""Standalone CLI utility for pruning expired snapshots and database event records."""

import argparse
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
_WORKSPACE_DIR = Path(__file__).resolve().parent.parent
if str(_WORKSPACE_DIR) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_DIR))

from engine.retention import cleanup_old_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Smart CCTV Data Retention & Auto-Purge Utility")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Retention threshold in days (default: 30). Records older than this will be deleted.",
    )
    args = parser.parse_args()

    if args.days < 1:
        print("[ERROR] Retention days must be at least 1.")
        sys.exit(1)

    print("==================================================")
    print("      Smart CCTV 2.0 - Data Retention Purge       ")
    print("==================================================")
    print(f"[INFO] Initiating purge for records older than {args.days} days...")

    stats = cleanup_old_records(retention_days=args.days)

    print("\n--- Maintenance Summary ---")
    print(f"  Deleted DB records   : {stats['deleted_records']}")
    print(f"  Deleted snapshot files: {stats['deleted_snapshots']}")
    print(f"  Failed file deletions : {stats['failed_snapshots']}")
    print("[SUCCESS] Retention maintenance completed.\n")


if __name__ == "__main__":
    main()
