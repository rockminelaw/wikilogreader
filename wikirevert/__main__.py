"""Command line: python -m wikirevert {run,stats,health,backup}"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time

from . import config, db
from .collector import Collector, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="wikirevert")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="collect edits, diffs and labels until stopped")
    sub.add_parser("stats", help="print counts from the database as JSON")
    h = sub.add_parser("health", help="exit 1 if no stream event has been stored recently")
    h.add_argument("--max-age", type=int, default=600, help="seconds (default 600)")
    b = sub.add_parser("backup", help="copy the database safely while the collector runs")
    b.add_argument("dest", help="file to write")
    args = parser.parse_args(argv)

    settings = config.load()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(threadName)s %(message)s",
        stream=sys.stdout,
    )

    if args.cmd == "run":
        Collector(settings).run()
        return 0

    # The other commands only read, and never create the database.
    if not os.path.exists(settings.db_path):
        print(f"no database at {settings.db_path}")
        return 1
    conn = db.connect(settings.db_path)

    if args.cmd == "stats":
        print(json.dumps(stats(conn), indent=2))
        return 0
    if args.cmd == "health":
        try:
            last = db.get_state(conn, "last_event_at")
        except sqlite3.OperationalError:
            last = None
        ok = last is not None and time.time() - float(last) < args.max_age
        print("ok" if ok else "stale")
        return 0 if ok else 1
    if args.cmd == "backup":
        tmp = args.dest + ".part"
        dest = sqlite3.connect(tmp)
        conn.backup(dest)  # consistent snapshot, safe while the collector writes
        dest.close()
        os.replace(tmp, args.dest)
        print(f"wrote {args.dest} ({os.path.getsize(args.dest) / 1e6:.1f} MB)")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
