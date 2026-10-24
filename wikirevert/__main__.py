"""Command line: python -m wikirevert {run,stats,health}"""

from __future__ import annotations

import argparse
import json
import logging
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

    conn = db.connect(settings.db_path)
    db.init(conn)
    if args.cmd == "stats":
        print(json.dumps(stats(conn), indent=2))
        return 0
    if args.cmd == "health":
        last = db.get_state(conn, "last_event_at")
        ok = last is not None and time.time() - float(last) < args.max_age
        print("ok" if ok else "stale")
        return 0 if ok else 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
