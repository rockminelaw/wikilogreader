"""Compact training export: one gzipped JSON-lines file per UTC day.

Only days whose edits have all been labelled are written, so every file is
complete and the time split in phase 2 can work file by file. Files that
already exist are skipped, so running it again only adds new days.

What is left out on purpose:
- the diff HTML (the extracted text is kept, cut to --max-chars per field);
- usernames, replaced by a salted hash so edits by one editor can still be
  grouped (the salt stays in the database, so the hash can't be reversed
  from the export alone);
- the tags read 48 hours later (labels.tags), which would leak the label.
Edit comments are kept as they are; they sometimes mention other editors.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import secrets
import sqlite3
import time
from datetime import datetime, timezone

from . import db

DAY = 86400
TEXT_FIELDS = ("added_text", "removed_text", "added_lines", "removed_lines")

QUERY = """
SELECT e.rev_id, e.old_rev_id, e.page_id, e.title, e.ts, e.user,
       e.is_bot, e.is_anon, e.is_temp, e.minor, e.comment,
       e.size_old, e.size_new, e.size_delta,
       e.user_registration, e.user_editcount, e.user_groups, e.tags_at_save,
       e.enriched_at - e.ts AS enrich_delay,
       d.status AS diff_status, d.fetched_at - e.ts AS diff_delay,
       d.added_text, d.removed_text, d.added_lines, d.removed_lines,
       d.lines_added, d.lines_removed, d.lines_changed, d.moved_paras,
       l.status AS label_status, l.reverted, l.reverted_48h, l.revert_seconds,
       l.revert_method, l.reverter_is_self
FROM edits e
LEFT JOIN diffs d ON d.rev_id = e.rev_id
LEFT JOIN labels l ON l.rev_id = e.rev_id
WHERE e.ts >= ? AND e.ts < ?
ORDER BY e.ts, e.rev_id
"""


def _salt(conn: sqlite3.Connection) -> str:
    salt = db.get_state(conn, "export_salt")
    if salt is None:
        db.set_state(conn, "export_salt", secrets.token_hex(16), only_if_missing=True)
        salt = db.get_state(conn, "export_salt")
    return salt


def user_hash(salt: str, user: str | None) -> str | None:
    if not user:
        return None
    return hashlib.sha256(f"{salt}\0{user}".encode()).hexdigest()[:16]


def ready_days(conn: sqlite3.Connection, label_delay_hours: float, now: float):
    """Yield (day_start, status) for every day with edits, where status is
    'ready' or the reason it isn't."""
    lo, hi = conn.execute("SELECT MIN(ts), MAX(ts) FROM edits").fetchone()
    if lo is None:
        return
    start = lo - lo % DAY
    # Leave a two-hour margin after the label delay for the label backlog.
    cutoff = now - label_delay_hours * 3600 - 2 * 3600
    while start <= hi:
        end = start + DAY
        if end > cutoff:
            yield start, "not old enough to be fully labelled yet"
        else:
            unlabelled = conn.execute(
                "SELECT COUNT(*) FROM edits e LEFT JOIN labels l ON l.rev_id = e.rev_id "
                "WHERE e.ts >= ? AND e.ts < ? AND (l.rev_id IS NULL OR l.status = 'failed')",
                (start, end),
            ).fetchone()[0]
            yield start, "ready" if unlabelled == 0 else f"{unlabelled} edits still unlabelled"
        start = end


def export(
    conn: sqlite3.Connection,
    out_dir: str,
    label_delay_hours: float = 48,
    max_chars: int = 3000,
    now: float | None = None,
) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    salt = _salt(conn)
    now = time.time() if now is None else now
    lines = []
    for start, status in ready_days(conn, label_delay_hours, now):
        day = datetime.fromtimestamp(start, timezone.utc).strftime("%Y-%m-%d")
        path = os.path.join(out_dir, f"edits-{day}.jsonl.gz")
        if os.path.exists(path):
            lines.append(f"{day}  already exported")
            continue
        if status != "ready":
            lines.append(f"{day}  skipped: {status}")
            continue
        n = reverted = 0
        tmp = path + ".part"
        cur = conn.execute(QUERY, (start, start + DAY))
        cols = [c[0] for c in cur.description]
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as f:
            for row in cur:
                rec = dict(zip(cols, row))
                rec["user_hash"] = user_hash(salt, rec.pop("user"))
                for k in TEXT_FIELDS:
                    if rec[k] is not None and len(rec[k]) > max_chars:
                        rec[k] = rec[k][:max_chars]
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                reverted += rec["reverted_48h"] == 1
        os.replace(tmp, path)
        size = os.path.getsize(path) / 1e6
        lines.append(f"{day}  wrote {n} edits, {reverted} reverted within 48 h, {size:.1f} MB")
    return lines
