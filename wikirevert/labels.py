"""Revert labels, checked once an edit is 48 hours old.

Step 1, in batches of 50: read each revision's change tags. MediaWiki adds
mw-reverted to an edit when a later edit undoes it (undo, rollback, or a manual
edit that restores an earlier version exactly). Revisions on deleted pages, or
that were removed, come back as "badrevids" and are stored as missing.

Step 2, only for reverted edits: read the page history from the edit's parent
onwards to find the reverting revision and how long the revert took. The
reverting revision is the first later revision that carries mw-rollback,
mw-undo or mw-manual-revert, or that restores the parent's exact content (same
sha1). This is a good approximation, not a certainty: when several reverts
happen close together it can pick the wrong one.
"""

from __future__ import annotations

import logging
import sqlite3
import time

from .enrich import parse_ts
from .http import ApiClient, ApiError

log = logging.getLogger(__name__)

BATCH = 50
REVERT_TAGS = ("mw-rollback", "mw-undo", "mw-manual-revert")
MAX_ATTEMPTS = 5


def pending(conn: sqlite3.Connection, cutoff_ts: int, limit: int = BATCH) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT e.rev_id, e.old_rev_id, e.page_id, e.user, e.ts
        FROM edits e LEFT JOIN labels l ON l.rev_id = e.rev_id
        WHERE e.ts <= ?
          AND (l.rev_id IS NULL
               OR (l.status = 'failed' AND l.attempts < ? AND l.checked_at < ?))
        ORDER BY e.rev_id
        LIMIT ?
        """,
        (cutoff_ts, MAX_ATTEMPTS, time.time() - 3600, limit),
    ).fetchall()


def parse_tags(data: dict) -> tuple[dict[int, dict], set[int]]:
    q = data.get("query") or {}
    found: dict[int, dict] = {}
    for page in q.get("pages") or []:
        for rev in page.get("revisions") or []:
            found[rev["revid"]] = {"tags": rev.get("tags") or [], "page_id": page.get("pageid")}
    missing = {int(k) for k in (q.get("badrevids") or {})}
    return found, missing


def find_revert(history: dict, rev_id: int, parent_id: int | None) -> dict | None:
    """Pick the reverting revision out of a page history that starts at the
    edit's parent (or the edit itself) and runs forwards in time."""
    revs = []
    for page in (history.get("query") or {}).get("pages") or []:
        revs.extend(page.get("revisions") or [])
    parent_sha1 = next((r.get("sha1") for r in revs if r["revid"] == parent_id), None)
    seen_edit = False
    for r in revs:
        if r["revid"] == rev_id:
            seen_edit = True
            continue
        if not seen_edit:
            continue
        tags = r.get("tags") or []
        method = next((t for t in REVERT_TAGS if t in tags), None)
        if method is None and parent_sha1 and r.get("sha1") == parent_sha1:
            method = "sha1"
        if method:
            return {
                "revert_rev_id": r["revid"],
                "revert_ts": parse_ts(r.get("timestamp")),
                "revert_method": method,
                "reverter": r.get("user"),
            }
    return None


def fetch_history(api: ApiClient, page_id: int, start_rev: int) -> dict:
    return api.get(
        {
            "action": "query",
            "prop": "revisions",
            "pageids": str(page_id),
            "rvstartid": str(start_rev),
            "rvdir": "newer",
            "rvlimit": "50",
            "rvprop": "ids|timestamp|tags|sha1|user",
        }
    )


def run_batch(conn: sqlite3.Connection, api: ApiClient, delay_hours: float) -> int:
    delay = int(delay_hours * 3600)
    rows = pending(conn, int(time.time()) - delay)
    if not rows:
        return 0
    data = api.get(
        {
            "action": "query",
            "prop": "revisions",
            "revids": "|".join(str(r["rev_id"]) for r in rows),
            "rvprop": "ids|tags",
        }
    )
    found, missing = parse_tags(data)
    now = time.time()
    results = []
    for r in rows:
        rid = r["rev_id"]
        if rid in missing or rid not in found:
            results.append(_row(rid, now, "missing"))
            continue
        tags = found[rid]["tags"]
        reverted = "mw-reverted" in tags
        row = _row(rid, now, "ok", tags=",".join(tags), reverted=int(reverted))
        if reverted:
            page_id = r["page_id"] or found[rid]["page_id"]
            try:
                hist = fetch_history(api, page_id, r["old_rev_id"] or rid)
                rv = find_revert(hist, rid, r["old_rev_id"])
            except ApiError as exc:
                log.warning("History lookup failed for %s: %s", rid, exc.code)
                rv = None
            if rv:
                secs = rv["revert_ts"] - r["ts"] if rv["revert_ts"] is not None else None
                row.update(
                    revert_rev_id=rv["revert_rev_id"],
                    revert_ts=rv["revert_ts"],
                    revert_seconds=secs,
                    revert_method=rv["revert_method"],
                    reverter_is_self=int(bool(r["user"]) and rv["reverter"] == r["user"]),
                    reverted_48h=None if secs is None else int(secs <= delay),
                )
        else:
            row["reverted_48h"] = 0
        results.append(row)
    _save(conn, results)
    return len(rows)


LABEL_COLUMNS = (
    "rev_id checked_at status tags reverted revert_rev_id revert_ts revert_seconds "
    "revert_method reverter_is_self reverted_48h"
).split()


def _row(rev_id: int, now: float, status: str, **kw) -> dict:
    row = {c: None for c in LABEL_COLUMNS}
    row.update(rev_id=rev_id, checked_at=now, status=status, **kw)
    return row


def _save(conn: sqlite3.Connection, rows: list[dict]) -> None:
    cols = ",".join(LABEL_COLUMNS)
    ph = ",".join("?" for _ in LABEL_COLUMNS)
    conn.execute("BEGIN")
    conn.executemany(
        f"INSERT OR REPLACE INTO labels({cols}, attempts) VALUES ({ph}, "
        "COALESCE((SELECT attempts FROM labels WHERE rev_id = ?), 0) + 1)",
        [tuple(r[c] for c in LABEL_COLUMNS) + (r["rev_id"],) for r in rows],
    )
    conn.execute("COMMIT")


def mark_failed(conn: sqlite3.Connection, rev_ids: list[int]) -> None:
    _save(conn, [_row(rid, time.time(), "failed") for rid in rev_ids])
