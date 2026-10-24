"""Editor details and save-time tags, fetched in batches of 50 edits.

One action=query request covers up to 50 edits: list=users for the editors
(registration date, edit count, groups) and prop=revisions for each revision's
change tags. It runs about a minute after the edit, so tags added on save (edit
filters, editing tool, mobile, etc.) are there, but most reverts have not
happened yet.

The mw-reverted tag is dropped here even if a revert already happened, because
it is the label. Phase 2 should still treat everything fetched after the edit
with suspicion (see README, "Leakage").
"""

from __future__ import annotations

import sqlite3
import time
from datetime import datetime

from .http import ApiClient
from .users import is_ip

BATCH = 50
OUTCOME_TAGS = {"mw-reverted"}


def parse_ts(value: str | None) -> int | None:
    if not value:
        return None
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())


def pending(conn: sqlite3.Connection, older_than: float, limit: int = BATCH) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT rev_id, user FROM edits WHERE enriched_at IS NULL AND received_at <= ? "
        "ORDER BY rev_id LIMIT ?",
        (older_than, limit),
    ).fetchall()


def build_params(rows: list[sqlite3.Row]) -> dict[str, str]:
    users = sorted({r["user"] for r in rows if r["user"] and not is_ip(r["user"])})
    params = {
        "action": "query",
        "prop": "revisions",
        "revids": "|".join(str(r["rev_id"]) for r in rows),
        "rvprop": "ids|tags",
    }
    if users:
        params.update(
            {"list": "users", "ususers": "|".join(users), "usprop": "registration|editcount|groups"}
        )
    return params


def parse_response(data: dict) -> tuple[dict[int, list[str]], dict[str, dict]]:
    q = data.get("query") or {}
    tags: dict[int, list[str]] = {}
    for page in q.get("pages") or []:
        for rev in page.get("revisions") or []:
            tags[rev["revid"]] = [t for t in rev.get("tags") or [] if t not in OUTCOME_TAGS]
    users: dict[str, dict] = {}
    for u in q.get("users") or []:
        if u.get("missing") or u.get("invalid"):
            continue
        users[u["name"]] = {
            "registration": parse_ts(u.get("registration")),
            "editcount": u.get("editcount"),
            "groups": ",".join(g for g in u.get("groups") or [] if g != "*"),
        }
    return tags, users


def run_batch(conn: sqlite3.Connection, api: ApiClient, delay_seconds: int) -> int:
    rows = pending(conn, time.time() - delay_seconds)
    if not rows:
        return 0
    data = api.get(build_params(rows))
    tags, users = parse_response(data)
    now = time.time()
    updates = []
    for r in rows:
        u = users.get(r["user"] or "", {})
        t = tags.get(r["rev_id"])
        updates.append(
            (
                now,
                u.get("registration"),
                u.get("editcount"),
                u.get("groups"),
                None if t is None else ",".join(t),
                r["rev_id"],
            )
        )
    conn.execute("BEGIN")
    conn.executemany(
        "UPDATE edits SET enriched_at=?, user_registration=?, user_editcount=?, "
        "user_groups=?, tags_at_save=? WHERE rev_id=?",
        updates,
    )
    conn.execute("COMMIT")
    return len(rows)
