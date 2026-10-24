"""Turning recentchange events into edit rows."""

from __future__ import annotations

import json
import time

from .users import is_anonymous, is_temp_account


def parse_edit(raw: str, wiki: str = "enwiki", namespace: int = 0) -> dict | None:
    """Return an edit row for a matching event, or None to skip it."""
    try:
        ev = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(ev, dict):
        return None
    meta = ev.get("meta") or {}
    # Canary events are synthetic test events emitted into every stream.
    if meta.get("domain") == "canary":
        return None
    if ev.get("wiki") != wiki or ev.get("type") != "edit" or ev.get("namespace") != namespace:
        return None
    rev = ev.get("revision") or {}
    new_id = rev.get("new")
    if not isinstance(new_id, int):
        return None
    length = ev.get("length") or {}
    size_old, size_new = length.get("old"), length.get("new")
    user = ev.get("user")
    patrolled = ev.get("patrolled")
    return {
        "rev_id": new_id,
        "old_rev_id": rev.get("old"),
        "rc_id": ev.get("id"),
        "title": ev.get("title") or "",
        "namespace": ev.get("namespace"),
        "user": user,
        "is_bot": int(bool(ev.get("bot"))),
        "is_anon": int(is_anonymous(user)),
        "is_temp": int(is_temp_account(user)),
        "comment": ev.get("comment"),
        "minor": int(bool(ev.get("minor"))),
        "patrolled": None if patrolled is None else int(bool(patrolled)),
        "size_old": size_old,
        "size_new": size_new,
        "size_delta": (size_new - size_old)
        if isinstance(size_old, int) and isinstance(size_new, int)
        else None,
        "ts": int(ev.get("timestamp") or 0),
        "meta_dt": meta.get("dt"),
        "received_at": time.time(),
    }
