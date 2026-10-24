"""SQLite storage. The database doubles as the work queue: each worker picks up
rows that are missing the data it is responsible for, so nothing is lost when
the container restarts."""

from __future__ import annotations

import os
import sqlite3
import time

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS edits (
    rev_id          INTEGER PRIMARY KEY,
    old_rev_id      INTEGER,
    rc_id           INTEGER,
    page_id         INTEGER,          -- filled in from the diff response
    title           TEXT NOT NULL,
    namespace       INTEGER NOT NULL,
    user            TEXT,
    is_bot          INTEGER NOT NULL, -- bot flag on the edit
    is_anon         INTEGER NOT NULL, -- IP or temporary account
    is_temp         INTEGER NOT NULL, -- temporary account (~2025-12345-67)
    comment         TEXT,
    minor           INTEGER NOT NULL,
    patrolled       INTEGER,          -- only present on some wikis
    size_old        INTEGER,
    size_new        INTEGER,
    size_delta      INTEGER,
    ts              INTEGER NOT NULL, -- edit time, unix seconds
    meta_dt         TEXT,
    received_at     REAL NOT NULL,    -- when the collector saw it

    -- Enrichment, fetched about a minute after the edit.
    enriched_at         REAL,
    user_registration   INTEGER,      -- unix seconds, NULL if unknown or very old
    user_editcount      INTEGER,      -- at fetch time, so it includes this edit
    user_groups         TEXT,         -- comma-separated
    tags_at_save        TEXT          -- comma-separated change tags shortly after saving
);
CREATE INDEX IF NOT EXISTS edits_ts ON edits(ts);
CREATE INDEX IF NOT EXISTS edits_enriched ON edits(enriched_at);

CREATE TABLE IF NOT EXISTS diffs (
    rev_id          INTEGER PRIMARY KEY REFERENCES edits(rev_id),
    status          TEXT NOT NULL,    -- ok | skipped | api_error:<code> | failed
    attempts        INTEGER NOT NULL DEFAULT 1,
    fetched_at      REAL NOT NULL,
    diff_size       INTEGER,          -- bytes of diff HTML
    added_text      TEXT,             -- inserted words plus wholly added lines
    removed_text    TEXT,             -- deleted words plus wholly removed lines
    added_lines     TEXT,             -- full new version of every changed line
    removed_lines   TEXT,             -- full old version of every changed line
    lines_added     INTEGER,
    lines_removed   INTEGER,
    lines_changed   INTEGER,
    moved_paras     INTEGER,
    error           TEXT,
    html_z          BLOB              -- zlib-compressed diff HTML, to re-extract later
);

CREATE TABLE IF NOT EXISTS labels (
    rev_id          INTEGER PRIMARY KEY REFERENCES edits(rev_id),
    checked_at      REAL NOT NULL,
    status          TEXT NOT NULL,    -- ok | missing | failed
    attempts        INTEGER NOT NULL DEFAULT 1,
    tags            TEXT,             -- tags on the revision at check time
    reverted        INTEGER,          -- carries the mw-reverted tag
    revert_rev_id   INTEGER,          -- the reverting revision, if found
    revert_ts       INTEGER,
    revert_seconds  INTEGER,          -- revert_ts - edit ts
    revert_method   TEXT,             -- mw-rollback | mw-undo | mw-manual-revert | sha1
    reverter_is_self INTEGER,
    reverted_48h    INTEGER           -- reverted and revert_seconds <= label delay
);

CREATE TABLE IF NOT EXISTS state (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def connect(path: str) -> sqlite3.Connection:
    if path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=60, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    set_state(conn, "schema_version", str(SCHEMA_VERSION), only_if_missing=True)


def get_state(conn: sqlite3.Connection, key: str) -> str | None:
    row = conn.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def set_state(
    conn: sqlite3.Connection, key: str, value: str, only_if_missing: bool = False
) -> None:
    verb = "INSERT OR IGNORE" if only_if_missing else "INSERT OR REPLACE"
    conn.execute(f"{verb} INTO state(key, value) VALUES (?, ?)", (key, value))


EDIT_COLUMNS = (
    "rev_id old_rev_id rc_id title namespace user is_bot is_anon is_temp comment "
    "minor patrolled size_old size_new size_delta ts meta_dt received_at"
).split()


def insert_edits(conn: sqlite3.Connection, edits: list[dict], last_event_id: str | None) -> int:
    """Insert edits and save the stream position in one transaction, so a crash
    can never move the position past edits that were not stored."""
    placeholders = ",".join("?" for _ in EDIT_COLUMNS)
    sql = f"INSERT OR IGNORE INTO edits({','.join(EDIT_COLUMNS)}) VALUES ({placeholders})"
    conn.execute("BEGIN IMMEDIATE")
    try:
        before = conn.total_changes
        conn.executemany(sql, [tuple(e[c] for c in EDIT_COLUMNS) for e in edits])
        inserted = conn.total_changes - before
        if last_event_id is not None:
            set_state(conn, "last_event_id", last_event_id)
            set_state(conn, "last_event_at", str(time.time()))
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    return inserted
