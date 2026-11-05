"""The long-running collector.

Threads:
- stream: reads EventStreams, keeps English Wikipedia main-namespace edits,
  stores them with the stream position in the same transaction.
- api-1 .. api-N (N = api_concurrency, default 3): call the Action API. All of
  them fetch diffs; api-1 also enriches edits in batches of 50 after a minute
  and labels them in batches of 50 after 48 hours. Diffs for edits from the
  last hour come first, so live edits get theirs within seconds; spare
  capacity works down through older edits that are still missing one.
  Bot edits are marked "skipped" in bulk unless fetch_bot_diffs is on.

The main thread logs progress and handles shutdown.
"""

from __future__ import annotations

import logging
import signal
import sqlite3
import threading
import time

from . import db, diffs, enrich, labels
from .config import Settings
from .http import ApiClient, ApiError, TransientError
from .sse import StreamClient
from .stream import parse_edit

log = logging.getLogger("wikirevert")

DIFF_MAX_ATTEMPTS = 5


class Collector:
    def __init__(self, settings: Settings):
        self.s = settings
        self.stop = threading.Event()
        self.api = ApiClient(settings)
        self.stream_events = 0
        self.stream_edits = 0
        self.stream_client: StreamClient | None = None
        self._init_claims()

    # ---------- stream thread ----------

    def run_stream(self) -> None:
        conn = db.connect(self.s.db_path)
        last_id = db.get_state(conn, "last_event_id")
        client = StreamClient(
            self.s.stream_url, self.s.user_agent, last_event_id=last_id, stop=self.stop
        )
        self.stream_client = client
        buf: list[dict] = []
        last_flush = time.monotonic()
        first = True
        try:
            for ev in client.events():
                self.stream_events += 1
                edit = parse_edit(ev.data, self.s.wiki, self.s.namespace)
                if edit:
                    if first:
                        lag = time.time() - edit["ts"]
                        log.info("First edit after connecting is %.0f s old", lag)
                        first = False
                    buf.append(edit)
                if buf and (len(buf) >= 50 or time.monotonic() - last_flush > 2) or (
                    time.monotonic() - last_flush > 10
                ):
                    self.stream_edits += db.insert_edits(conn, buf, client.last_event_id)
                    buf = []
                    last_flush = time.monotonic()
                if self.stop.is_set():
                    break
        except Exception:
            log.exception("Stream thread crashed")
            self.stop.set()
        finally:
            if buf:
                db.insert_edits(conn, buf, client.last_event_id)
            conn.close()

    # ---------- api threads ----------

    PENDING = """
        SELECT e.rev_id, e.old_rev_id, e.is_bot, d.attempts
        FROM edits e LEFT JOIN diffs d ON d.rev_id = e.rev_id
        WHERE {where}
          AND (d.rev_id IS NULL
               OR (d.status = 'failed' AND d.attempts < ? AND d.fetched_at < ?))
        ORDER BY e.rev_id DESC LIMIT ?
    """

    def _init_claims(self) -> None:
        self._claim_lock = threading.Lock()
        self._in_flight: set[int] = set()
        self._queue: list = []  # backlog rows, newest first
        self._cursor: int | None = None  # backlog scan position (rev_id)
        self._next_skip = 0.0
        self.diffs_fetched = 0

    def skip_bot_diffs(self, conn: sqlite3.Connection) -> int:
        if self.s.fetch_bot_diffs:
            return 0
        conn.execute("BEGIN")
        cur = conn.execute(
            "INSERT OR IGNORE INTO diffs(rev_id, status, attempts, fetched_at) "
            "SELECT e.rev_id, 'skipped', 0, ? FROM edits e "
            "WHERE e.is_bot = 1 AND NOT EXISTS (SELECT 1 FROM diffs d WHERE d.rev_id = e.rev_id)",
            (time.time(),),
        )
        conn.execute("COMMIT")
        return cur.rowcount

    def _claim(self, conn: sqlite3.Connection):
        """Hand out one edit that needs a diff, never the same one to two threads."""
        retry_before = time.time() - 600
        with self._claim_lock:
            if time.time() >= self._next_skip:
                n = self.skip_bot_diffs(conn)
                if n:
                    log.info("Marked %d bot edits as not needing a diff", n)
                self._next_skip = time.time() + 300
            fresh = conn.execute(
                self.PENDING.format(where="e.ts >= ?"),
                (int(time.time()) - 3600, DIFF_MAX_ATTEMPTS, retry_before, 50),
            ).fetchall()
            for row in fresh:
                if row["rev_id"] not in self._in_flight:
                    self._in_flight.add(row["rev_id"])
                    return row
            for _ in range(2):
                while self._queue:
                    row = self._queue.pop(0)
                    if row["rev_id"] not in self._in_flight:
                        self._in_flight.add(row["rev_id"])
                        return row
                where, args = ("e.rev_id < ?", (self._cursor,)) if self._cursor else ("1 = 1", ())
                self._queue = conn.execute(
                    self.PENDING.format(where=where),
                    (*args, DIFF_MAX_ATTEMPTS, retry_before, 200),
                ).fetchall()
                if self._queue:
                    self._cursor = self._queue[-1]["rev_id"]
                elif self._cursor is not None:
                    self._cursor = None  # reached the oldest edit; start a new sweep
                else:
                    return None
            return None

    def _release(self, rev_id: int) -> None:
        with self._claim_lock:
            self._in_flight.discard(rev_id)

    def diff_step(self, conn: sqlite3.Connection) -> int:
        row = self._claim(conn)
        if row is None:
            return 0
        try:
            return self._fetch_one(conn, row)
        finally:
            self._release(row["rev_id"])

    def _fetch_one(self, conn: sqlite3.Connection, row) -> int:
        rid, attempts = row["rev_id"], (row["attempts"] or 0) + 1
        if not row["old_rev_id"] or (row["is_bot"] and not self.s.fetch_bot_diffs):
            self._save_diff(conn, {"rev_id": rid, "status": "skipped", "attempts": attempts})
            return 1
        try:
            result = diffs.fetch(self.api, row["old_rev_id"], rid)
        except ApiError as exc:
            # The revision or page is gone or hidden; retrying won't help.
            self._save_diff(
                conn,
                {"rev_id": rid, "status": f"api_error:{exc.code}", "attempts": attempts,
                 "error": exc.info[:300]},
            )
            return 1
        except TransientError as exc:
            self._save_diff(
                conn, {"rev_id": rid, "status": "failed", "attempts": attempts, "error": str(exc)}
            )
            raise
        out = diffs.to_row(rid, result, time.time(), self.s.max_text_chars)
        out["attempts"] = attempts
        self._save_diff(conn, out, page_id=result.page_id)
        self.diffs_fetched += 1
        return 1

    def _save_diff(self, conn: sqlite3.Connection, row: dict, page_id: int | None = None) -> None:
        row.setdefault("fetched_at", time.time())
        cols = list(row)
        conn.execute("BEGIN")
        conn.execute(
            f"INSERT OR REPLACE INTO diffs({','.join(cols)}) VALUES ({','.join('?' for _ in cols)})",
            [row[c] for c in cols],
        )
        if page_id:
            conn.execute("UPDATE edits SET page_id = ? WHERE rev_id = ?", (page_id, row["rev_id"]))
        conn.execute("COMMIT")

    def run_api(self, batches: bool = True) -> None:
        conn = db.connect(self.s.db_path)
        next_enrich = next_label = 0.0
        try:
            while not self.stop.is_set():
                worked = 0
                try:
                    now = time.time()
                    if batches and now >= next_enrich:
                        n = self._guard_batch(
                            conn, lambda: enrich.run_batch(conn, self.api, self.s.enrich_delay_seconds),
                            on_api_error=self._skip_enrich,
                        )
                        worked += n
                        next_enrich = now + (10 if n >= enrich.BATCH else 30)
                    if batches and now >= next_label:
                        n = self._guard_batch(
                            conn, lambda: labels.run_batch(conn, self.api, self.s.label_delay_hours),
                            on_api_error=self._fail_labels,
                        )
                        worked += n
                        next_label = now + (20 if n >= labels.BATCH else 120)
                    worked += self.diff_step(conn)
                except TransientError as exc:
                    log.warning("API unavailable (%s); pausing for a minute", exc)
                    self.stop.wait(60)
                    continue
                if not worked:
                    self.stop.wait(1)
        except Exception:
            log.exception("API thread crashed")
            self.stop.set()
        finally:
            conn.close()

    def _guard_batch(self, conn, fn, on_api_error) -> int:
        try:
            return fn()
        except ApiError as exc:
            log.error("Batch rejected by the API (%s); setting it aside", exc.code)
            on_api_error(conn)
            return 0

    def _skip_enrich(self, conn: sqlite3.Connection) -> None:
        rows = enrich.pending(conn, time.time() - self.s.enrich_delay_seconds)
        conn.execute("BEGIN")
        conn.executemany(
            "UPDATE edits SET enriched_at = ? WHERE rev_id = ?",
            [(time.time(), r["rev_id"]) for r in rows],
        )
        conn.execute("COMMIT")

    def _fail_labels(self, conn: sqlite3.Connection) -> None:
        cutoff = int(time.time() - self.s.label_delay_hours * 3600)
        labels.mark_failed(conn, [r["rev_id"] for r in labels.pending(conn, cutoff)])

    # ---------- main ----------

    def run(self) -> None:
        conn = db.connect(self.s.db_path)
        db.init(conn)
        log.info("Collector starting. User-Agent: %s", self.s.user_agent)
        log.info("Database: %s", self.s.db_path)

        def handle(signum, _frame):
            log.info("Signal %d received, stopping", signum)
            self.stop.set()

        signal.signal(signal.SIGTERM, handle)
        signal.signal(signal.SIGINT, handle)

        threads = [threading.Thread(target=self.run_stream, name="stream", daemon=True)]
        for i in range(max(1, self.s.api_concurrency)):
            threads.append(threading.Thread(
                target=self.run_api, kwargs={"batches": i == 0}, name=f"api-{i + 1}", daemon=True))
        for t in threads:
            t.start()
        next_stats = time.time() + 60
        while not self.stop.is_set():
            self.stop.wait(5)
            if time.time() >= next_stats:
                self.log_stats(conn)
                next_stats = time.time() + self.s.stats_interval_seconds
        for t in threads:
            t.join(timeout=20)
        self.log_stats(conn)
        conn.close()
        log.info("Collector stopped")

    def log_stats(self, conn: sqlite3.Connection) -> None:
        st = stats(conn)
        c = self.stream_client
        log.info(
            "edits=%d diffs_pending=%d enrich_pending=%d labelled=%d reverted=%d "
            "api_requests=%d diffs_fetched=%d http_429=%d stream_reconnects=%d last_edit_age=%ss",
            st["edits"], st["diffs_pending"], st["enrich_pending"], st["labelled"],
            st["reverted"], self.api.requests_made, self.diffs_fetched, self.api.rate_limited,
            c.disconnects if c else 0, st["last_edit_age"],
        )


def stats(conn: sqlite3.Connection) -> dict:
    one = lambda sql, *a: conn.execute(sql, a).fetchone()[0]  # noqa: E731
    last_ts = one("SELECT MAX(ts) FROM edits")
    return {
        "edits": one("SELECT COUNT(*) FROM edits"),
        "bot_edits": one("SELECT COUNT(*) FROM edits WHERE is_bot = 1"),
        "anon_edits": one("SELECT COUNT(*) FROM edits WHERE is_anon = 1"),
        "diffs_ok": one("SELECT COUNT(*) FROM diffs WHERE status = 'ok'"),
        "diffs_other": one("SELECT COUNT(*) FROM diffs WHERE status != 'ok'"),
        "diffs_pending": one(
            "SELECT COUNT(*) FROM edits e LEFT JOIN diffs d ON d.rev_id = e.rev_id "
            "WHERE d.rev_id IS NULL"
        ),
        "enrich_pending": one("SELECT COUNT(*) FROM edits WHERE enriched_at IS NULL"),
        "labelled": one("SELECT COUNT(*) FROM labels WHERE status = 'ok'"),
        "label_missing": one("SELECT COUNT(*) FROM labels WHERE status = 'missing'"),
        "reverted": one("SELECT COUNT(*) FROM labels WHERE reverted = 1"),
        "reverted_48h": one("SELECT COUNT(*) FROM labels WHERE reverted_48h = 1"),
        "last_edit_age": None if last_ts is None else int(time.time() - last_ts),
        "last_event_at": db.get_state(conn, "last_event_at"),
    }
