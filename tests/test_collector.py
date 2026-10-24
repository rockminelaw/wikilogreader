"""Tests against responses recorded from the live services on 24 October 2026."""

from __future__ import annotations

import json
import sqlite3
import time
import zlib
from pathlib import Path

import pytest
import requests

from wikirevert import config, db, diffs, enrich, labels
from wikirevert.collector import Collector
from wikirevert.http import ApiClient, ApiError, TransientError
from wikirevert.sse import SSEParser, parse_all
from wikirevert.stream import parse_edit
from wikirevert.users import is_anonymous, is_ip, is_temp_account

FIX = Path(__file__).parent / "fixtures"


def load(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


# ---------- users ----------


def test_anonymous_detection():
    assert is_temp_account("~2026-56098-08")
    assert not is_temp_account("ROLAND CES")
    assert not is_temp_account("~Tilde user")
    assert is_ip("192.0.2.1") and is_ip("2001:db8::1")
    assert is_anonymous("~2025-12345-67") and is_anonymous("203.0.113.9")
    assert not is_anonymous("Jimbo Wales") and not is_anonymous(None)


# ---------- SSE parsing ----------


def test_sse_parser_handles_split_chunks_and_line_endings():
    raw = (
        b": comment\r\n"
        b"event: message\r\nid: [1]\r\ndata: {\"a\":\r\ndata: 1}\r\n\r\n"
        b"id: [2]\ndata: x\n\n"
        b"data: no id\r\r"
        b"id: [3]\ndata: z\n\n"
    )
    # Feed one byte at a time to make sure chunk boundaries don't matter.
    events = parse_all(raw[i : i + 1] for i in range(len(raw)))
    assert [e.data for e in events] == ['{"a":\n1}', "x", "no id", "z"]
    assert [e.id for e in events] == ["[1]", "[2]", "[2]", "[3]"]  # id persists


def test_recorded_stream_filters_to_enwiki_main_namespace_edits():
    events = parse_all([(FIX / "stream_sample.sse").read_bytes()])
    assert len(events) >= 9
    assert all(e.id and e.id.startswith("[") for e in events)
    edits = [x for x in (parse_edit(e.data) for e in events) if x]
    assert len(edits) == 8  # the canary copy of an edit is dropped
    first = edits[0]
    assert first["rev_id"] == 1381682188 and first["old_rev_id"] == 1381682133
    assert first["title"] == "1996 FBC Melgar season"
    assert first["size_delta"] == 3975 - 3739
    assert first["is_anon"] == 0 and first["is_bot"] == 0
    assert all(e["namespace"] == 0 for e in edits)


def test_parse_edit_marks_temporary_accounts():
    ev = {
        "meta": {"domain": "en.wikipedia.org", "dt": "2026-10-24T00:00:00Z"},
        "wiki": "enwiki", "type": "edit", "namespace": 0, "title": "X", "id": 1,
        "user": "~2026-56098-08", "bot": False, "minor": False, "timestamp": 1792800000,
        "length": {"old": 10, "new": 5}, "revision": {"old": 1, "new": 2},
    }
    row = parse_edit(json.dumps(ev))
    assert row["is_anon"] == 1 and row["is_temp"] == 1 and row["size_delta"] == -5
    for change in ({"type": "new"}, {"namespace": 1}, {"wiki": "dewiki"},
                   {"meta": {"domain": "canary"}}):
        assert parse_edit(json.dumps({**ev, **change})) is None
    assert parse_edit("not json") is None


# ---------- diffs ----------


def test_extract_changed_words_from_reverted_edit():
    body = load("compare_reverted.json")["compare"]["body"]
    t = diffs.extract(body)
    assert t.lines_changed >= 1
    # The edit removed "in San Francisco" from the birthplace sentence.
    assert any("in San Francisco" in s for s in t.removed_text)
    assert any(s.startswith("Guilfoyle was born") for s in t.removed_lines)
    assert t.moved_paras == 0


def test_extract_added_lines():
    body = load("compare_plain.json")["compare"]["body"]
    t = diffs.extract(body)
    assert t.lines_added == 6 and t.lines_removed == 0 and t.lines_changed == 0
    assert "[[Category:FBC Melgar seasons]]" in t.added_text
    assert t.removed_text == []


def test_extract_skips_moved_paragraphs():
    body = (
        '<tr><td class="diff-marker"></td><td class="diff-deletedline diff-side-deleted">'
        '<div><a class="mw-diff-movedpara-left" href="#m1">&#x26AB;</a>Moved text</div></td>'
        '<td colspan="2" class="diff-empty diff-side-added"></td></tr>'
        '<tr><td colspan="2" class="diff-empty diff-side-deleted"></td><td class="diff-marker"></td>'
        '<td class="diff-addedline diff-side-added"><div>New line</div></td></tr>'
    )
    t = diffs.extract(body)
    assert t.moved_paras == 1 and t.added_text == ["New line"] and t.removed_text == []


def test_diff_row_round_trips_html():
    data = load("compare_plain.json")
    result = diffs.DiffResult(1, "t", data["compare"]["body"], diffs.extract(data["compare"]["body"]))
    row = diffs.to_row(5, result, 0.0, max_chars=40)
    assert len(row["added_text"]) == 40  # truncated
    assert zlib.decompress(row["html_z"]).decode() == data["compare"]["body"]


# ---------- enrichment ----------


def test_enrich_parses_users_and_drops_outcome_tag():
    tags, _ = enrich.parse_response(load("revisions_batch.json"))
    _, users = enrich.parse_response(load("users_batch.json"))
    assert tags[1381246404] == ["visualeditor"]  # mw-reverted removed
    assert "mobile edit" in tags[1381682188]
    temp = users["~2026-56098-08"]
    assert temp["editcount"] == 6 and temp["groups"] == "temp"
    assert temp["registration"] == enrich.parse_ts("2026-10-20T23:55:12Z")
    assert "NoSuchUserXyz123" not in users
    assert "autoconfirmed" in users["ROLAND CES"]["groups"]


def test_enrich_params_skip_ip_usernames():
    rows = [{"rev_id": 1, "user": "203.0.113.9"}, {"rev_id": 2, "user": "Alice"}]
    p = enrich.build_params(rows)
    assert p["ususers"] == "Alice" and p["revids"] == "1|2"


# ---------- labels ----------


def test_find_revert_identifies_rollback():
    hist = load("history_after.json")
    rv = labels.find_revert(hist, 1381246404, 1381246160)
    assert rv["revert_rev_id"] == 1381246584
    assert rv["revert_method"] == "mw-rollback"
    assert rv["revert_ts"] - enrich.parse_ts("2026-10-20T23:59:44Z") == 129
    assert rv["reverter"] == "Discospinster"


def test_find_revert_by_sha1():
    hist = {"query": {"pages": [{"revisions": [
        {"revid": 10, "sha1": "a", "timestamp": "2026-01-01T00:00:00Z", "tags": []},
        {"revid": 11, "sha1": "b", "timestamp": "2026-01-01T00:01:00Z", "tags": []},
        {"revid": 12, "sha1": "c", "timestamp": "2026-01-01T00:02:00Z", "tags": []},
        {"revid": 13, "sha1": "a", "timestamp": "2026-01-01T00:03:00Z", "tags": [], "user": "X"},
    ]}]}}
    rv = labels.find_revert(hist, 11, 10)
    assert rv["revert_rev_id"] == 13 and rv["revert_method"] == "sha1"


def test_parse_tags_reports_missing_revisions():
    found, missing = labels.parse_tags(load("revisions_missing.json"))
    assert missing == {999999999999}
    assert "mw-reverted" in found[1381246404]["tags"]


# ---------- end to end with a fake API ----------


class FakeResponse:
    def __init__(self, payload, status=200, headers=None):
        self.payload, self.status_code, self.headers = payload, status, headers or {}
        self.text = json.dumps(payload) if isinstance(payload, dict) else str(payload)

    def json(self):
        if not isinstance(self.payload, dict):
            raise ValueError("not json")
        return self.payload


class FakeSession(requests.Session):
    def __init__(self, handler):
        super().__init__()
        self.handler, self.calls = handler, []

    def get(self, url, params=None, **kw):
        self.calls.append(dict(params or {}))
        return self.handler(params or {})


def make_api(handler, tmp_path=None, **overrides):
    s = config.load({})
    s = config.Settings(**{**s.__dict__, "min_request_interval": 0, **overrides})
    sleeps: list[float] = []
    api = ApiClient(s, session=FakeSession(handler), sleep=sleeps.append)
    return api, sleeps


def test_user_agent_follows_policy():
    ua = config.load({}).user_agent
    assert ua.startswith("WikiLogReader/") and "celarry1@gmail.com" in ua
    assert "python-requests/" in ua


def test_api_retries_429_with_retry_after_then_succeeds():
    responses = iter([
        FakeResponse("too many", 429, {"Retry-After": "7"}),
        FakeResponse({"error": {"code": "maxlag", "info": "lag"}}, 200, {"Retry-After": "3"}),
        FakeResponse({"ok": True}),
    ])
    api, sleeps = make_api(lambda p: next(responses))
    assert api.get({"action": "query"}) == {"ok": True}
    assert 7 in sleeps and 3 in sleeps and api.rate_limited == 1
    sent = api.session.calls[0]
    assert sent["maxlag"] == "5" and sent["formatversion"] == "2"


def test_api_gives_up_after_repeated_server_errors():
    api, sleeps = make_api(lambda p: FakeResponse("oops", 502))
    with pytest.raises(TransientError):
        api.get({"action": "query"})
    assert len(sleeps) == api.max_attempts
    assert sleeps[1] > sleeps[0]  # exponential backoff


def test_api_raises_on_real_errors():
    api, _ = make_api(lambda p: FakeResponse(load("compare_error.json")))
    with pytest.raises(ApiError) as exc:
        api.get({"action": "compare"})
    assert exc.value.code == "nosuchrevid"


def test_requests_per_minute_budget():
    clock = [0.0]
    sleeps: list[float] = []

    def sleep(sec):
        sleeps.append(sec)
        clock[0] += sec

    s = config.Settings(min_request_interval=0, requests_per_minute=3)
    api = ApiClient(s, session=FakeSession(lambda p: FakeResponse({"ok": 1})), sleep=sleep,
                    clock=lambda: clock[0])
    for _ in range(4):
        api.get({})
    assert len(sleeps) == 1 and sleeps[0] == pytest.approx(60.05)


def test_pipeline_end_to_end(tmp_path):
    """Store two edits, fetch diffs, enrich, then label after the delay."""
    s = config.Settings(db_path=str(tmp_path / "t.sqlite"), min_request_interval=0)
    conn = db.connect(s.db_path)
    db.init(conn)
    old = int(time.time()) - 3 * 86400
    base = dict(namespace=0, is_bot=0, is_anon=1, is_temp=1, comment="", minor=0,
                patrolled=None, size_old=57918, size_new=57913, size_delta=-5,
                meta_dt=None, received_at=old, rc_id=1)
    edits = [
        dict(base, rev_id=1381246404, old_rev_id=1381246160, title="Kimberly Guilfoyle",
             user="~2026-56098-08", ts=enrich.parse_ts("2026-10-20T23:59:44Z")),
        dict(base, rev_id=1381682188, old_rev_id=1381682133, title="1996 FBC Melgar season",
             user="ROLAND CES", is_anon=0, is_temp=0, ts=old),
    ]
    assert db.insert_edits(conn, edits, '[{"offset":1}]') == 2
    assert db.insert_edits(conn, edits, '[{"offset":2}]') == 0  # duplicates ignored on resume
    assert db.get_state(conn, "last_event_id") == '[{"offset":2}]'

    def handler(p):
        if p.get("action") == "compare":
            return FakeResponse(load("compare_reverted.json") if p["torev"] == "1381246404"
                                else load("compare_plain.json"))
        if p.get("list") == "users":
            merged = load("revisions_batch.json")
            merged["query"]["users"] = load("users_batch.json")["query"]["users"]
            return FakeResponse(merged)
        if "pageids" in p:
            return FakeResponse(load("history_after.json"))
        return FakeResponse(load("revisions_batch.json"))

    c = Collector(s)
    c.api = ApiClient(s, session=FakeSession(handler), sleep=lambda x: None)
    assert c.diff_step(conn) == 1 and c.diff_step(conn) == 1 and c.diff_step(conn) == 0
    assert enrich.run_batch(conn, c.api, 60) == 2
    assert labels.run_batch(conn, c.api, 48) == 2

    rows = {r["rev_id"]: dict(r) for r in conn.execute(
        "SELECT * FROM edits JOIN labels USING(rev_id) JOIN diffs USING(rev_id)")}
    g = rows[1381246404]
    assert g["page_id"] == 1385937 and g["status"] == "ok"
    assert g["reverted"] == 1 and g["revert_seconds"] == 129 and g["reverted_48h"] == 1
    assert g["revert_method"] == "mw-rollback" and g["reverter_is_self"] == 0
    assert g["user_editcount"] == 6 and g["tags_at_save"] == "visualeditor"
    m = rows[1381682188]
    assert m["reverted"] == 0 and m["reverted_48h"] == 0
    assert "Category:FBC Melgar seasons" in m["added_text"]
