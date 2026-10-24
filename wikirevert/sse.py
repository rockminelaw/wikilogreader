"""Server-sent events client for Wikimedia EventStreams.

EventStreams closes connections from time to time, so the client reconnects
and sends the last event id it stored (Last-Event-ID header). The id encodes
Kafka positions, so the server resumes where we left off as long as those
messages are still retained (about a week).
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator

import requests

log = logging.getLogger(__name__)


@dataclass
class Event:
    data: str
    id: str | None = None
    event: str = "message"


@dataclass
class SSEParser:
    """Incremental parser following the WHATWG event-stream format."""

    last_id: str | None = None
    _buffer: bytes = b""
    _data: list[str] = field(default_factory=list)
    _event: str = ""
    _pending_id: str | None = None

    def feed(self, chunk: bytes) -> Iterator[Event]:
        self._buffer += chunk
        while True:
            # Lines end in \n, \r\n or \r. A trailing \r might be the start of
            # \r\n, so keep it until more data arrives.
            idx_n = self._buffer.find(b"\n")
            idx_r = self._buffer.find(b"\r")
            if idx_n == -1 and (idx_r == -1 or idx_r == len(self._buffer) - 1):
                return
            if idx_r != -1 and (idx_n == -1 or idx_r < idx_n):
                if idx_r + 1 < len(self._buffer) and self._buffer[idx_r + 1 : idx_r + 2] == b"\n":
                    line, self._buffer = self._buffer[:idx_r], self._buffer[idx_r + 2 :]
                else:
                    line, self._buffer = self._buffer[:idx_r], self._buffer[idx_r + 1 :]
            else:
                line, self._buffer = self._buffer[:idx_n], self._buffer[idx_n + 1 :]
            event = self._line(line.decode("utf-8", errors="replace"))
            if event is not None:
                yield event

    def _line(self, line: str) -> Event | None:
        if line == "":
            return self._dispatch()
        if line.startswith(":"):
            return None  # comment / keep-alive
        name, sep, value = line.partition(":")
        if sep and value.startswith(" "):
            value = value[1:]
        if name == "data":
            self._data.append(value)
        elif name == "event":
            self._event = value
        elif name == "id":
            if "\0" not in value:
                self._pending_id = value
        # "retry" is ignored: reconnect timing is handled by our own backoff.
        return None

    def _dispatch(self) -> Event | None:
        if self._pending_id is not None:
            self.last_id = self._pending_id
            self._pending_id = None
        if not self._data:
            self._event = ""
            return None
        ev = Event(data="\n".join(self._data), id=self.last_id, event=self._event or "message")
        self._data = []
        self._event = ""
        return ev


class StreamClient:
    """Yields events forever, reconnecting with Last-Event-ID after any drop."""

    def __init__(
        self,
        url: str,
        user_agent: str,
        last_event_id: str | None = None,
        session: requests.Session | None = None,
        stop: threading.Event | None = None,
        read_timeout: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.url = url
        self.last_event_id = last_event_id
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = user_agent
        self.stop = stop or threading.Event()
        self.read_timeout = read_timeout
        self._sleep = sleep
        self.connects = 0
        self.disconnects = 0

    def events(self) -> Iterator[Event]:
        backoff = 1.0
        while not self.stop.is_set():
            headers = {"Accept": "text/event-stream", "Cache-Control": "no-cache"}
            if self.last_event_id:
                headers["Last-Event-ID"] = self.last_event_id
            try:
                with self.session.get(
                    self.url, headers=headers, stream=True, timeout=(15, self.read_timeout)
                ) as resp:
                    if resp.status_code == 429 or resp.status_code >= 500:
                        wait = _retry_after(resp, backoff)
                        log.warning("Stream returned %d, retrying in %.0fs", resp.status_code, wait)
                        self._pause(wait)
                        backoff = min(backoff * 2, 300)
                        continue
                    if resp.status_code != 200:
                        log.error("Stream returned %d: %s", resp.status_code, resp.text[:200])
                        self._pause(_jitter(backoff))
                        backoff = min(backoff * 2, 300)
                        continue
                    self.connects += 1
                    log.info("Stream connected (resume=%s)", bool(self.last_event_id))
                    parser = SSEParser(last_id=self.last_event_id)
                    got_any = False
                    for chunk in resp.iter_content(chunk_size=None):
                        if self.stop.is_set():
                            return
                        for ev in parser.feed(chunk):
                            got_any = True
                            if ev.id:
                                self.last_event_id = ev.id
                            yield ev
                        if got_any:
                            backoff = 1.0
                    log.info("Stream closed by server")
            except requests.RequestException as exc:
                log.warning("Stream dropped: %s", exc.__class__.__name__)
            self.disconnects += 1
            self._pause(_jitter(backoff))
            backoff = min(backoff * 2, 300)

    def _pause(self, seconds: float) -> None:
        if self._sleep is time.sleep:
            self.stop.wait(seconds)  # wakes early if asked to stop
        else:
            self._sleep(seconds)


def _jitter(seconds: float) -> float:
    return seconds * random.uniform(0.75, 1.25)


def _retry_after(resp: requests.Response, default: float) -> float:
    try:
        return max(1.0, min(float(resp.headers.get("Retry-After", "")), 600.0))
    except ValueError:
        return default


def parse_all(chunks: Iterable[bytes]) -> list[Event]:
    """Convenience for tests: parse a whole recorded stream."""
    parser = SSEParser()
    out: list[Event] = []
    for c in chunks:
        out.extend(parser.feed(c))
    return out
