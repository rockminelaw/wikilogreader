"""Polite client for the MediaWiki Action API.

- Descriptive User-Agent with contact details (Wikimedia User-Agent policy).
- At most `api_concurrency` requests in flight (default 3, the most the
  Wikimedia rate-limit page allows), a minimum gap between request starts, and
  a per-minute ceiling shared by every thread. A 429 pauses all threads.
- maxlag on every request; a lagged response waits and retries.
- 429 and 503 honour Retry-After; other server and network errors back off
  exponentially with jitter.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from collections import deque
from typing import Any

import requests

from .config import Settings

log = logging.getLogger(__name__)


class ApiError(Exception):
    """The API answered with an error code that retrying won't fix."""

    def __init__(self, code: str, info: str = ""):
        super().__init__(f"{code}: {info}")
        self.code = code
        self.info = info


class TransientError(Exception):
    """Gave up after repeated network or server errors; try again later."""


class ApiClient:
    def __init__(
        self,
        settings: Settings,
        session: requests.Session | None = None,
        sleep=time.sleep,
        clock=time.monotonic,
        max_attempts: int = 6,
    ):
        self.settings = settings
        # An injected session (tests) is shared; otherwise each thread gets its own.
        self._shared_session = session
        self._local = threading.local()
        if session is not None:
            self._prepare(session)
        self._sleep = sleep
        self._clock = clock
        self._slots = threading.BoundedSemaphore(max(1, settings.api_concurrency))
        self._rate_lock = threading.Lock()
        self._pause_until = -1e9
        self._last_start = -1e9
        self._window: deque[float] = deque()
        self.rate_limited = 0  # count of 429 responses, for the stats log
        self.max_attempts = max_attempts
        self.requests_made = 0

    def _prepare(self, session: requests.Session) -> requests.Session:
        session.headers["User-Agent"] = self.settings.user_agent
        session.headers["Accept-Encoding"] = "gzip"
        if self.settings.api_token:
            session.headers["Authorization"] = f"Bearer {self.settings.api_token}"
        return session

    @property
    def session(self) -> requests.Session:
        if self._shared_session is not None:
            return self._shared_session
        sess = getattr(self._local, "session", None)
        if sess is None:
            sess = self._local.session = self._prepare(requests.Session())
        return sess

    def _pause_all(self, seconds: float) -> None:
        with self._rate_lock:
            self._pause_until = max(self._pause_until, self._clock() + seconds)

    def _wait_turn(self) -> None:
        with self._rate_lock:
            self._wait_turn_locked()

    def _wait_turn_locked(self) -> None:
        now = self._clock()
        if self._pause_until > now:
            self._sleep(self._pause_until - now)
            now = self._clock()
        gap = self.settings.min_request_interval - (now - self._last_start)
        if gap > 0:
            self._sleep(gap)
            now = self._clock()
        # Sliding 60-second window.
        while self._window and now - self._window[0] >= 60:
            self._window.popleft()
        if len(self._window) >= self.settings.requests_per_minute:
            self._sleep(60 - (now - self._window[0]) + 0.05)
            now = self._clock()
            while self._window and now - self._window[0] >= 60:
                self._window.popleft()
        self._last_start = now
        self._window.append(now)

    def get(self, params: dict[str, Any]) -> dict:
        query = {
            "format": "json",
            "formatversion": "2",
            "maxlag": str(self.settings.maxlag),
            **params,
        }
        backoff = 2.0
        with self._slots:  # at most api_concurrency requests in flight
            for attempt in range(1, self.max_attempts + 1):
                self._wait_turn()
                self.requests_made += 1
                try:
                    resp = self.session.get(self.settings.api_url, params=query, timeout=(10, 60))
                except requests.RequestException as exc:
                    log.warning("API request failed (%s), attempt %d", exc.__class__.__name__, attempt)
                    self._sleep(_jitter(backoff))
                    backoff = min(backoff * 2, 300)
                    continue

                if resp.status_code in (429, 503):
                    if resp.status_code == 429:
                        self.rate_limited += 1
                    # Rate-limit page: without Retry-After, wait at least 5 s.
                    wait = _retry_after(resp, default=max(5.0, backoff))
                    log.warning("API returned %d, waiting %.0fs", resp.status_code, wait)
                    self._pause_all(wait)
                    backoff = min(backoff * 2, 300)
                    continue
                if resp.status_code >= 500:
                    log.warning("API returned %d, attempt %d", resp.status_code, attempt)
                    self._sleep(_jitter(backoff))
                    backoff = min(backoff * 2, 300)
                    continue
                if resp.status_code != 200:
                    raise ApiError(f"http{resp.status_code}", resp.text[:200])

                try:
                    data = resp.json()
                except ValueError:
                    log.warning("API returned non-JSON, attempt %d", attempt)
                    self._sleep(_jitter(backoff))
                    backoff = min(backoff * 2, 300)
                    continue

                err = data.get("error")
                if err:
                    code = err.get("code", "unknown")
                    if code == "maxlag":
                        wait = _retry_after(resp, default=5.0)
                        log.info("Replication lag high, waiting %.0fs", wait)
                        self._pause_all(wait)
                        continue
                    if code in ("ratelimited", "internal_api_error_DBQueryError"):
                        self._sleep(_jitter(backoff))
                        backoff = min(backoff * 2, 300)
                        continue
                    raise ApiError(code, err.get("info", ""))
                return data
        raise TransientError(f"gave up after {self.max_attempts} attempts")


def _jitter(seconds: float) -> float:
    return seconds * random.uniform(0.75, 1.25)


def _retry_after(resp: requests.Response, default: float) -> float:
    value = resp.headers.get("Retry-After")
    if value:
        try:
            return max(1.0, min(float(value), 600.0))
        except ValueError:
            pass
    return default
