"""Polite client for the MediaWiki Action API.

- Descriptive User-Agent with contact details (Wikimedia User-Agent policy).
- One request at a time with a minimum gap between requests (API:Etiquette).
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
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = settings.user_agent
        self.session.headers["Accept-Encoding"] = "gzip"
        if settings.api_token:
            self.session.headers["Authorization"] = f"Bearer {settings.api_token}"
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._last_start = -1e9
        self._window: deque[float] = deque()
        self.rate_limited = 0  # count of 429 responses, for the stats log
        self.max_attempts = max_attempts
        self.requests_made = 0

    def _wait_turn(self) -> None:
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
        with self._lock:  # never more than one request in flight
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
                    self._sleep(wait)
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
                        self._sleep(wait)
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
