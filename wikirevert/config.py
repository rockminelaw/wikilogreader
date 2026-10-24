"""Settings. Defaults live here; environment variables override them (see README)."""

from __future__ import annotations

import os
from dataclasses import dataclass, fields, replace

import requests

from . import __version__

CONTACT_EMAIL = "celarry1@gmail.com"
PROJECT_URL = "https://github.com/rockminelaw/wikilogreader"


@dataclass(frozen=True)
class Settings:
    db_path: str = "data/edits.sqlite"
    stream_url: str = "https://stream.wikimedia.org/v2/stream/recentchange"
    api_url: str = "https://en.wikipedia.org/w/api.php"
    wiki: str = "enwiki"
    namespace: int = 0

    # Etiquette. Every Action API request goes through one thread, one at a time,
    # with at least this many seconds between the starts of consecutive requests,
    # and no more than requests_per_minute in any 60-second window. Wikimedia's
    # 2026 limit for an unauthenticated client with a proper User-Agent is 200 a
    # minute; English Wikipedia needs about 100 a minute at normal edit rates.
    min_request_interval: float = 0.25
    requests_per_minute: int = 150
    maxlag: int = 5
    # Optional personal API token from api.wikimedia.org, sent as a Bearer token.
    # It raises the rate limit for authenticated clients. Never logged.
    api_token: str = ""

    # Labels are checked once an edit is this old.
    label_delay_hours: float = 48.0
    # User info and save-time tags are fetched once an edit is this old.
    enrich_delay_seconds: int = 60

    fetch_bot_diffs: bool = True
    max_text_chars: int = 50_000
    stats_interval_seconds: int = 300

    @property
    def user_agent(self) -> str:
        # Wikimedia User-Agent policy format:
        # <client name>/<version> (<contact information>) <library>/<version>
        return (
            f"WikiLogReader/{__version__} ({PROJECT_URL}; {CONTACT_EMAIL}) "
            f"python-requests/{requests.__version__}"
        )


# Environment variable for each setting, e.g. WRR_DB_PATH, WRR_LABEL_DELAY_HOURS.
def load(environ: dict | None = None) -> Settings:
    env = os.environ if environ is None else environ
    base = Settings()
    overrides = {}
    for f in fields(Settings):
        key = "WRR_" + f.name.upper()
        if key not in env:
            continue
        raw = env[key]
        current = getattr(base, f.name)
        if isinstance(current, bool):
            overrides[f.name] = raw.strip().lower() in {"1", "true", "yes"}
        elif isinstance(current, int):
            overrides[f.name] = int(raw)
        elif isinstance(current, float):
            overrides[f.name] = float(raw)
        else:
            overrides[f.name] = raw
    return replace(base, **overrides)
