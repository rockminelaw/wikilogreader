"""Telling logged-out editors apart from accounts.

Since 4 November 2025 English Wikipedia gives logged-out editors a temporary
account named like "~2025-12345-67" instead of showing their IP address. Edits
from before then (and a few edge cases) can still carry an IP address as the
username, so both forms count as anonymous.
"""

from __future__ import annotations

import ipaddress
import re

TEMP_ACCOUNT_RE = re.compile(r"^~\d{4}-\d+-\d+$")


def is_temp_account(user: str | None) -> bool:
    return bool(user) and TEMP_ACCOUNT_RE.match(user) is not None


def is_ip(user: str | None) -> bool:
    if not user:
        return False
    try:
        ipaddress.ip_address(user)
        return True
    except ValueError:
        return False


def is_anonymous(user: str | None) -> bool:
    return is_temp_account(user) or is_ip(user)
