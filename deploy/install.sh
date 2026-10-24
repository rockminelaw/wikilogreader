#!/usr/bin/env bash
# Install or update the collector as a systemd service on Debian or Ubuntu.
# Run as root inside the container (or any Debian-based machine):
#   bash /opt/wikilogreader/deploy/install.sh
# Running it again pulls the latest code and restarts the service; the
# database in /var/lib/wikilogreader is left alone.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/rockminelaw/wikilogreader}"
APP_DIR=/opt/wikilogreader
DATA_DIR=/var/lib/wikilogreader
SVC_USER=wikilogreader

if [ "$(id -u)" -ne 0 ]; then
    echo "Run this as root." >&2
    exit 1
fi

echo "==> Installing system packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq python3 python3-venv git sqlite3 ca-certificates >/dev/null

echo "==> Creating the $SVC_USER user"
if ! id "$SVC_USER" >/dev/null 2>&1; then
    useradd --system --home-dir "$DATA_DIR" --shell /usr/sbin/nologin "$SVC_USER"
fi
install -d -o "$SVC_USER" -g "$SVC_USER" -m 750 "$DATA_DIR"

echo "==> Fetching the code into $APP_DIR"
if [ -d "$APP_DIR/.git" ]; then
    git -C "$APP_DIR" pull --ff-only
else
    git clone "$REPO_URL" "$APP_DIR"
fi

echo "==> Installing Python dependencies"
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install -q --upgrade pip
"$APP_DIR/.venv/bin/pip" install -q "$APP_DIR"

echo "==> Installing the service"
install -m 755 "$APP_DIR/deploy/wikilogreader" /usr/local/bin/wikilogreader
install -m 644 "$APP_DIR/deploy/wikilogreader.service" \
    "$APP_DIR/deploy/wikilogreader-health.service" \
    "$APP_DIR/deploy/wikilogreader-health.timer" /etc/systemd/system/
systemctl daemon-reload
systemctl enable -q wikilogreader.service wikilogreader-health.timer
systemctl restart wikilogreader.service
systemctl start wikilogreader-health.timer

sleep 5
if systemctl is-active --quiet wikilogreader.service; then
    echo "==> Running. Follow it with:  journalctl -u wikilogreader -f"
    echo "    Counts:                    wikilogreader stats"
else
    echo "==> The service did not start. Recent log:" >&2
    journalctl -u wikilogreader -n 30 --no-pager >&2
    exit 1
fi
