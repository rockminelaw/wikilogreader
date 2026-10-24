#!/usr/bin/env bash
# Create a small Debian 13 container on a Proxmox VE host and install the
# collector in it. Run on the Proxmox host as root (Shell in the web UI):
#
#   bash -c "$(curl -fsSL https://raw.githubusercontent.com/rockminelaw/wikilogreader/main/deploy/proxmox-create-ct.sh)"
#
# Defaults can be changed with environment variables, for example:
#   STORAGE=local-zfs DISK_GB=100 bash proxmox-create-ct.sh
set -euo pipefail

CTID="${CTID:-$(pvesh get /cluster/nextid)}"
CT_NAME="${CT_NAME:-wikilogreader}"
STORAGE="${STORAGE:-local-lvm}"          # where the container's disk goes
TEMPLATE_STORAGE="${TEMPLATE_STORAGE:-local}"
DISK_GB="${DISK_GB:-64}"                 # about 1.2 GB a day of collected data
MEMORY_MB="${MEMORY_MB:-1024}"
CORES="${CORES:-2}"
BRIDGE="${BRIDGE:-vmbr0}"
REPO_URL="${REPO_URL:-https://github.com/rockminelaw/wikilogreader}"

command -v pct >/dev/null || { echo "Run this on the Proxmox host." >&2; exit 1; }

if ! pvesm status --content rootdir | awk 'NR>1 {print $1}' | grep -qx "$STORAGE"; then
    echo "Storage '$STORAGE' can't hold containers. Pick one of these with STORAGE=...:" >&2
    pvesm status --content rootdir >&2
    exit 1
fi

echo "==> Finding the latest Debian 13 template"
pveam update >/dev/null
TEMPLATE="$(pveam available --section system | awk '{print $2}' | grep '^debian-13-standard' | sort -V | tail -1)"
[ -n "$TEMPLATE" ] || { echo "No Debian 13 template found." >&2; exit 1; }
if ! pveam list "$TEMPLATE_STORAGE" | grep -q "$TEMPLATE"; then
    pveam download "$TEMPLATE_STORAGE" "$TEMPLATE"
fi

echo "==> Creating container $CTID ($CT_NAME) from $TEMPLATE"
pct create "$CTID" "$TEMPLATE_STORAGE:vztmpl/$TEMPLATE" \
    --hostname "$CT_NAME" \
    --unprivileged 1 \
    --features nesting=1 \
    --cores "$CORES" \
    --memory "$MEMORY_MB" \
    --swap 512 \
    --rootfs "$STORAGE:$DISK_GB" \
    --net0 "name=eth0,bridge=$BRIDGE,ip=dhcp" \
    --onboot 1 \
    --start 1 \
    --description "Wikipedia edit collector: $REPO_URL"

echo "==> Waiting for the network"
for _ in $(seq 1 60); do
    if pct exec "$CTID" -- getent hosts deb.debian.org >/dev/null 2>&1; then
        break
    fi
    sleep 2
done
pct exec "$CTID" -- getent hosts deb.debian.org >/dev/null || {
    echo "The container has no network. Check the bridge ($BRIDGE) and DHCP." >&2
    exit 1
}

echo "==> Installing the collector inside the container"
pct exec "$CTID" -- bash -c "
    set -e
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq
    apt-get install -y -qq git ca-certificates >/dev/null
    [ -d /opt/wikilogreader/.git ] || git clone '$REPO_URL' /opt/wikilogreader
    REPO_URL='$REPO_URL' bash /opt/wikilogreader/deploy/install.sh
"

echo
echo "Done. Container $CTID is running the collector and starts with the host."
echo "  Log:     pct exec $CTID -- journalctl -u wikilogreader -f"
echo "  Counts:  pct exec $CTID -- wikilogreader stats"
