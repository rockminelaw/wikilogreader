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

ARCH="$(dpkg --print-architecture)"      # amd64 on Intel/AMD hosts
echo "==> Finding the latest Debian 13 template for $ARCH"
pveam update >/dev/null
# Proxmox lists templates for several architectures; take only this host's.
TEMPLATE="$(pveam available --section system | awk '{print $2}' \
    | grep "^debian-13-standard_.*_${ARCH}\.tar\." | sort -V | tail -1)"
[ -n "$TEMPLATE" ] || { echo "No Debian 13 template for $ARCH found." >&2; exit 1; }
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
    --description "Wikipedia edit collector: $REPO_URL"

if ! pct start "$CTID" || [ "$(pct status "$CTID" | awk '{print $2}')" != running ]; then
    echo "Container $CTID was created but would not start." >&2
    echo "See why with:  pct start $CTID --debug" >&2
    echo "Remove it with: pct destroy $CTID --purge" >&2
    exit 1
fi

echo "==> Waiting for the network (up to 2 minutes)"
net_ok() { timeout 5 pct exec "$CTID" -- getent hosts deb.debian.org >/dev/null 2>&1; }
deadline=$((SECONDS + 120))
until net_ok; do
    if [ "$SECONDS" -ge "$deadline" ]; then
        echo "The container has no working network after 2 minutes." >&2
        echo "  Its address:  $(pct exec "$CTID" -- ip -4 -br addr show eth0 2>/dev/null)" >&2
        echo "  Its DNS:      $(pct exec "$CTID" -- grep nameserver /etc/resolv.conf 2>/dev/null | tr '\n' ' ')" >&2
        echo "No address: your network may not offer DHCP; set a static IP with" >&2
        echo "  pct set $CTID --net0 name=eth0,bridge=$BRIDGE,ip=192.168.x.y/24,gw=192.168.x.1" >&2
        echo "Address but no DNS:  pct set $CTID --nameserver 1.1.1.1 && pct reboot $CTID" >&2
        echo "Then finish with:" >&2
        echo "  pct exec $CTID -- bash -c 'apt-get update && apt-get install -y git && git clone $REPO_URL /opt/wikilogreader && bash /opt/wikilogreader/deploy/install.sh'" >&2
        exit 1
    fi
    sleep 2
done

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
