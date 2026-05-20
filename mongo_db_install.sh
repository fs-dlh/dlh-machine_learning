#!/bin/bash
set -euo pipefail

# ──────────────────────────────────────
# MongoDB 4.4.29 installation script for Ubuntu 22.04 (Jammy)
# ──────────────────────────────────────

# Script options
AUTO_YES=false
if [[ "${1:-}" == "-y" || "${1:-}" == "--yes" ]]; then
    AUTO_YES=true
fi

# Colour codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ──────────────────────────────────────
# 1. Root check
# ──────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
    log_error "This script must be run as root (use sudo)."
    exit 1
fi

# ──────────────────────────────────────
# 2. Install prerequisites early (curl, gnupg, iproute2, lsb-release)
# ──────────────────────────────────────
log_info "Installing prerequisites (curl, gnupg, iproute2, lsb-release)..."
apt-get update -qq
apt-get install -y curl gnupg iproute2 lsb-release

# ──────────────────────────────────────
# 3. Ubuntu version check
# ──────────────────────────────────────
if ! grep -q '^ID=ubuntu$' /etc/os-release 2>/dev/null; then
    log_error "This script is intended for Ubuntu only."
    exit 1
fi

UBUNTU_VERSION=$(lsb_release -rs)
if [[ "$UBUNTU_VERSION" != "22.04" ]]; then
    log_error "This script supports Ubuntu 22.04 only. Detected: $UBUNTU_VERSION"
    exit 1
fi
log_info "Ubuntu 22.04 detected."

# ──────────────────────────────────────
# 4. Architecture check (amd64 / arm64)
# ──────────────────────────────────────
ARCH=$(dpkg --print-architecture)
if [[ "$ARCH" != "amd64" && "$ARCH" != "arm64" ]]; then
    log_error "Unsupported architecture: $ARCH. Only amd64 and arm64 are supported by MongoDB 4.4."
    exit 1
fi
log_info "Architecture: $ARCH"

# ──────────────────────────────────────
# 5. Disk space check (>= 1 GB free on /)
# ──────────────────────────────────────
AVAIL_KB=$(df --output=avail / | tail -1 | tr -d ' ')
if [[ $AVAIL_KB -lt 1048576 ]]; then   # 1 GB = 1048576 KB
    log_error "Less than 1 GB of free disk space. At least 1 GB is required."
    exit 1
fi
log_info "Sufficient disk space available."

# ──────────────────────────────────────
# 6. Port 27017 check
# ──────────────────────────────────────
if ss -tulpn | grep -q ':27017 '; then
    log_warn "Port 27017 is already in use. If a previous MongoDB instance is running, it will be stopped during cleanup."
fi

# ──────────────────────────────────────
# 7. Handle existing MongoDB installations
# ──────────────────────────────────────
NEED_INSTALL=true
if dpkg -l mongodb-org &>/dev/null; then
    INSTALLED_VER=$(dpkg -l mongodb-org | grep '^ii' | awk '{print $3}')
    if [[ "$INSTALLED_VER" == "4.4.29" ]]; then
        log_info "MongoDB 4.4.29 is already installed."
        NEED_INSTALL=false
        # Try to start the service (handling both systemd and non-systemd)
        if [[ -d /run/systemd/system ]]; then
            systemctl enable mongod --now 2>/dev/null || true
        else
            service mongod start 2>/dev/null || mongod --fork --config /etc/mongod.conf 2>/dev/null || true
        fi
        if pgrep -x mongod > /dev/null; then
            log_info "mongod is running."
            exit 0
        else
            log_warn "mongod is not running. Attempting to start it..."
            if [[ -d /run/systemd/system ]]; then
                systemctl start mongod
            else
                service mongod start || mongod --fork --config /etc/mongod.conf
            fi
            if pgrep -x mongod > /dev/null; then
                log_info "mongod started successfully."
                exit 0
            else
                log_error "Failed to start mongod. Check logs: /var/log/mongodb/mongod.log"
                exit 1
            fi
        fi
    else
        log_warn "MongoDB version $INSTALLED_VER is installed instead of 4.4.29."
    fi
fi

if $NEED_INSTALL; then
    if dpkg -l mongodb-org &>/dev/null; then
        log_warn "An existing MongoDB installation will be REMOVED (including packages, configuration, and data directories)."
        if ! $AUTO_YES; then
            read -p "Do you want to continue? [y/N] " -r
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                log_info "Aborted by user."
                exit 0
            fi
        fi

        log_info "Stopping and removing existing MongoDB..."
        if [[ -d /run/systemd/system ]]; then
            systemctl stop mongod 2>/dev/null || true
            systemctl disable mongod 2>/dev/null || true
        else
            service mongod stop 2>/dev/null || true
            pkill -x mongod 2>/dev/null || true
        fi
        apt-get purge -y mongodb-org mongodb-org-server mongodb-org-shell mongodb-org-mongos mongodb-org-tools 2>/dev/null || true
        rm -rf /var/lib/mongodb /var/log/mongodb /etc/mongod.conf 2>/dev/null || true
        # Remove any old repository files
        rm -f /etc/apt/sources.list.d/mongodb*.list /usr/share/keyrings/mongodb*.gpg 2>/dev/null || true
        log_info "Old MongoDB completely removed."
    fi
fi

# Only continue if a fresh installation is needed
if ! $NEED_INSTALL; then
    exit 0
fi

# ──────────────────────────────────────
# 8. Import MongoDB 4.4 GPG key
# ──────────────────────────────────────
log_info "Importing MongoDB 4.4 GPG key..."
curl -fsSL https://www.mongodb.org/static/pgp/server-4.4.asc | \
    gpg --dearmor --yes -o /usr/share/keyrings/mongodb-server-4.4.gpg

# ──────────────────────────────────────
# 9. Add MongoDB 4.4 repository (using Ubuntu 20.04 "focal" packages)
# ──────────────────────────────────────
log_info "Adding MongoDB 4.4 repository..."
echo "deb [ arch=${ARCH} signed-by=/usr/share/keyrings/mongodb-server-4.4.gpg ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/4.4 multiverse" \
    > /etc/apt/sources.list.d/mongodb-org-4.4.list

# ──────────────────────────────────────
# 10. Install libssl1.1 (required by MongoDB 4.4 on Ubuntu 22.04)
# ──────────────────────────────────────
log_info "Installing libssl1.1 from Ubuntu 20.04 (focal-updates)..."
echo "deb http://archive.ubuntu.com/ubuntu focal-updates main" \
    > /etc/apt/sources.list.d/focal-updates-libssl.list

apt-get update -qq
apt-get install -y libssl1.1

rm -f /etc/apt/sources.list.d/focal-updates-libssl.list
apt-get update -qq

# ──────────────────────────────────────
# 11. Install MongoDB 4.4.29
# ──────────────────────────────────────
log_info "Installing MongoDB 4.4.29..."
apt-get install -y \
    mongodb-org=4.4.29 \
    mongodb-org-server=4.4.29 \
    mongodb-org-shell=4.4.29 \
    mongodb-org-mongos=4.4.29 \
    mongodb-org-tools=4.4.29

log_info "Holding MongoDB packages at version 4.4.29..."
apt-mark hold mongodb-org mongodb-org-server mongodb-org-shell mongodb-org-mongos mongodb-org-tools

# ──────────────────────────────────────
# 12. Start and enable MongoDB (systemd or SysV)
# ──────────────────────────────────────
log_info "Starting mongod..."
if [[ -d /run/systemd/system ]]; then
    # systemd is PID 1
    systemctl daemon-reload
    systemctl enable mongod --now
    sleep 2
    if systemctl is-active --quiet mongod; then
        log_info "MongoDB 4.4.29 is installed and running (systemd)."
    else
        log_error "mongod failed to start via systemd. Check: journalctl -xeu mongod"
        exit 1
    fi
else
    # Non-systemd environment (WSL1, Docker, SysV init)
    log_info "Systemd not detected – starting mongod manually..."
    # Ensure log and data directories exist
    mkdir -p /var/log/mongodb /var/lib/mongodb
    chown -R mongodb:mongodb /var/log/mongodb /var/lib/mongodb

    # Try the SysV init script first, fall back to direct fork
    if service mongod start 2>/dev/null; then
        log_info "mongod started via service command."
    else
        log_info "SysV init script unavailable, starting mongod with --fork..."
        # Start mongod using its config file
        mongod --config /etc/mongod.conf --fork
    fi

    sleep 2
    if pgrep -x mongod > /dev/null; then
        log_info "MongoDB 4.4.29 is installed and running (pid $(pgrep -x mongod))."
    else
        log_error "mongod failed to start. Check log: /var/log/mongodb/mongod.log"
        exit 1
    fi
fi

echo ""
echo "  Check status:  pgrep -x mongod"
echo "  Connect:       mongo"
echo ""
