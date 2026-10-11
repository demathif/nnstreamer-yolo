#!/usr/bin/env bash
#
# setup.sh - install everything required to run the COCO128 -> nnstreamer ->
# YOLO pipeline on an Ubuntu x86_64 host.
#
# The script is idempotent: every step checks first and installs only what is
# missing, so re-running it is always safe. It uses apt (plus the nnstreamer
# PPA) and pip only - nothing is cloned or compiled.
#
# Run it as a normal user; sudo is used for apt and will prompt for a password.
#
# Usage:
#   scripts/setup.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_DIR="$PROJECT_ROOT/.venv"
REQ_RUNTIME="$SCRIPT_DIR/requirements.txt"

# nnstreamer is not in the official Ubuntu repositories; it comes from this PPA.
NNSTREAMER_PPA="ppa:nnstreamer/ppa"

# GStreamer plugins and tools used by the pipeline.
BASE_PACKAGES="
gstreamer1.0-tools
gstreamer1.0-plugins-base
gstreamer1.0-plugins-good
gstreamer1.0-plugins-bad
gstreamer1.0-plugins-ugly
gstreamer1.0-libav
python3-venv
python3-pip
curl
unzip
software-properties-common
"

# nnstreamer core plus its ONNX Runtime backend, both from the PPA.
NNSTREAMER_PACKAGES="nnstreamer nnstreamer-onnxruntime"

log() { printf '==> %s\n' "$*"; }

# install_pkgs PKG... -> install only the packages that are not installed yet.
install_pkgs() {
    local missing=()
    local pkg
    for pkg in "$@"; do
        if dpkg -s "$pkg" >/dev/null 2>&1; then
            printf 'already installed: %s\n' "$pkg"
        else
            missing+=("$pkg")
        fi
    done
    if [ "${#missing[@]}" -eq 0 ]; then
        return 0
    fi
    log "installing: ${missing[*]}"
    $SUDO apt-get install -y --no-install-recommends "${missing[@]}"
}

# --- platform checks -------------------------------------------------------
if [ "$(uname -m)" != "x86_64" ]; then
    echo "error: this project targets x86_64 (found $(uname -m))" >&2
    exit 1
fi
if ! command -v apt-get >/dev/null 2>&1; then
    echo "error: apt-get not found (Ubuntu/Debian required)" >&2
    exit 1
fi

# Prefix apt commands with sudo unless we are already root.
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

# --- base packages ---------------------------------------------------------
log "refreshing apt package lists"
$SUDO apt-get update

log "installing base packages"
# shellcheck disable=SC2086
install_pkgs $BASE_PACKAGES

# --- nnstreamer PPA and packages -------------------------------------------
if apt-cache policy nnstreamer 2>/dev/null | grep -qi nnstreamer \
    || ls /etc/apt/sources.list.d/*nnstreamer* >/dev/null 2>&1; then
    echo "nnstreamer PPA already configured"
else
    log "adding the nnstreamer PPA ($NNSTREAMER_PPA)"
    $SUDO add-apt-repository -y "$NNSTREAMER_PPA"
    $SUDO apt-get update
fi

log "installing nnstreamer packages"
# shellcheck disable=SC2086
install_pkgs $NNSTREAMER_PACKAGES

# --- python virtual environment -------------------------------------------
log "preparing the Python virtual environment"
if [ -x "$VENV_DIR/bin/python" ]; then
    echo "already exists: $VENV_DIR"
else
    python3 -m venv "$VENV_DIR"
fi
"$VENV_DIR/bin/pip" install --upgrade pip
if [ -f "$REQ_RUNTIME" ]; then
    "$VENV_DIR/bin/pip" install -r "$REQ_RUNTIME"
fi

# --- verify ----------------------------------------------------------------
log "verifying the nnstreamer elements"
missing=0
for e in tensor_filter tensor_converter tensor_decoder; do
    if gst-inspect-1.0 "$e" >/dev/null 2>&1; then
        printf 'present: %s\n' "$e"
    else
        printf 'MISSING: %s\n' "$e"
        missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo "error: nnstreamer is incomplete; open a new shell or run 'sudo ldconfig'" >&2
    exit 1
fi

log "setup complete - next: scripts/prepare.sh then scripts/coco128_yolo.sh"
