#!/bin/sh
#
# install_model_to_image.sh — Install an .npxm model file into a gem5 disk image.
#
# Mounts the Ubuntu ARM64 disk image via losetup and copies the model container
# to /usr/local/share/opennpux/ inside the guest filesystem.
#
# The .npxm format is the OpenNPUX model container: a binary file with a
# 32-byte header, one or more operator command descriptors, and tensor data.
# See create_sample_model.py for the format definition.
#
# Installed path inside the image:
#   /usr/local/share/opennpux/heterogeneous-smoke.npxm
#
# Usage:
#   sudo $0 <disk-image> [model.npxm]
#
# The optional [model.npxm] argument overrides the default
# build/models/heterogeneous-smoke.npxm.
#
# @guest-tools-spec  v1  2025-07-29

set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 <disk-image> [model.npxm]" >&2
    exit 2
fi

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
IMAGE="$1"
MODEL="${2:-${ROOT_DIR}/build/models/heterogeneous-smoke.npxm}"

# ---------------------------------------------------------------------------
# Mount via losetup --partscan with cleanup trap.
# ---------------------------------------------------------------------------
MNT="$(mktemp -d)"
LOOP=

cleanup() {
    set +e
    if mountpoint -q "${MNT}"; then
        sudo umount "${MNT}"
    fi
    [ -z "${LOOP}" ] || sudo losetup -d "${LOOP}"
    rmdir "${MNT}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------------------
# Preflight: verify inputs exist.
# ---------------------------------------------------------------------------
[ -f "${IMAGE}" ] || {
    echo "error: image not found: ${IMAGE}" >&2
    exit 1
}
[ -f "${MODEL}" ] || {
    echo "error: model not found: ${MODEL}" >&2
    echo "hint: run tools/models/create_sample_model.py first" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Set up loop device, mount, and install.
# ---------------------------------------------------------------------------
LOOP="$(sudo losetup --find --partscan --show "${IMAGE}")"
PART="${LOOP}p1"
[ -b "${PART}" ] || PART="${LOOP}"

if ! sudo mount "${PART}" "${MNT}"; then
    echo "error: unable to mount image filesystem from ${PART}" >&2
    lsblk -f "${LOOP}" >&2 || true
    exit 1
fi

sudo install -D -m 0644 "${MODEL}" \
    "${MNT}/usr/local/share/opennpux/heterogeneous-smoke.npxm"
sync
echo "installed model into ${IMAGE}"
