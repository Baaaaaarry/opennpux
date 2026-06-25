#!/bin/sh

set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 <disk-image> [coralctl-binary]" >&2
    exit 2
fi

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
IMAGE="$1"
BIN="${2:-${ROOT_DIR}/build/guest-tools/coralctl-aarch64}"
MNT="$(mktemp -d)"
LOOP=

cleanup()
{
    set +e
    if mountpoint -q "${MNT}"; then
        sudo umount "${MNT}"
    fi
    if [ -n "${LOOP}" ]; then
        sudo losetup -d "${LOOP}"
    fi
    rmdir "${MNT}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [ ! -f "${IMAGE}" ]; then
    echo "error: disk image not found: ${IMAGE}" >&2
    exit 1
fi
if [ ! -x "${BIN}" ]; then
    echo "error: coralctl binary not found or not executable: ${BIN}" >&2
    echo "hint: run tools/guest_tools/build_coralctl.sh first" >&2
    exit 1
fi

LOOP="$(sudo losetup --find --partscan --show "${IMAGE}")"
PART="${LOOP}p1"
if [ ! -b "${PART}" ]; then
    PART="${LOOP}"
fi

if ! sudo mount "${PART}" "${MNT}"; then
    echo "error: unable to mount image filesystem from ${PART}" >&2
    echo "detected loop layout:" >&2
    lsblk -f "${LOOP}" >&2 || true
    sudo blkid "${LOOP}" "${LOOP}"p* >&2 || true
    exit 1
fi

sudo install -D -m 0755 "${BIN}" "${MNT}/usr/local/bin/coralctl"
sync
echo "installed ${BIN} to ${IMAGE}:/usr/local/bin/coralctl"
