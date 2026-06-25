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

cleanup()
{
    if mountpoint -q "${MNT}" 2>/dev/null; then
        sudo umount "${MNT}"
    fi
    rmdir "${MNT}"
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

sudo mount -o loop,offset=$((2048 * 512)) "${IMAGE}" "${MNT}"
sudo install -D -m 0755 "${BIN}" "${MNT}/usr/local/bin/coralctl"
sync
echo "installed ${BIN} to ${IMAGE}:/usr/local/bin/coralctl"
