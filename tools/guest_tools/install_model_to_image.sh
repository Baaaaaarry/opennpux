#!/bin/sh

set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 <disk-image> [model.npxm]" >&2
    exit 2
fi

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
IMAGE="$1"
MODEL="${2:-${ROOT_DIR}/build/models/heterogeneous-smoke.npxm}"
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

[ -f "${IMAGE}" ] || {
    echo "error: image not found: ${IMAGE}" >&2
    exit 1
}
[ -f "${MODEL}" ] || {
    echo "error: model not found: ${MODEL}" >&2
    exit 1
}

LOOP="$(sudo losetup --find --partscan --show "${IMAGE}")"
PART="${LOOP}p1"
[ -b "${PART}" ] || PART="${LOOP}"
sudo mount "${PART}" "${MNT}"
sudo install -D -m 0644 "${MODEL}" \
    "${MNT}/usr/local/share/opennpux/heterogeneous-smoke.npxm"
sync
echo "installed model into ${IMAGE}"
