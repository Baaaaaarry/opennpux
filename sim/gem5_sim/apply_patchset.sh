#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
SUPER_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
GEM5_REPO="${GEM5_REPO:-${SUPER_ROOT}/thirdparty/gem5}"

patch_file="${SCRIPT_DIR}/patches/0001-coral-stagea-integration.patch"
overlay_dir="${SCRIPT_DIR}/overlay"

if [ ! -d "${GEM5_REPO}/.git" ]; then
    echo "error: gem5 submodule not found: ${GEM5_REPO}" >&2
    exit 1
fi

echo "[gem5-patchset] applying ${patch_file}"
git -C "${GEM5_REPO}" apply "${patch_file}"

echo "[gem5-patchset] copying overlay files from ${overlay_dir}"
cp -R "${overlay_dir}/." "${GEM5_REPO}/"

echo "[gem5-patchset] done"
