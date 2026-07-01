#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
SUPER_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
GEM5_REPO="${GEM5_REPO:-${SUPER_ROOT}/thirdparty/gem5}"

if ! git -C "${GEM5_REPO}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "error: gem5 submodule not found: ${GEM5_REPO}" >&2
    exit 1
fi

has_delta=0
for path in configs src ext include system util tests scripts; do
    if [ -e "${SCRIPT_DIR}/${path}" ]; then
        has_delta=1
        break
    fi
done

for file in SConstruct requirements.txt run_multicore.sh; do
    if [ -e "${SCRIPT_DIR}/${file}" ]; then
        has_delta=1
        break
    fi
done

if [ "${has_delta}" -ne 1 ]; then
    echo "error: no mirrored gem5 delta directories found under ${SCRIPT_DIR}" >&2
    exit 1
fi

copy_if_changed()
{
    source_path="$1"
    destination_path="$2"
    if [ ! -f "${destination_path}" ] ||
       [ "${source_path}" -nt "${destination_path}" ] ||
       ! cmp -s "${source_path}" "${destination_path}"; then
        mkdir -p "$(dirname "${destination_path}")"
        cp "${source_path}" "${destination_path}"
        # SCons must see overlay updates as newer than existing objects.
        touch "${destination_path}"
    fi
}

sync_tree()
{
    source_root="$1"
    destination_root="$2"
    (cd "${source_root}" && find . -type d -print) |
        while IFS= read -r relative_path; do
            mkdir -p "${destination_root}/${relative_path}"
        done
    (cd "${source_root}" && find . -type f -print) |
        while IFS= read -r relative_path; do
            copy_if_changed "${source_root}/${relative_path}" \
                "${destination_root}/${relative_path}"
        done
}

echo "[gem5-patchset] syncing mirrored directories into ${GEM5_REPO}"
for path in configs src ext include system util tests scripts; do
    if [ -e "${SCRIPT_DIR}/${path}" ]; then
        sync_tree "${SCRIPT_DIR}/${path}" "${GEM5_REPO}/${path}"
    fi
done

for file in SConstruct requirements.txt run_multicore.sh; do
    if [ -e "${SCRIPT_DIR}/${file}" ]; then
        copy_if_changed "${SCRIPT_DIR}/${file}" "${GEM5_REPO}/${file}"
    fi
done

echo "[gem5-patchset] done"
