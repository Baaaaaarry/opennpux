#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
SUPER_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
CORAL_REPO="${CORAL_REPO:-${SUPER_ROOT}/thirdparty/coralnpu}"
DELETE_LIST="${SCRIPT_DIR}/overlay_delete.txt"
RESTORE_LIST="${SCRIPT_DIR}/overlay_restore.txt"

if ! git -C "${CORAL_REPO}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "error: coralnpu submodule not found: ${CORAL_REPO}" >&2
    exit 1
fi

has_delta=0
for path in doc examples hdl hw_sim internal rules sw tests toolchain third_party util WORKSPACE BUILD.bazel MODULE.bazel; do
    if [ -e "${SCRIPT_DIR}/${path}" ]; then
        has_delta=1
        break
    fi
done

if [ "${has_delta}" -ne 1 ]; then
    echo "error: no mirrored coralnpu delta paths found under ${SCRIPT_DIR}" >&2
    exit 1
fi

copy_if_changed()
{
    source_path="$1"
    destination_path="$2"
    if [ ! -f "${destination_path}" ] ||
       ! cmp -s "${source_path}" "${destination_path}"; then
        mkdir -p "$(dirname "${destination_path}")"
        cp "${source_path}" "${destination_path}"
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

echo "[coralnpu-patchset] syncing mirrored paths into ${CORAL_REPO}"
if [ -f "${RESTORE_LIST}" ]; then
    while IFS= read -r path; do
        case "${path}" in
            ""|\#*) continue ;;
            /*|../*|*/../*)
                echo "error: unsafe overlay restore path: ${path}" >&2
                exit 1
                ;;
        esac
        git -C "${CORAL_REPO}" checkout HEAD -- "${path}"
    done < "${RESTORE_LIST}"
fi

if [ -f "${DELETE_LIST}" ]; then
    while IFS= read -r path; do
        case "${path}" in
            ""|\#*) continue ;;
            /*|../*|*/../*)
                echo "error: unsafe overlay delete path: ${path}" >&2
                exit 1
                ;;
        esac
        rm -rf "${CORAL_REPO}/${path}"
    done < "${DELETE_LIST}"
fi

for path in doc examples hdl hw_sim internal rules sw tests toolchain third_party util WORKSPACE BUILD.bazel MODULE.bazel; do
    if [ -e "${SCRIPT_DIR}/${path}" ]; then
        if [ -d "${SCRIPT_DIR}/${path}" ]; then
            sync_tree "${SCRIPT_DIR}/${path}" "${CORAL_REPO}/${path}"
        else
            copy_if_changed "${SCRIPT_DIR}/${path}" "${CORAL_REPO}/${path}"
        fi
    fi
done

echo "[coralnpu-patchset] done"
