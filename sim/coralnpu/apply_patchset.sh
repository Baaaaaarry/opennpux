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
            mkdir -p "${CORAL_REPO}/${path}"
            cp -R "${SCRIPT_DIR}/${path}/." "${CORAL_REPO}/${path}/"
        else
            cp "${SCRIPT_DIR}/${path}" "${CORAL_REPO}/${path}"
        fi
    fi
done

echo "[coralnpu-patchset] done"
