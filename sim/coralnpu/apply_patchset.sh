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
       [ "${source_path}" -nt "${destination_path}" ] ||
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

# The bridge consumes the same binary tensor-plan implementation as the guest
# runtime. Generate these files in the Coral Bazel workspace rather than
# maintaining a second ABI/parser implementation in the overlay.
RUNTIME_PLAN_DIR="${CORAL_REPO}/hw_sim/gem5_bridge/opennpux"
mkdir -p "${RUNTIME_PLAN_DIR}"
copy_if_changed \
    "${SUPER_ROOT}/runtime/host/include/opennpux/npu_tensor_plan.h" \
    "${RUNTIME_PLAN_DIR}/npu_tensor_plan.h"
copy_if_changed \
    "${SUPER_ROOT}/runtime/host/src/npu_tensor_plan.c" \
    "${RUNTIME_PLAN_DIR}/npu_tensor_plan.c"
for runtime_file in \
    npu_submission.h npu_functional_request.h npu_functional_materializer.h \
    npu_weight_ranges.h; do
    copy_if_changed \
        "${SUPER_ROOT}/runtime/host/include/opennpux/${runtime_file}" \
        "${RUNTIME_PLAN_DIR}/${runtime_file}"
done
for runtime_file in npu_submission.c npu_functional_materializer.c; do
    copy_if_changed \
        "${SUPER_ROOT}/runtime/host/src/${runtime_file}" \
        "${RUNTIME_PLAN_DIR}/${runtime_file}"
done

echo "[coralnpu-patchset] done"
