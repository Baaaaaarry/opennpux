#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
MODULAR_DIR="${MODULAR_SOURCE_DIR:-${ROOT_DIR}/thirdparty/modular}"
RULES_MOJO_SOURCE="${ROOT_DIR}/thirdparty/rules_mojo"
RULES_CC_SOURCE="${ROOT_DIR}/thirdparty/rules_cc"
RULES_MOJO_CACHE="${ROOT_DIR}/.cache/modular-deps/rules_mojo"
BATS_CORE_SOURCE="${ROOT_DIR}/thirdparty/bats-core"
BATS_CORE_CACHE="${ROOT_DIR}/.cache/modular-deps/bats-core"
MOJO_CONFIG="${MODULAR_MOJO_CONFIG:-prebuilt-mojo}"
BUILD_DIR="${MAX_SOURCE_BUILD_DIR:-${ROOT_DIR}/build/local-tests/max-source}"
OUTPUT="${1:-${BUILD_DIR}/max-source-projection.npxg}"
LOWERING_LIBRARY="${2:-${ROOT_DIR}/build/local-tests/mojo-max-xgraph/libopennpux_xgraph_codegen.so}"
OVERLAY_SOURCE="${ROOT_DIR}/integrations/modular/source_export"
OVERLAY_DESTINATION="${MODULAR_DIR}/max/opennpux"
EXPORT_JSON="${BUILD_DIR}/max-source-projection.json"

case "${MOJO_CONFIG}" in
    prebuilt-mojo|build-mojo) ;;
    *)
        echo "max_source_export=FAIL invalid-mojo-config=${MOJO_CONFIG}" >&2
        exit 2
        ;;
esac

"${SCRIPT_DIR}/setup_modular_source.sh"

rules_mojo_revision="$(git -C "${RULES_MOJO_SOURCE}" rev-parse HEAD)"
RULES_MOJO_PATCHED="${RULES_MOJO_CACHE}/${rules_mojo_revision}"
patch_revision_file="${RULES_MOJO_PATCHED}/.opennpux-source-revision"
patched_revision=
if [ -f "${patch_revision_file}" ]; then
    patched_revision="$(cat "${patch_revision_file}")"
fi
if [ "${patched_revision}" != "${rules_mojo_revision}" ]; then
    if [ -e "${RULES_MOJO_PATCHED}" ]; then
        echo "max_source_export=FAIL incomplete-cache=${RULES_MOJO_PATCHED}" >&2
        exit 1
    fi
    patched_temporary="${RULES_MOJO_PATCHED}.tmp.$$"
    mkdir -p "${RULES_MOJO_CACHE}" "${patched_temporary}"
    cp -R "${RULES_MOJO_SOURCE}/." "${patched_temporary}/"
    (cd "${patched_temporary}" && patch -p1 < \
        "${MODULAR_DIR}/bazel/public-patches/rules_mojo_toolchain.patch")
    printf '%s\n' "${rules_mojo_revision}" > \
        "${patched_temporary}/.opennpux-source-revision"
    mv "${patched_temporary}" "${RULES_MOJO_PATCHED}"
fi

bats_core_revision="$(git -C "${BATS_CORE_SOURCE}" rev-parse HEAD)"
BATS_CORE_LOCAL="${BATS_CORE_CACHE}/${bats_core_revision}"
if [ ! -f "${BATS_CORE_LOCAL}/.opennpux-source-revision" ]; then
    if [ -e "${BATS_CORE_LOCAL}" ]; then
        echo "max_source_export=FAIL incomplete-cache=${BATS_CORE_LOCAL}" >&2
        exit 1
    fi
    bats_temporary="${BATS_CORE_LOCAL}.tmp.$$"
    mkdir -p "${BATS_CORE_CACHE}" "${bats_temporary}"
    cp -R "${BATS_CORE_SOURCE}/." "${bats_temporary}/"
    cp "${ROOT_DIR}/integrations/modular/bats-core.BUILD.bazel" \
        "${bats_temporary}/BUILD.bazel"
    cp "${ROOT_DIR}/integrations/modular/bats-core.REPO.bazel" \
        "${bats_temporary}/REPO.bazel"
    printf '%s\n' "${bats_core_revision}" > \
        "${bats_temporary}/.opennpux-source-revision"
    mv "${bats_temporary}" "${BATS_CORE_LOCAL}"
fi
# Repair caches produced by older runners without rebuilding the source mirror.
if [ ! -f "${BATS_CORE_LOCAL}/REPO.bazel" ]; then
    cp "${ROOT_DIR}/integrations/modular/bats-core.REPO.bazel" \
        "${BATS_CORE_LOCAL}/REPO.bazel"
fi

mkdir -p "${OVERLAY_DESTINATION}" "${BUILD_DIR}" "$(dirname -- "${OUTPUT}")"
cp "${OVERLAY_SOURCE}/BUILD.bazel" "${OVERLAY_DESTINATION}/BUILD.bazel"
cp "${OVERLAY_SOURCE}/export_projection.py" \
    "${OVERLAY_DESTINATION}/export_projection.py"

(cd "${MODULAR_DIR}" && \
    ./bazelw run --config="${MOJO_CONFIG}" \
        --override_repository="rules_mojo=${RULES_MOJO_PATCHED}" \
        --override_repository="rules_cc=${RULES_CC_SOURCE}" \
        --override_repository="aspect_bazel_lib++toolchains+bats_toolchains=${BATS_CORE_LOCAL}" \
        //max/opennpux:export_projection -- "${EXPORT_JSON}")

if [ ! -f "${LOWERING_LIBRARY}" ]; then
    echo "max_source_export=FAIL missing-lowering-library=${LOWERING_LIBRARY}" >&2
    echo "run ./tools/models/test_mojo_max_xgraph_codegen.sh first" >&2
    exit 1
fi

PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}" python3 \
    "${SCRIPT_DIR}/compile_mojo_max_xgraph.py" \
    "${EXPORT_JSON}" "${OUTPUT}" \
    --lowering-library "${LOWERING_LIBRARY}"

echo "max_source_artifact=${OUTPUT}"
echo "max_source_opennpux_export=PASS"
