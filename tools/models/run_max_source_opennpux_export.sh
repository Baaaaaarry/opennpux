#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
MODULAR_DIR="${MODULAR_SOURCE_DIR:-${ROOT_DIR}/thirdparty/modular}"
BUILD_DIR="${MAX_SOURCE_BUILD_DIR:-${ROOT_DIR}/build/local-tests/max-source}"
OUTPUT="${1:-${BUILD_DIR}/max-source-projection.npxg}"
LOWERING_LIBRARY="${2:-${ROOT_DIR}/build/local-tests/mojo-max-xgraph/libopennpux_xgraph_codegen.so}"
OVERLAY_SOURCE="${ROOT_DIR}/integrations/modular/source_export"
OVERLAY_DESTINATION="${MODULAR_DIR}/max/opennpux"
EXPORT_JSON="${BUILD_DIR}/max-source-projection.json"

"${SCRIPT_DIR}/setup_modular_source.sh"

mkdir -p "${OVERLAY_DESTINATION}" "${BUILD_DIR}" "$(dirname -- "${OUTPUT}")"
cp "${OVERLAY_SOURCE}/BUILD.bazel" "${OVERLAY_DESTINATION}/BUILD.bazel"
cp "${OVERLAY_SOURCE}/export_projection.py" \
    "${OVERLAY_DESTINATION}/export_projection.py"

(cd "${MODULAR_DIR}" && \
    ./bazelw run //max/opennpux:export_projection -- "${EXPORT_JSON}")

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
