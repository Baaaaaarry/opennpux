#!/bin/sh

set -eu

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
SUPER_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"
DEFAULT_CORAL_REPO="${SUPER_ROOT}/thirdparty/coralnpu"

CORAL_REPO="${1:-${CORAL_REPO:-${DEFAULT_CORAL_REPO}}}"

if [ -z "${CORAL_REPO}" ]; then
    echo "usage: $0 /abs/path/to/google-coral/coralnpu" >&2
    exit 1
fi

if [ ! -d "${CORAL_REPO}" ]; then
    echo "error: Coral repo not found: ${CORAL_REPO}" >&2
    exit 1
fi

required_paths="
README.md
doc/integration_guide.md
hdl/chisel/src/coralnpu/CoreAxi.scala
hdl/chisel/src/coralnpu/CoreAxiCSR.scala
hw_sim/core_mini_axi_wrapper.h
"

echo "[coral-checkout] validating Coral checkout: ${CORAL_REPO}"

missing=0
for rel in ${required_paths}; do
    if [ ! -e "${CORAL_REPO}/${rel}" ]; then
        echo "[coral-checkout] missing: ${rel}" >&2
        missing=1
    else
        echo "[coral-checkout] found:   ${rel}"
    fi
done

if [ "${missing}" -ne 0 ]; then
    echo "[coral-checkout] validation failed" >&2
    exit 1
fi

cat <<EOF
[coral-checkout] checkout looks structurally correct

Expected repository-local path:
  ${DEFAULT_CORAL_REPO}

Recommended Coral standalone build path:
  ./tools/coralnpu/build_coral_standalone_docker.sh

Direct host builds are only expected to work on x86_64 Linux because the
official Coral host and target toolchains are registered for that exec
platform. On macOS/arm64, use the Docker wrapper above.

Bridge target for the gem5 RTL bridge:
  ${CORAL_REPO}/hw_sim/core_mini_axi_wrapper.h
EOF
