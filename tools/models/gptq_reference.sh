#!/bin/sh
# Recompute a staged GPTQ projection image on the host and print the exact
# output checksum the Coral bridge kernel must reproduce.
#
# Usage: gptq_reference.sh <gptq-projection.bin> [device-base]
# Environment: CC (host compiler, default cc)
# Output: ${ROOT_DIR}/build/local-tests/gptq-reference
# Callers: tools/models/test_model_package.sh,
#          tools/coralnpu/run_gptq_projection_test.sh
#
# -ffp-contract=off is mandatory: the reference accumulates in float32 in the
# same order as the bridge kernel, and a fused multiply-add would change the
# rounding and therefore the checksum.
set -eu

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
    echo "usage: $0 <gptq-projection.bin> [device-base]" >&2
    exit 2
fi

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
OUT_DIR="${ROOT_DIR}/build/local-tests"
CC="${CC:-cc}"
mkdir -p "$OUT_DIR"
"$CC" -O2 -Wall -Wextra -Werror -std=c11 -ffp-contract=off \
    -I"${ROOT_DIR}/runtime/host/include" \
    "${ROOT_DIR}/runtime/host/src/npu_gptq_reference.c" \
    "${ROOT_DIR}/runtime/host/tools/gptq_reference.c" \
    -lm \
    -o "${OUT_DIR}/gptq-reference"
exec "${OUT_DIR}/gptq-reference" "$@"
