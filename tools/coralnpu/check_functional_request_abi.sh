#!/bin/sh
set -eu

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd -P)"
CC="${CC:-cc}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "${TMP_DIR}"' EXIT HUP INT TERM

emit_probe()
{
    header="$1"
    source="$2"
    cat >"${source}" <<EOF
#include <stdio.h>
#include "${header}"
int main(void) {
    printf("%zu %zu %u %u %u %u %u %u\n",
           sizeof(struct opennpux_npu_functional_request),
           sizeof(struct opennpux_npu_functional_operand),
           OPENNPUX_NPU_FUNCTIONAL_MAX_OPERANDS,
           OPENNPUX_NPU_OPERAND_INPUT_INDICES,
           OPENNPUX_NPU_OPERAND_INPUT_TERTIARY,
           OPENNPUX_NPU_OPERAND_INPUT_QUATERNARY,
           OPENNPUX_NPU_OPERAND_OUTPUT_SECONDARY,
           OPENNPUX_NPU_OPERAND_OUTPUT_TERTIARY);
    return 0;
}
EOF
}

emit_probe "opennpux/npu_functional_request.h" "${TMP_DIR}/host.c"
emit_probe "hw_sim/gem5_bridge/npu_functional_request.h" "${TMP_DIR}/rtl.c"
"${CC}" -std=c11 -I"${ROOT_DIR}/runtime/host/include" \
    "${TMP_DIR}/host.c" -o "${TMP_DIR}/host"
"${CC}" -std=c11 -I"${ROOT_DIR}/sim/coralnpu" \
    "${TMP_DIR}/rtl.c" -o "${TMP_DIR}/rtl"
"${TMP_DIR}/host" >"${TMP_DIR}/host.out"
"${TMP_DIR}/rtl" >"${TMP_DIR}/rtl.out"
cmp "${TMP_DIR}/host.out" "${TMP_DIR}/rtl.out"
echo "Coral functional request ABI headers match"
