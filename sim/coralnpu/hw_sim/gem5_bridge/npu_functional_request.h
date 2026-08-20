#ifndef HW_SIM_GEM5_BRIDGE_NPU_FUNCTIONAL_REQUEST_H_
#define HW_SIM_GEM5_BRIDGE_NPU_FUNCTIONAL_REQUEST_H_

#include <stdint.h>

#define OPENNPUX_NPU_FUNCTIONAL_MAGIC UINT32_C(0x4658504e)
#define OPENNPUX_NPU_FUNCTIONAL_VERSION UINT32_C(2)
#define OPENNPUX_NPU_FUNCTIONAL_MAX_OPERANDS UINT32_C(20)
#define OPENNPUX_NPU_FUNCTIONAL_REQUEST_SIZE UINT32_C(432)

enum opennpux_npu_functional_operand_role {
    OPENNPUX_NPU_OPERAND_INPUT = 1,
    OPENNPUX_NPU_OPERAND_SECONDARY = 2,
    OPENNPUX_NPU_OPERAND_WEIGHT = 3,
    OPENNPUX_NPU_OPERAND_POSITIONS = 4,
    OPENNPUX_NPU_OPERAND_OUTPUT = 5,
    OPENNPUX_NPU_OPERAND_OUTPUT_INDICES = 6,
    OPENNPUX_NPU_OPERAND_QWEIGHT = 7,
    OPENNPUX_NPU_OPERAND_QZEROS = 8,
    OPENNPUX_NPU_OPERAND_SCALES = 9,
    OPENNPUX_NPU_OPERAND_G_IDX = 10,
    OPENNPUX_NPU_OPERAND_GATE_QWEIGHT = 11,
    OPENNPUX_NPU_OPERAND_GATE_QZEROS = 12,
    OPENNPUX_NPU_OPERAND_GATE_SCALES = 13,
    OPENNPUX_NPU_OPERAND_GATE_G_IDX = 14,
    OPENNPUX_NPU_OPERAND_UP_QWEIGHT = 15,
    OPENNPUX_NPU_OPERAND_UP_QZEROS = 16,
    OPENNPUX_NPU_OPERAND_UP_SCALES = 17,
    OPENNPUX_NPU_OPERAND_UP_G_IDX = 18,
    OPENNPUX_NPU_OPERAND_DOWN_QWEIGHT = 19,
    OPENNPUX_NPU_OPERAND_DOWN_QZEROS = 20,
    OPENNPUX_NPU_OPERAND_DOWN_SCALES = 21,
    OPENNPUX_NPU_OPERAND_DOWN_G_IDX = 22,
    OPENNPUX_NPU_OPERAND_GATE_OUTPUT = 23,
    OPENNPUX_NPU_OPERAND_UP_OUTPUT = 24,
    OPENNPUX_NPU_OPERAND_ACTIVATED = 25,
    OPENNPUX_NPU_OPERAND_INPUT_INDICES = 26,
    OPENNPUX_NPU_OPERAND_INPUT_TERTIARY = 27,
    OPENNPUX_NPU_OPERAND_INPUT_QUATERNARY = 28,
    OPENNPUX_NPU_OPERAND_OUTPUT_SECONDARY = 29,
    OPENNPUX_NPU_OPERAND_OUTPUT_TERTIARY = 30,
    OPENNPUX_NPU_OPERAND_Q_QWEIGHT = 31,
    OPENNPUX_NPU_OPERAND_Q_QZEROS = 32,
    OPENNPUX_NPU_OPERAND_Q_SCALES = 33,
    OPENNPUX_NPU_OPERAND_Q_G_IDX = 34,
    OPENNPUX_NPU_OPERAND_K_QWEIGHT = 35,
    OPENNPUX_NPU_OPERAND_K_QZEROS = 36,
    OPENNPUX_NPU_OPERAND_K_SCALES = 37,
    OPENNPUX_NPU_OPERAND_K_G_IDX = 38,
    OPENNPUX_NPU_OPERAND_V_QWEIGHT = 39,
    OPENNPUX_NPU_OPERAND_V_QZEROS = 40,
    OPENNPUX_NPU_OPERAND_V_SCALES = 41,
    OPENNPUX_NPU_OPERAND_V_G_IDX = 42,
    OPENNPUX_NPU_OPERAND_LINEAR_QKV_WEIGHT = 43,
    OPENNPUX_NPU_OPERAND_LINEAR_ALPHA_WEIGHT = 44,
    OPENNPUX_NPU_OPERAND_LINEAR_BETA_WEIGHT = 45,
    OPENNPUX_NPU_OPERAND_LINEAR_GATE_WEIGHT = 46,
    OPENNPUX_NPU_OPERAND_LINEAR_NORM_WEIGHT = 47,
};

struct opennpux_npu_functional_operand {
    uint32_t role;
    uint32_t address;
    uint32_t byte_size;
    uint32_t reserved;
};

// Firmware creates this address-based envelope after resolving the generic
// executable's tensor plan. The bridge translates it to host pointers only at
// the execution boundary, so no host pointer crosses the firmware ABI.
struct opennpux_npu_functional_request {
    uint32_t magic;
    uint32_t version;
    uint32_t struct_size;
    uint32_t opcode;
    uint32_t command_id;
    uint32_t state;
    uint32_t error;
    uint32_t operand_count;
    uint32_t parameter_address;
    uint32_t parameter_size;
    uint32_t rows;
    uint32_t features;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t top_k;
    uint32_t vocabulary_size;
    uint32_t kv_heads;
    uint32_t kv_length;
    float epsilon;
    float rope_theta;
    struct opennpux_npu_functional_operand
        operands[OPENNPUX_NPU_FUNCTIONAL_MAX_OPERANDS];
    uint64_t operation_count;
    uint64_t modeled_cycles;
    uint64_t bytes_read;
    uint64_t bytes_written;
};

#if defined(__cplusplus)
static_assert(sizeof(struct opennpux_npu_functional_operand) == 16);
static_assert(sizeof(struct opennpux_npu_functional_request) ==
              OPENNPUX_NPU_FUNCTIONAL_REQUEST_SIZE);
#else
_Static_assert(sizeof(struct opennpux_npu_functional_operand) == 16,
               "functional operand ABI size changed");
_Static_assert(sizeof(struct opennpux_npu_functional_request) ==
                   OPENNPUX_NPU_FUNCTIONAL_REQUEST_SIZE,
               "functional request ABI size changed");
#endif

#endif  // HW_SIM_GEM5_BRIDGE_NPU_FUNCTIONAL_REQUEST_H_
