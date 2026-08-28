#ifndef HW_SIM_GEM5_BRIDGE_XOPENNPUX_ISA_H_
#define HW_SIM_GEM5_BRIDGE_XOPENNPUX_ISA_H_

#include <cstdint>

namespace xopennpux {

// XOpenNPUX v0.1 uses the standard RISC-V custom-3 major opcode. Operator
// semantics are decoded by the NPU, not by the Coral scalar decoder.
constexpr uint32_t kCustom3Opcode = 0x7b;
constexpr uint32_t kMmaFunct3 = 0;
constexpr uint32_t kTmmaFunct7 = 0;
constexpr uint32_t kTensorTensorFunct3 = 1;
constexpr uint32_t kTaddFunct7 = 1;
constexpr uint32_t kTmulFunct7 = 2;
constexpr uint32_t kUnaryReductionFunct3 = 2;
constexpr uint32_t kRmsNormFunct7 = 0x31;
constexpr uint32_t kSoftmaxFunct7 = 0x32;
constexpr uint32_t kRopeFunct7 = 0x33;
constexpr uint32_t kSiluFunct7 = 0x46;
constexpr uint32_t kMoveFunct3 = 3;
constexpr uint32_t kGatherFunct7 = 0x10;
constexpr uint32_t kMultiOutputFunct3 = 4;
constexpr uint32_t kTopKFunct7 = 0;
constexpr uint32_t kFenceFunct3 = 6;

constexpr uint16_t kCsrMmaShape = 0x800;
constexpr uint16_t kCsrMmaDataType = 0x801;
constexpr uint16_t kCsrTensorShape = 0x802;
constexpr uint16_t kCsrTensorDataType = 0x806;
constexpr uint16_t kCsrScalarParam0 = 0x80b;

constexpr uint32_t kShapeFieldMask = 0x3ff;
constexpr uint32_t kShapeMShift = 0;
constexpr uint32_t kShapeNShift = 10;
constexpr uint32_t kShapeKShift = 20;

constexpr uint32_t kDataTypeFieldMask = 0xf;
constexpr uint32_t kDataTypeSrc1Shift = 0;
constexpr uint32_t kDataTypeSrc2Shift = 4;
constexpr uint32_t kDataTypeDstShift = 8;

enum class DataType : uint8_t {
  kFp16 = 0,
  kBf16 = 1,
  kFp32 = 2,
  kInt16 = 3,
  kInt8 = 4,
  kFp8E4M3 = 5,
  kFp8E5M2 = 6,
  kInt4 = 7,
  kInt2 = 8,
  kMxfp6 = 9,
  kMxfp4 = 10,
};

struct MmaShape {
  uint16_t m = 0;
  uint16_t n = 0;
  uint16_t k = 0;
};

struct MmaDataTypes {
  DataType src1 = DataType::kFp16;
  DataType src2 = DataType::kFp16;
  DataType dst = DataType::kFp16;
};

struct TensorShape {
  uint16_t rows = 0;
  uint16_t features = 0;
};

constexpr uint32_t EncodeTensorShape(uint32_t rows, uint32_t features) {
  return (rows & 0xffff) | ((features & 0xffff) << 16);
}

constexpr TensorShape DecodeTensorShape(uint32_t value) {
  return TensorShape{static_cast<uint16_t>(value & 0xffff),
                     static_cast<uint16_t>((value >> 16) & 0xffff)};
}

constexpr uint32_t EncodeMmaShape(uint32_t m, uint32_t n, uint32_t k) {
  return ((m & kShapeFieldMask) << kShapeMShift) |
         ((n & kShapeFieldMask) << kShapeNShift) |
         ((k & kShapeFieldMask) << kShapeKShift);
}

constexpr MmaShape DecodeMmaShape(uint32_t value) {
  return MmaShape{
      static_cast<uint16_t>((value >> kShapeMShift) & kShapeFieldMask),
      static_cast<uint16_t>((value >> kShapeNShift) & kShapeFieldMask),
      static_cast<uint16_t>((value >> kShapeKShift) & kShapeFieldMask),
  };
}

constexpr uint32_t EncodeMmaDataTypes(DataType src1, DataType src2,
                                      DataType dst) {
  return ((static_cast<uint32_t>(src1) & kDataTypeFieldMask)
          << kDataTypeSrc1Shift) |
         ((static_cast<uint32_t>(src2) & kDataTypeFieldMask)
          << kDataTypeSrc2Shift) |
         ((static_cast<uint32_t>(dst) & kDataTypeFieldMask)
          << kDataTypeDstShift);
}

constexpr MmaDataTypes DecodeMmaDataTypes(uint32_t value) {
  return MmaDataTypes{
      static_cast<DataType>((value >> kDataTypeSrc1Shift) &
                            kDataTypeFieldMask),
      static_cast<DataType>((value >> kDataTypeSrc2Shift) &
                            kDataTypeFieldMask),
      static_cast<DataType>((value >> kDataTypeDstShift) &
                            kDataTypeFieldMask),
  };
}

constexpr uint32_t EncodeTmma(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return (kTmmaFunct7 << 25) | ((rs2 & 0x1f) << 20) |
         ((rs1 & 0x1f) << 15) | (kMmaFunct3 << 12) |
         ((rd & 0x1f) << 7) | kCustom3Opcode;
}

constexpr bool IsCustom3(uint32_t instruction) {
  return (instruction & 0x7f) == kCustom3Opcode;
}

constexpr bool IsTmma(uint32_t instruction) {
  return IsCustom3(instruction) && ((instruction >> 12) & 0x7) == kMmaFunct3 &&
         ((instruction >> 25) & 0x7f) == kTmmaFunct7;
}

constexpr uint32_t EncodeTadd(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return (kTaddFunct7 << 25) | ((rs2 & 0x1f) << 20) |
         ((rs1 & 0x1f) << 15) | (kTensorTensorFunct3 << 12) |
         ((rd & 0x1f) << 7) | kCustom3Opcode;
}

constexpr bool IsTadd(uint32_t instruction) {
  return IsCustom3(instruction) &&
         ((instruction >> 12) & 0x7) == kTensorTensorFunct3 &&
         ((instruction >> 25) & 0x7f) == kTaddFunct7;
}

constexpr uint32_t EncodeTmul(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return (kTmulFunct7 << 25) | ((rs2 & 0x1f) << 20) |
         ((rs1 & 0x1f) << 15) | (kTensorTensorFunct3 << 12) |
         ((rd & 0x1f) << 7) | kCustom3Opcode;
}

constexpr bool IsTmul(uint32_t instruction) {
  return IsCustom3(instruction) &&
         ((instruction >> 12) & 0x7) == kTensorTensorFunct3 &&
         ((instruction >> 25) & 0x7f) == kTmulFunct7;
}

constexpr uint32_t EncodeTrmsnorm(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return (kRmsNormFunct7 << 25) | ((rs2 & 0x1f) << 20) |
         ((rs1 & 0x1f) << 15) | (kUnaryReductionFunct3 << 12) |
         ((rd & 0x1f) << 7) | kCustom3Opcode;
}

constexpr bool IsTrmsnorm(uint32_t instruction) {
  return IsCustom3(instruction) &&
         ((instruction >> 12) & 0x7) == kUnaryReductionFunct3 &&
         ((instruction >> 25) & 0x7f) == kRmsNormFunct7;
}

constexpr uint32_t EncodeTsoftmax(uint32_t rd, uint32_t rs1) {
  return (kSoftmaxFunct7 << 25) | ((rs1 & 0x1f) << 15) |
         (kUnaryReductionFunct3 << 12) | ((rd & 0x1f) << 7) |
         kCustom3Opcode;
}

constexpr bool IsTsoftmax(uint32_t instruction) {
  return IsCustom3(instruction) &&
         ((instruction >> 12) & 0x7) == kUnaryReductionFunct3 &&
         ((instruction >> 25) & 0x7f) == kSoftmaxFunct7 &&
         ((instruction >> 20) & 0x1f) == 0;
}

constexpr uint32_t EncodeTrope(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return (kRopeFunct7 << 25) | ((rs2 & 0x1f) << 20) |
         ((rs1 & 0x1f) << 15) | (kUnaryReductionFunct3 << 12) |
         ((rd & 0x1f) << 7) | kCustom3Opcode;
}

constexpr bool IsTrope(uint32_t instruction) {
  return IsCustom3(instruction) &&
         ((instruction >> 12) & 0x7) == kUnaryReductionFunct3 &&
         ((instruction >> 25) & 0x7f) == kRopeFunct7;
}

constexpr uint32_t EncodeTsilu(uint32_t rd, uint32_t rs1) {
  return (kSiluFunct7 << 25) | ((rs1 & 0x1f) << 15) |
         (kUnaryReductionFunct3 << 12) | ((rd & 0x1f) << 7) |
         kCustom3Opcode;
}

constexpr bool IsTsilu(uint32_t instruction) {
  return IsCustom3(instruction) &&
         ((instruction >> 12) & 0x7) == kUnaryReductionFunct3 &&
         ((instruction >> 25) & 0x7f) == kSiluFunct7 &&
         ((instruction >> 20) & 0x1f) == 0;
}

constexpr uint32_t EncodeTgather(uint32_t rd, uint32_t rs1, uint32_t rs2) {
  return (kGatherFunct7 << 25) | ((rs2 & 0x1f) << 20) |
         ((rs1 & 0x1f) << 15) | (kMoveFunct3 << 12) |
         ((rd & 0x1f) << 7) | kCustom3Opcode;
}

constexpr bool IsTgather(uint32_t instruction) {
  return IsCustom3(instruction) &&
         ((instruction >> 12) & 0x7) == kMoveFunct3 &&
         ((instruction >> 25) & 0x7f) == kGatherFunct7;
}

constexpr uint32_t EncodeTtopk(uint32_t rd, uint32_t rs1) {
  return (kTopKFunct7 << 25) | ((rs1 & 0x1f) << 15) |
         (kMultiOutputFunct3 << 12) | ((rd & 0x1f) << 7) |
         kCustom3Opcode;
}

constexpr bool IsTtopk(uint32_t instruction) {
  return IsCustom3(instruction) &&
         ((instruction >> 12) & 0x7) == kMultiOutputFunct3 &&
         ((instruction >> 25) & 0x7f) == kTopKFunct7 &&
         ((instruction >> 20) & 0x1f) == 0;
}

constexpr uint32_t EncodeTfence() {
  return (kFenceFunct3 << 12) | kCustom3Opcode;
}

constexpr bool IsTfence(uint32_t instruction) {
  return instruction == EncodeTfence();
}

enum class Operation : uint8_t {
  kInvalid,
  kTmma,
  kTadd,
  kTmul,
  kTrmsnorm,
  kTsoftmax,
  kTrope,
  kTsilu,
  kTgather,
  kTtopk,
  kTfence,
};

constexpr Operation DecodeOperation(uint32_t instruction) {
  return IsTmma(instruction) ? Operation::kTmma
         : IsTadd(instruction) ? Operation::kTadd
         : IsTmul(instruction) ? Operation::kTmul
         : IsTrmsnorm(instruction) ? Operation::kTrmsnorm
         : IsTsoftmax(instruction) ? Operation::kTsoftmax
         : IsTrope(instruction) ? Operation::kTrope
         : IsTsilu(instruction) ? Operation::kTsilu
         : IsTgather(instruction) ? Operation::kTgather
         : IsTtopk(instruction) ? Operation::kTtopk
         : IsTfence(instruction) ? Operation::kTfence
                                 : Operation::kInvalid;
}

constexpr const char* OperationName(Operation operation) {
  switch (operation) {
    case Operation::kTmma:
      return "tmma";
    case Operation::kTadd:
      return "tadd";
    case Operation::kTmul:
      return "tmul";
    case Operation::kTrmsnorm:
      return "trmsnorm";
    case Operation::kTsoftmax:
      return "tsoftmax";
    case Operation::kTrope:
      return "trope";
    case Operation::kTsilu:
      return "tsilu";
    case Operation::kTgather:
      return "tgather";
    case Operation::kTtopk:
      return "ttopk";
    case Operation::kTfence:
      return "tfence";
    default:
      return "invalid";
  }
}

}  // namespace xopennpux

#endif  // HW_SIM_GEM5_BRIDGE_XOPENNPUX_ISA_H_
