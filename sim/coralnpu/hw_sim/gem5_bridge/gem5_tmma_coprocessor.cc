#include "hw_sim/gem5_bridge/gem5_tmma_coprocessor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace {

bool MatrixRangeValid(uint32_t address, uint64_t elements,
                      uint32_t memory_base, size_t memory_size) {
  constexpr uint64_t kElementBytes = sizeof(float);
  if ((address & (kElementBytes - 1)) != 0 || address < memory_base) {
    return false;
  }
  const uint64_t offset = static_cast<uint64_t>(address - memory_base);
  const uint64_t bytes = elements * kElementBytes;
  return offset <= memory_size && bytes <= memory_size - offset;
}

bool ByteRangeValid(uint32_t address, uint64_t bytes, uint32_t alignment,
                    uint32_t memory_base, size_t memory_size) {
  if (alignment == 0 || address % alignment != 0 || address < memory_base) {
    return false;
  }
  const uint64_t offset = static_cast<uint64_t>(address - memory_base);
  return bytes != 0 && offset <= memory_size && bytes <= memory_size - offset;
}

bool StridedMatrixRangeValid(uint32_t address, uint32_t rows,
                             uint32_t row_bytes, uint32_t stride,
                             uint32_t memory_base, size_t memory_size) {
  if (rows == 0 || row_bytes == 0 || stride < row_bytes ||
      (stride & (sizeof(float) - 1)) != 0) {
    return false;
  }
  const uint64_t bytes = static_cast<uint64_t>(rows - 1) * stride + row_bytes;
  return ByteRangeValid(address, bytes, sizeof(float), memory_base,
                        memory_size);
}

float LoadFloat(const std::vector<uint8_t>& memory, size_t offset) {
  float value = 0.0f;
  std::memcpy(&value, memory.data() + offset, sizeof(value));
  return value;
}

void StoreFloat(std::vector<uint8_t>* memory, size_t offset, float value) {
  std::memcpy(memory->data() + offset, &value, sizeof(value));
}

uint32_t LoadUint32(const std::vector<uint8_t>& memory, size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, memory.data() + offset, sizeof(value));
  return value;
}

void StoreUint32(std::vector<uint8_t>* memory, size_t offset, uint32_t value) {
  std::memcpy(memory->data() + offset, &value, sizeof(value));
}

float DecodeFloat(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float RoundBfloat16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000)) {
    bits += UINT32_C(0x00007fff) + ((bits >> 16) & 1U);
    bits &= UINT32_C(0xffff0000);
    std::memcpy(&value, &bits, sizeof(bits));
  }
  return value;
}

float DecodeFloat16(uint16_t value) {
  const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
  const uint32_t exponent = (value >> 10) & 0x1fu;
  uint32_t mantissa = value & 0x3ffu;
  if (exponent == 0) {
    if (mantissa == 0) return DecodeFloat(sign);
    uint32_t normalized_exponent = 113;
    while ((mantissa & 0x400u) == 0) {
      mantissa <<= 1;
      --normalized_exponent;
    }
    return DecodeFloat(sign | (normalized_exponent << 23) |
                       ((mantissa & 0x3ffu) << 13));
  }
  if (exponent == 0x1f) {
    return DecodeFloat(sign | 0x7f800000u | (mantissa << 13));
  }
  return DecodeFloat(sign | ((exponent + 112) << 23) | (mantissa << 13));
}

float LoadScale(const std::vector<uint8_t>& memory, size_t offset,
                xopennpux::DataType data_type) {
  if (data_type == xopennpux::DataType::kFp32) {
    return LoadFloat(memory, offset);
  }
  uint16_t value = 0;
  std::memcpy(&value, memory.data() + offset, sizeof(value));
  if (data_type == xopennpux::DataType::kBf16) {
    return DecodeFloat(static_cast<uint32_t>(value) << 16);
  }
  return DecodeFloat16(value);
}

uint32_t Fnv1a(const uint8_t* data, size_t size) {
  uint32_t hash = 2166136261u;
  for (size_t index = 0; index < size; ++index) {
    hash = (hash ^ data[index]) * 16777619u;
  }
  return hash;
}

}  // namespace

void Gem5XOpenNpuFunctionalCoprocessor::Reset() {
  queue_.fill({});
  queue_head_ = 0;
  queue_size_ = 0;
  mma_shape_ = 0;
  mma_data_type_ = 0;
  tensor_shape_ = 0;
  tensor_data_type_ = 0;
  scalar_param0_ = 0;
  quant_qzeros_address_ = 0;
  quant_scales_address_ = 0;
  quant_g_idx_address_ = 0;
  quant_config_ = 0;
  quant_qweight_stride_ = 0;
  quant_qzeros_stride_ = 0;
  quant_scales_stride_ = 0;
  quant_group_range_ = 0;
  tensor_aux_source_address_ = 0;
  tensor_aux_destination_address_ = 0;
  attention_heads_ = 0;
  attention_head_dim_flags_ = 0;
  attention_kv_length_ = 0;
  recurrent_heads_ = 0;
  recurrent_dims_ = 0;
  recurrent_beta_address_ = 0;
  recurrent_a_log_address_ = 0;
  recurrent_dt_bias_address_ = 0;
  conv_input_hw_ = 0;
  conv_output_hw_ = 0;
  conv_channels_groups_ = 0;
  conv_kernel_hw_ = 0;
  conv_stride_hw_ = 0;
  conv_padding_tl_ = 0;
  conv_padding_br_ = 0;
  conv_dilation_hw_ = 0;
  conv_bias_address_ = 0;
  mma_lhs_stride_ = 0;
  mma_rhs_stride_ = 0;
  mma_dst_stride_ = 0;
  mma_flags_ = 0;
  tensor_flags_ = 0;
  csr_epoch_ = 0;
}

bool Gem5XOpenNpuFunctionalCoprocessor::WriteCsr(uint16_t address,
                                                 uint32_t value) {
  switch (address) {
    case xopennpux::kCsrMmaShape:
      mma_shape_ = value;
      break;
    case xopennpux::kCsrMmaDataType:
      mma_data_type_ = value;
      break;
    case xopennpux::kCsrTensorShape:
      tensor_shape_ = value;
      break;
    case xopennpux::kCsrTensorDataType:
      tensor_data_type_ = value;
      break;
    case xopennpux::kCsrScalarParam0:
      scalar_param0_ = value;
      break;
    case xopennpux::kCsrQuantQzerosAddress:
      quant_qzeros_address_ = value;
      break;
    case xopennpux::kCsrQuantScalesAddress:
      quant_scales_address_ = value;
      break;
    case xopennpux::kCsrQuantGIdxAddress:
      quant_g_idx_address_ = value;
      break;
    case xopennpux::kCsrQuantConfig:
      quant_config_ = value;
      break;
    case xopennpux::kCsrQuantQweightStride:
      quant_qweight_stride_ = value;
      break;
    case xopennpux::kCsrQuantQzerosStride:
      quant_qzeros_stride_ = value;
      break;
    case xopennpux::kCsrQuantScalesStride:
      quant_scales_stride_ = value;
      break;
    case xopennpux::kCsrQuantGroupRange:
      quant_group_range_ = value;
      break;
    case xopennpux::kCsrTensorAuxSourceAddress:
      tensor_aux_source_address_ = value;
      break;
    case xopennpux::kCsrTensorAuxDestinationAddress:
      tensor_aux_destination_address_ = value;
      break;
    case xopennpux::kCsrAttentionHeads:
      attention_heads_ = value;
      break;
    case xopennpux::kCsrAttentionHeadDimFlags:
      attention_head_dim_flags_ = value;
      break;
    case xopennpux::kCsrAttentionKvLength:
      attention_kv_length_ = value;
      break;
    case xopennpux::kCsrRecurrentHeads:
      recurrent_heads_ = value;
      break;
    case xopennpux::kCsrRecurrentDims:
      recurrent_dims_ = value;
      break;
    case xopennpux::kCsrRecurrentBetaAddress:
      recurrent_beta_address_ = value;
      break;
    case xopennpux::kCsrRecurrentALogAddress:
      recurrent_a_log_address_ = value;
      break;
    case xopennpux::kCsrRecurrentDtBiasAddress:
      recurrent_dt_bias_address_ = value;
      break;
    case xopennpux::kCsrConvInputHw:
      conv_input_hw_ = value;
      break;
    case xopennpux::kCsrConvOutputHw:
      conv_output_hw_ = value;
      break;
    case xopennpux::kCsrConvChannelsGroups:
      conv_channels_groups_ = value;
      break;
    case xopennpux::kCsrConvKernelHw:
      conv_kernel_hw_ = value;
      break;
    case xopennpux::kCsrConvStrideHw:
      conv_stride_hw_ = value;
      break;
    case xopennpux::kCsrConvPaddingTl:
      conv_padding_tl_ = value;
      break;
    case xopennpux::kCsrConvPaddingBr:
      conv_padding_br_ = value;
      break;
    case xopennpux::kCsrConvDilationHw:
      conv_dilation_hw_ = value;
      break;
    case xopennpux::kCsrConvBiasAddress:
      conv_bias_address_ = value;
      break;
    case xopennpux::kCsrMmaLhsStride:
      mma_lhs_stride_ = value;
      break;
    case xopennpux::kCsrMmaRhsStride:
      mma_rhs_stride_ = value;
      break;
    case xopennpux::kCsrMmaDstStride:
      mma_dst_stride_ = value;
      break;
    case xopennpux::kCsrMmaFlags:
      mma_flags_ = value;
      break;
    case xopennpux::kCsrTensorFlags:
      tensor_flags_ = value;
      break;
    default:
      return false;
  }
  ++csr_epoch_;
  return true;
}

bool Gem5XOpenNpuFunctionalCoprocessor::ReadCsr(uint16_t address,
                                                uint32_t* value) const {
  if (value == nullptr) {
    return false;
  }
  switch (address) {
    case xopennpux::kCsrMmaShape:
      *value = mma_shape_;
      return true;
    case xopennpux::kCsrMmaDataType:
      *value = mma_data_type_;
      return true;
    case xopennpux::kCsrTensorShape:
      *value = tensor_shape_;
      return true;
    case xopennpux::kCsrTensorDataType:
      *value = tensor_data_type_;
      return true;
    case xopennpux::kCsrScalarParam0:
      *value = scalar_param0_;
      return true;
    case xopennpux::kCsrQuantQzerosAddress:
      *value = quant_qzeros_address_;
      return true;
    case xopennpux::kCsrQuantScalesAddress:
      *value = quant_scales_address_;
      return true;
    case xopennpux::kCsrQuantGIdxAddress:
      *value = quant_g_idx_address_;
      return true;
    case xopennpux::kCsrQuantConfig:
      *value = quant_config_;
      return true;
    case xopennpux::kCsrQuantQweightStride:
      *value = quant_qweight_stride_;
      return true;
    case xopennpux::kCsrQuantQzerosStride:
      *value = quant_qzeros_stride_;
      return true;
    case xopennpux::kCsrQuantScalesStride:
      *value = quant_scales_stride_;
      return true;
    case xopennpux::kCsrQuantGroupRange:
      *value = quant_group_range_;
      return true;
    case xopennpux::kCsrTensorAuxSourceAddress:
      *value = tensor_aux_source_address_;
      return true;
    case xopennpux::kCsrTensorAuxDestinationAddress:
      *value = tensor_aux_destination_address_;
      return true;
    case xopennpux::kCsrAttentionHeads:
      *value = attention_heads_;
      return true;
    case xopennpux::kCsrAttentionHeadDimFlags:
      *value = attention_head_dim_flags_;
      return true;
    case xopennpux::kCsrAttentionKvLength:
      *value = attention_kv_length_;
      return true;
    case xopennpux::kCsrRecurrentHeads:
      *value = recurrent_heads_;
      return true;
    case xopennpux::kCsrRecurrentDims:
      *value = recurrent_dims_;
      return true;
    case xopennpux::kCsrRecurrentBetaAddress:
      *value = recurrent_beta_address_;
      return true;
    case xopennpux::kCsrRecurrentALogAddress:
      *value = recurrent_a_log_address_;
      return true;
    case xopennpux::kCsrRecurrentDtBiasAddress:
      *value = recurrent_dt_bias_address_;
      return true;
    case xopennpux::kCsrConvInputHw:
      *value = conv_input_hw_;
      return true;
    case xopennpux::kCsrConvOutputHw:
      *value = conv_output_hw_;
      return true;
    case xopennpux::kCsrConvChannelsGroups:
      *value = conv_channels_groups_;
      return true;
    case xopennpux::kCsrConvKernelHw:
      *value = conv_kernel_hw_;
      return true;
    case xopennpux::kCsrConvStrideHw:
      *value = conv_stride_hw_;
      return true;
    case xopennpux::kCsrConvPaddingTl:
      *value = conv_padding_tl_;
      return true;
    case xopennpux::kCsrConvPaddingBr:
      *value = conv_padding_br_;
      return true;
    case xopennpux::kCsrConvDilationHw:
      *value = conv_dilation_hw_;
      return true;
    case xopennpux::kCsrConvBiasAddress:
      *value = conv_bias_address_;
      return true;
    case xopennpux::kCsrMmaLhsStride:
      *value = mma_lhs_stride_;
      return true;
    case xopennpux::kCsrMmaRhsStride:
      *value = mma_rhs_stride_;
      return true;
    case xopennpux::kCsrMmaDstStride:
      *value = mma_dst_stride_;
      return true;
    case xopennpux::kCsrMmaFlags:
      *value = mma_flags_;
      return true;
    case xopennpux::kCsrTensorFlags:
      *value = tensor_flags_;
      return true;
    default:
      return false;
  }
}

Gem5TmmaSubmitResult Gem5XOpenNpuFunctionalCoprocessor::Classify(
    const Gem5TmmaDispatchPacket& packet) const {
  if (xopennpux::IsTfence(packet.instruction)) {
    return queue_size_ == 0 ? Gem5TmmaSubmitResult::kAccepted
                            : Gem5TmmaSubmitResult::kBackpressure;
  }
  const xopennpux::Operation operation =
      xopennpux::DecodeOperation(packet.instruction);
  if (operation != xopennpux::Operation::kTmma &&
      operation != xopennpux::Operation::kTadd &&
      operation != xopennpux::Operation::kTmul &&
      operation != xopennpux::Operation::kTrowScale &&
      operation != xopennpux::Operation::kTrmsnorm &&
      operation != xopennpux::Operation::kTsoftmax &&
      operation != xopennpux::Operation::kTrope &&
      operation != xopennpux::Operation::kTsilu &&
      operation != xopennpux::Operation::kTsigmoid &&
      operation != xopennpux::Operation::kTgather &&
      operation != xopennpux::Operation::kTdequant &&
      operation != xopennpux::Operation::kTdma &&
      operation != xopennpux::Operation::kTcausalconv &&
      operation != xopennpux::Operation::kTattention &&
      operation != xopennpux::Operation::kTrecurrent &&
      operation != xopennpux::Operation::kTconv &&
      operation != xopennpux::Operation::kTtopk) {
    return Gem5TmmaSubmitResult::kIllegalInstruction;
  }
  if (!ready()) {
    return Gem5TmmaSubmitResult::kBackpressure;
  }

  const bool is_mma = operation == xopennpux::Operation::kTmma ||
                      operation == xopennpux::Operation::kTdequant;
  const uint32_t shape_csr = packet.csr_epoch == 0
                                 ? (is_mma ? mma_shape_ : tensor_shape_)
                                 : (is_mma ? packet.mma_shape
                                           : packet.tensor_shape);
  const uint32_t data_type_csr = packet.csr_epoch == 0
                                     ? (is_mma ? mma_data_type_
                                               : tensor_data_type_)
                                     : (is_mma ? packet.mma_data_type
                                               : packet.tensor_data_type);
  const xopennpux::MmaShape shape = xopennpux::DecodeMmaShape(shape_csr);
  const xopennpux::TensorShape tensor_shape =
      xopennpux::DecodeTensorShape(shape_csr);
  const xopennpux::MmaDataTypes data_types =
      xopennpux::DecodeMmaDataTypes(data_type_csr);
  constexpr uint32_t kShapeReservedMask = 0xc0000000;
  constexpr uint32_t kDataTypeReservedMask = 0xfffff000;
  if ((is_mma && (shape.m == 0 || shape.n == 0 || shape.k == 0 ||
                  (shape_csr & kShapeReservedMask) != 0)) ||
      (!is_mma &&
       (tensor_shape.rows == 0 || tensor_shape.features == 0)) ||
      (data_type_csr & kDataTypeReservedMask) != 0 ||
      (operation != xopennpux::Operation::kTdequant &&
       (data_types.src1 != xopennpux::DataType::kFp32 ||
        data_types.src2 != xopennpux::DataType::kFp32 ||
        data_types.dst != xopennpux::DataType::kFp32))) {
    return Gem5TmmaSubmitResult::kInvalidCsrState;
  }
  if (operation == xopennpux::Operation::kTmma) {
    const uint32_t flags =
        packet.csr_epoch == 0 ? mma_flags_ : packet.mma_flags;
    const bool transpose_rhs =
        (flags & xopennpux::kMmaFlagTransposeRhs) != 0;
    const uint32_t lhs_stride = packet.csr_epoch == 0
                                    ? mma_lhs_stride_
                                    : packet.mma_lhs_stride;
    const uint32_t rhs_stride = packet.csr_epoch == 0
                                    ? mma_rhs_stride_
                                    : packet.mma_rhs_stride;
    const uint32_t dst_stride = packet.csr_epoch == 0
                                    ? mma_dst_stride_
                                    : packet.mma_dst_stride;
    const uint32_t effective_lhs_stride =
        lhs_stride == 0 ? shape.k * sizeof(float) : lhs_stride;
    const uint32_t effective_rhs_stride =
        rhs_stride == 0
            ? (transpose_rhs ? shape.k : shape.n) * sizeof(float)
            : rhs_stride;
    const uint32_t effective_dst_stride =
        dst_stride == 0 ? shape.n * sizeof(float) : dst_stride;
    const uint32_t minimum_rhs_stride =
        (transpose_rhs ? shape.k : shape.n) * sizeof(float);
    if ((flags & ~(xopennpux::kMmaFlagTransposeRhs |
                   xopennpux::kMmaFlagAccumulate)) != 0 ||
        effective_lhs_stride < shape.k * sizeof(float) ||
        effective_rhs_stride < minimum_rhs_stride ||
        effective_dst_stride < shape.n * sizeof(float) ||
        ((effective_lhs_stride | effective_rhs_stride |
          effective_dst_stride) &
         (sizeof(float) - 1)) != 0) {
      return Gem5TmmaSubmitResult::kInvalidCsrState;
    }
  }
  if (operation == xopennpux::Operation::kTrmsnorm) {
    const uint32_t flags =
        packet.csr_epoch == 0 ? tensor_flags_ : packet.tensor_flags;
    if ((flags & ~(xopennpux::kTensorFlagNormWeightOffset |
                   xopennpux::kTensorFlagBfloat16Input)) != 0) {
      return Gem5TmmaSubmitResult::kInvalidCsrState;
    }
  }
  if (operation == xopennpux::Operation::kTdequant) {
    const uint32_t config_value = packet.csr_epoch == 0
                                      ? quant_config_
                                      : packet.quant_config;
    const xopennpux::QuantConfig config =
        xopennpux::DecodeQuantConfig(config_value);
    const uint32_t qzeros_address = packet.csr_epoch == 0
                                        ? quant_qzeros_address_
                                        : packet.quant_qzeros_address;
    const uint32_t scales_address = packet.csr_epoch == 0
                                        ? quant_scales_address_
                                        : packet.quant_scales_address;
    const uint32_t g_idx_address = packet.csr_epoch == 0
                                       ? quant_g_idx_address_
                                       : packet.quant_g_idx_address;
    const uint32_t qweight_stride = packet.csr_epoch == 0
                                        ? quant_qweight_stride_
                                        : packet.quant_qweight_stride;
    const uint32_t qzeros_stride = packet.csr_epoch == 0
                                       ? quant_qzeros_stride_
                                       : packet.quant_qzeros_stride;
    const uint32_t scales_stride = packet.csr_epoch == 0
                                       ? quant_scales_stride_
                                       : packet.quant_scales_stride;
    const uint32_t group_range = packet.csr_epoch == 0
                                     ? quant_group_range_
                                     : packet.quant_group_range;
    const uint32_t group_base = group_range & 0xffffu;
    const uint32_t group_count = group_range >> 16;
    const uint32_t scale_bytes =
        config.scale_data_type == xopennpux::DataType::kFp32 ? 4 : 2;
    const uint32_t minimum_qweight_stride = shape.n * 4;
    const uint32_t minimum_qzeros_stride = ((shape.n + 7) / 8) * 4;
    const uint32_t minimum_scales_stride = shape.n * scale_bytes;
    if ((config_value & 0xfe000000u) != 0 || config.group_size == 0 ||
        (config.scale_data_type != xopennpux::DataType::kFp16 &&
         config.scale_data_type != xopennpux::DataType::kBf16 &&
         config.scale_data_type != xopennpux::DataType::kFp32) ||
        data_types.src1 != xopennpux::DataType::kInt4 ||
        data_types.dst != xopennpux::DataType::kFp32 ||
        qzeros_address == 0 || scales_address == 0 || group_count == 0 ||
        group_base >= group_count ||
        (config.has_g_idx && g_idx_address == 0) ||
        qweight_stride < minimum_qweight_stride ||
        qweight_stride % 4 != 0 ||
        qzeros_stride < minimum_qzeros_stride || qzeros_stride % 4 != 0 ||
        scales_stride < minimum_scales_stride ||
        scales_stride % scale_bytes != 0) {
      return Gem5TmmaSubmitResult::kInvalidCsrState;
    }
  }
  const uint32_t scalar_param0 =
      packet.csr_epoch == 0 ? scalar_param0_ : packet.scalar_param0;
  if (operation == xopennpux::Operation::kTrmsnorm &&
      (!(DecodeFloat(scalar_param0) > 0.0f) ||
       !std::isfinite(DecodeFloat(scalar_param0)))) {
    return Gem5TmmaSubmitResult::kInvalidCsrState;
  }
  if (operation == xopennpux::Operation::kTrope &&
      (tensor_shape.features % 2 != 0 || scalar_param0 > 1)) {
    return Gem5TmmaSubmitResult::kInvalidCsrState;
  }
  if (operation == xopennpux::Operation::kTgather && scalar_param0 == 0) {
    return Gem5TmmaSubmitResult::kInvalidCsrState;
  }
  if (operation == xopennpux::Operation::kTtopk &&
      (scalar_param0 == 0 || scalar_param0 > tensor_shape.features)) {
    return Gem5TmmaSubmitResult::kInvalidCsrState;
  }
  if (operation == xopennpux::Operation::kTcausalconv) {
    const uint32_t kernel_width = scalar_param0 & 0xffffu;
    const uint32_t flags = scalar_param0 >> 16;
    const uint32_t state_source = packet.csr_epoch == 0
                                      ? tensor_aux_source_address_
                                      : packet.tensor_aux_source_address;
    const uint32_t state_destination =
        packet.csr_epoch == 0 ? tensor_aux_destination_address_
                              : packet.tensor_aux_destination_address;
    const bool stateful = (flags & 1u) != 0;
    if (kernel_width == 0 || (flags & ~3u) != 0 ||
        (stateful && kernel_width > 1 &&
         (state_source == 0 || state_destination == 0)) ||
        (!stateful && (state_source != 0 || state_destination != 0))) {
      return Gem5TmmaSubmitResult::kInvalidCsrState;
    }
  }
  if (operation == xopennpux::Operation::kTattention) {
    const uint32_t packed_heads = packet.csr_epoch == 0
                                      ? attention_heads_
                                      : packet.attention_heads;
    const uint32_t head_dim_flags = packet.csr_epoch == 0
                                        ? attention_head_dim_flags_
                                        : packet.attention_head_dim_flags;
    const uint32_t kv_length = packet.csr_epoch == 0
                                   ? attention_kv_length_
                                   : packet.attention_kv_length;
    const uint32_t heads = packed_heads & 0xffffu;
    const uint32_t kv_heads = packed_heads >> 16;
    const uint32_t head_dim = head_dim_flags & 0xffffu;
    const uint32_t flags = head_dim_flags >> 16;
    const uint32_t gate_address = packet.csr_epoch == 0
                                      ? tensor_aux_source_address_
                                      : packet.tensor_aux_source_address;
    if (heads == 0 || kv_heads == 0 || head_dim == 0 || kv_length == 0 ||
        (flags & ~1u) != 0 || heads % kv_heads != 0 ||
        (((flags & 1u) != 0) != (gate_address != 0)) ||
        tensor_shape.rows > kv_length ||
        tensor_shape.features != heads * head_dim) {
      return Gem5TmmaSubmitResult::kInvalidCsrState;
    }
  }
  if (operation == xopennpux::Operation::kTrecurrent) {
    const uint32_t heads = packet.csr_epoch == 0
                               ? recurrent_heads_
                               : packet.recurrent_heads;
    const uint32_t dims = packet.csr_epoch == 0
                              ? recurrent_dims_
                              : packet.recurrent_dims;
    const uint32_t key_heads = heads & 0xffffu;
    const uint32_t value_heads = heads >> 16;
    const uint32_t key_dim = dims & 0xffffu;
    const uint32_t value_dim = dims >> 16;
    const uint32_t beta = packet.csr_epoch == 0
                              ? recurrent_beta_address_
                              : packet.recurrent_beta_address;
    const uint32_t state = packet.csr_epoch == 0
                               ? tensor_aux_destination_address_
                               : packet.tensor_aux_destination_address;
    const uint32_t a_log = packet.csr_epoch == 0
                               ? recurrent_a_log_address_
                               : packet.recurrent_a_log_address;
    const uint32_t dt_bias = packet.csr_epoch == 0
                                 ? recurrent_dt_bias_address_
                                 : packet.recurrent_dt_bias_address;
    if (key_heads == 0 || value_heads == 0 || key_dim == 0 ||
        value_dim == 0 || value_heads % key_heads != 0 || beta == 0 ||
        state == 0 || a_log == 0 || dt_bias == 0 ||
        tensor_shape.features != key_heads * key_dim) {
      return Gem5TmmaSubmitResult::kInvalidCsrState;
    }
  }
  if (operation == xopennpux::Operation::kTconv) {
    const uint32_t input_hw = packet.csr_epoch == 0
                                  ? conv_input_hw_
                                  : packet.conv_input_hw;
    const uint32_t output_hw = packet.csr_epoch == 0
                                   ? conv_output_hw_
                                   : packet.conv_output_hw;
    const uint32_t channels_groups = packet.csr_epoch == 0
                                         ? conv_channels_groups_
                                         : packet.conv_channels_groups;
    const uint32_t kernel_hw = packet.csr_epoch == 0
                                   ? conv_kernel_hw_
                                   : packet.conv_kernel_hw;
    const uint32_t stride_hw = packet.csr_epoch == 0
                                   ? conv_stride_hw_
                                   : packet.conv_stride_hw;
    const uint32_t padding_tl = packet.csr_epoch == 0
                                    ? conv_padding_tl_
                                    : packet.conv_padding_tl;
    const uint32_t padding_br = packet.csr_epoch == 0
                                    ? conv_padding_br_
                                    : packet.conv_padding_br;
    const uint32_t dilation_hw = packet.csr_epoch == 0
                                     ? conv_dilation_hw_
                                     : packet.conv_dilation_hw;
    const uint32_t input_h = input_hw & 0xffffu;
    const uint32_t input_w = input_hw >> 16;
    const uint32_t output_h = output_hw & 0xffffu;
    const uint32_t output_w = output_hw >> 16;
    const uint32_t output_channels = channels_groups & 0xffffu;
    const uint32_t groups = channels_groups >> 16;
    const uint32_t kernel_h = kernel_hw & 0xffffu;
    const uint32_t kernel_w = kernel_hw >> 16;
    const uint32_t stride_h = stride_hw & 0xffffu;
    const uint32_t stride_w = stride_hw >> 16;
    const uint32_t dilation_h = dilation_hw & 0xffffu;
    const uint32_t dilation_w = dilation_hw >> 16;
    const uint64_t padded_h = static_cast<uint64_t>(input_h) +
                              (padding_tl & 0xffffu) +
                              (padding_br & 0xffffu);
    const uint64_t padded_w = static_cast<uint64_t>(input_w) +
                              (padding_tl >> 16) + (padding_br >> 16);
    const uint64_t effective_h =
        static_cast<uint64_t>(kernel_h - (kernel_h != 0)) * dilation_h + 1;
    const uint64_t effective_w =
        static_cast<uint64_t>(kernel_w - (kernel_w != 0)) * dilation_w + 1;
    if (input_h == 0 || input_w == 0 || output_h == 0 || output_w == 0 ||
        output_channels == 0 || groups == 0 || kernel_h == 0 ||
        kernel_w == 0 || stride_h == 0 || stride_w == 0 ||
        dilation_h == 0 || dilation_w == 0 ||
        tensor_shape.features % groups != 0 ||
        output_channels % groups != 0 || padded_h < effective_h ||
        padded_w < effective_w ||
        output_h != (padded_h - effective_h) / stride_h + 1 ||
        output_w != (padded_w - effective_w) / stride_w + 1) {
      return Gem5TmmaSubmitResult::kInvalidCsrState;
    }
  }
  return Gem5TmmaSubmitResult::kAccepted;
}

Gem5TmmaSubmitResult Gem5XOpenNpuFunctionalCoprocessor::Submit(
    const Gem5TmmaDispatchPacket& packet) {
  const Gem5TmmaSubmitResult classification = Classify(packet);
  if (classification != Gem5TmmaSubmitResult::kAccepted) {
    return classification;
  }
  if (xopennpux::IsTfence(packet.instruction)) {
    return Gem5TmmaSubmitResult::kAccepted;
  }

  const uint32_t shape_csr = packet.csr_epoch == 0 ? mma_shape_
                                                    : packet.mma_shape;
  const uint32_t data_type_csr = packet.csr_epoch == 0
                                     ? mma_data_type_
                                     : packet.mma_data_type;
  const xopennpux::MmaShape shape = xopennpux::DecodeMmaShape(shape_csr);
  const xopennpux::MmaDataTypes data_types =
      xopennpux::DecodeMmaDataTypes(data_type_csr);

  const size_t tail = (queue_head_ + queue_size_) % kQueueCapacity;
  queue_[tail].dispatch = packet;
  queue_[tail].operation = xopennpux::DecodeOperation(packet.instruction);
  queue_[tail].shape = shape;
  queue_[tail].data_types = data_types;
  const uint32_t tensor_shape_csr =
      packet.csr_epoch == 0 ? tensor_shape_ : packet.tensor_shape;
  const uint32_t tensor_data_type_csr =
      packet.csr_epoch == 0 ? tensor_data_type_ : packet.tensor_data_type;
  queue_[tail].tensor_shape =
      xopennpux::DecodeTensorShape(tensor_shape_csr);
  queue_[tail].tensor_data_types =
      xopennpux::DecodeMmaDataTypes(tensor_data_type_csr);
  queue_[tail].scalar_param0 =
      packet.csr_epoch == 0 ? scalar_param0_ : packet.scalar_param0;
  queue_[tail].csr_epoch = packet.csr_epoch == 0 ? csr_epoch_
                                                  : packet.csr_epoch;
  queue_[tail].quant_qzeros_address = packet.csr_epoch == 0
                                           ? quant_qzeros_address_
                                           : packet.quant_qzeros_address;
  queue_[tail].quant_scales_address = packet.csr_epoch == 0
                                          ? quant_scales_address_
                                          : packet.quant_scales_address;
  queue_[tail].quant_g_idx_address = packet.csr_epoch == 0
                                         ? quant_g_idx_address_
                                         : packet.quant_g_idx_address;
  queue_[tail].quant_config = packet.csr_epoch == 0
                                  ? quant_config_
                                  : packet.quant_config;
  queue_[tail].quant_qweight_stride = packet.csr_epoch == 0
                                           ? quant_qweight_stride_
                                           : packet.quant_qweight_stride;
  queue_[tail].quant_qzeros_stride = packet.csr_epoch == 0
                                          ? quant_qzeros_stride_
                                          : packet.quant_qzeros_stride;
  queue_[tail].quant_scales_stride = packet.csr_epoch == 0
                                          ? quant_scales_stride_
                                          : packet.quant_scales_stride;
  queue_[tail].quant_group_range = packet.csr_epoch == 0
                                        ? quant_group_range_
                                        : packet.quant_group_range;
  queue_[tail].tensor_aux_source_address =
      packet.csr_epoch == 0 ? tensor_aux_source_address_
                            : packet.tensor_aux_source_address;
  queue_[tail].tensor_aux_destination_address =
      packet.csr_epoch == 0 ? tensor_aux_destination_address_
                            : packet.tensor_aux_destination_address;
  queue_[tail].attention_heads =
      packet.csr_epoch == 0 ? attention_heads_ : packet.attention_heads;
  queue_[tail].attention_head_dim_flags =
      packet.csr_epoch == 0 ? attention_head_dim_flags_
                            : packet.attention_head_dim_flags;
  queue_[tail].attention_kv_length = packet.csr_epoch == 0
                                         ? attention_kv_length_
                                         : packet.attention_kv_length;
  queue_[tail].recurrent_heads =
      packet.csr_epoch == 0 ? recurrent_heads_ : packet.recurrent_heads;
  queue_[tail].recurrent_dims =
      packet.csr_epoch == 0 ? recurrent_dims_ : packet.recurrent_dims;
  queue_[tail].recurrent_beta_address =
      packet.csr_epoch == 0 ? recurrent_beta_address_
                            : packet.recurrent_beta_address;
  queue_[tail].recurrent_a_log_address =
      packet.csr_epoch == 0 ? recurrent_a_log_address_
                            : packet.recurrent_a_log_address;
  queue_[tail].recurrent_dt_bias_address =
      packet.csr_epoch == 0 ? recurrent_dt_bias_address_
                            : packet.recurrent_dt_bias_address;
  queue_[tail].conv_input_hw =
      packet.csr_epoch == 0 ? conv_input_hw_ : packet.conv_input_hw;
  queue_[tail].conv_output_hw =
      packet.csr_epoch == 0 ? conv_output_hw_ : packet.conv_output_hw;
  queue_[tail].conv_channels_groups = packet.csr_epoch == 0
                                          ? conv_channels_groups_
                                          : packet.conv_channels_groups;
  queue_[tail].conv_kernel_hw =
      packet.csr_epoch == 0 ? conv_kernel_hw_ : packet.conv_kernel_hw;
  queue_[tail].conv_stride_hw =
      packet.csr_epoch == 0 ? conv_stride_hw_ : packet.conv_stride_hw;
  queue_[tail].conv_padding_tl =
      packet.csr_epoch == 0 ? conv_padding_tl_ : packet.conv_padding_tl;
  queue_[tail].conv_padding_br =
      packet.csr_epoch == 0 ? conv_padding_br_ : packet.conv_padding_br;
  queue_[tail].conv_dilation_hw =
      packet.csr_epoch == 0 ? conv_dilation_hw_ : packet.conv_dilation_hw;
  queue_[tail].conv_bias_address =
      packet.csr_epoch == 0 ? conv_bias_address_ : packet.conv_bias_address;
  queue_[tail].mma_lhs_stride =
      packet.csr_epoch == 0 ? mma_lhs_stride_ : packet.mma_lhs_stride;
  queue_[tail].mma_rhs_stride =
      packet.csr_epoch == 0 ? mma_rhs_stride_ : packet.mma_rhs_stride;
  queue_[tail].mma_dst_stride =
      packet.csr_epoch == 0 ? mma_dst_stride_ : packet.mma_dst_stride;
  queue_[tail].mma_flags =
      packet.csr_epoch == 0 ? mma_flags_ : packet.mma_flags;
  queue_[tail].tensor_flags =
      packet.csr_epoch == 0 ? tensor_flags_ : packet.tensor_flags;
  ++queue_size_;
  return Gem5TmmaSubmitResult::kAccepted;
}

bool Gem5XOpenNpuFunctionalCoprocessor::ExecuteNext(
    std::vector<uint8_t>* memory, uint32_t memory_base,
    Gem5TmmaCompletion* completion) {
  if (memory == nullptr || completion == nullptr || queue_size_ == 0) {
    return false;
  }

  const QueuedOperation command = queue_[queue_head_];
  queue_head_ = (queue_head_ + 1) % kQueueCapacity;
  --queue_size_;

  *completion = {};
  completion->sequence_id = command.dispatch.sequence_id;
  completion->csr_epoch = command.csr_epoch;
  completion->pc = command.dispatch.pc;
  completion->instruction = command.dispatch.instruction;
  completion->hart_id = command.dispatch.hart_id;
  completion->operation = command.operation;
  completion->destination_address = command.dispatch.rd_value;
  const uint32_t conv_input_h = command.conv_input_hw & 0xffffu;
  const uint32_t conv_input_w = command.conv_input_hw >> 16;
  const uint32_t conv_output_h = command.conv_output_hw & 0xffffu;
  const uint32_t conv_output_w = command.conv_output_hw >> 16;
  const uint32_t conv_output_channels =
      command.conv_channels_groups & 0xffffu;
  const uint32_t conv_groups = command.conv_channels_groups >> 16;
  const uint32_t conv_kernel_h = command.conv_kernel_hw & 0xffffu;
  const uint32_t conv_kernel_w = command.conv_kernel_hw >> 16;
  const uint64_t tensor_elements =
      command.operation == xopennpux::Operation::kTmma
          ? static_cast<uint64_t>(command.shape.m) * command.shape.n *
                command.shape.k
          : static_cast<uint64_t>(command.tensor_shape.rows) *
                command.tensor_shape.features;
  if (command.operation == xopennpux::Operation::kTmma) {
    completion->mac_operations = tensor_elements;
    completion->modeled_cycles = completion->mac_operations;
  } else if (command.operation == xopennpux::Operation::kTdequant) {
    completion->element_operations =
        static_cast<uint64_t>(command.shape.k) * command.shape.n;
    completion->modeled_cycles = completion->element_operations;
  } else if (command.operation == xopennpux::Operation::kTrmsnorm) {
    completion->element_operations = tensor_elements * 4;
    completion->modeled_cycles = completion->element_operations;
  } else if (command.operation == xopennpux::Operation::kTsoftmax) {
    completion->element_operations = tensor_elements * 4;
    completion->modeled_cycles = completion->element_operations;
  } else if (command.operation == xopennpux::Operation::kTrope) {
    completion->element_operations = tensor_elements * 3;
    completion->modeled_cycles = completion->element_operations;
  } else if (command.operation == xopennpux::Operation::kTsilu) {
    completion->element_operations = tensor_elements * 3;
    completion->modeled_cycles = completion->element_operations;
  } else if (command.operation == xopennpux::Operation::kTsigmoid) {
    completion->element_operations = tensor_elements * 3;
    completion->modeled_cycles = completion->element_operations;
  } else if (command.operation == xopennpux::Operation::kTgather) {
    completion->element_operations = tensor_elements;
    completion->modeled_cycles = completion->element_operations;
  } else if (command.operation == xopennpux::Operation::kTtopk) {
    completion->element_operations =
        tensor_elements * command.scalar_param0;
    completion->modeled_cycles = completion->element_operations;
  } else if (command.operation == xopennpux::Operation::kTcausalconv) {
    const uint64_t kernel_width = command.scalar_param0 & 0xffffu;
    completion->element_operations = tensor_elements * kernel_width * 2;
    completion->modeled_cycles = completion->element_operations;
  } else if (command.operation == xopennpux::Operation::kTattention) {
    const uint64_t rows = command.tensor_shape.rows;
    const uint64_t kv_length = command.attention_kv_length;
    const uint64_t visible_positions =
        rows * (kv_length - rows + 1) + rows * (rows - 1) / 2;
    const uint64_t heads = command.attention_heads & 0xffffu;
    const uint64_t head_dim = command.attention_head_dim_flags & 0xffffu;
    const uint64_t gate_operations =
        (command.attention_head_dim_flags >> 16) != 0
            ? rows * heads * head_dim * 4
            : 0;
    completion->element_operations =
        visible_positions * heads * head_dim * 4 + gate_operations;
    completion->modeled_cycles = completion->element_operations;
  } else if (command.operation == xopennpux::Operation::kTrecurrent) {
    const uint64_t rows = command.tensor_shape.rows;
    const uint64_t value_heads = command.recurrent_heads >> 16;
    const uint64_t key_dim = command.recurrent_dims & 0xffffu;
    const uint64_t value_dim = command.recurrent_dims >> 16;
    completion->element_operations =
        rows * value_heads *
        (key_dim * 4 + value_dim * key_dim * 6 + value_dim * 3 + 8);
    completion->modeled_cycles = completion->element_operations;
  } else if (command.operation == xopennpux::Operation::kTconv) {
    completion->mac_operations =
        static_cast<uint64_t>(command.tensor_shape.rows) * conv_output_h *
        conv_output_w * conv_output_channels * conv_kernel_h * conv_kernel_w *
        (command.tensor_shape.features / conv_groups);
    completion->modeled_cycles = completion->mac_operations;
  } else {
    completion->element_operations = tensor_elements;
    completion->modeled_cycles = completion->element_operations;
  }

  const xopennpux::MmaDataTypes data_types =
      command.operation == xopennpux::Operation::kTmma ||
              command.operation == xopennpux::Operation::kTdequant
          ? command.data_types
          : command.tensor_data_types;
  if (command.operation != xopennpux::Operation::kTdequant &&
      (data_types.src1 != xopennpux::DataType::kFp32 ||
      data_types.src2 != xopennpux::DataType::kFp32 ||
       data_types.dst != xopennpux::DataType::kFp32)) {
    completion->error = Gem5TmmaExecutionError::kUnsupportedDataType;
    return true;
  }

  const uint64_t lhs_elements =
      command.operation == xopennpux::Operation::kTmma
          ? static_cast<uint64_t>(command.shape.m) * command.shape.k
          : command.operation == xopennpux::Operation::kTconv
              ? static_cast<uint64_t>(command.tensor_shape.rows) *
                    conv_input_h * conv_input_w *
                    command.tensor_shape.features
          : command.operation == xopennpux::Operation::kTrecurrent
              ? static_cast<uint64_t>(command.tensor_shape.rows) *
                    (2 * (command.recurrent_heads & 0xffffu) *
                         (command.recurrent_dims & 0xffffu) +
                     (command.recurrent_heads >> 16) *
                         (command.recurrent_dims >> 16))
          : command.operation == xopennpux::Operation::kTgather
              ? static_cast<uint64_t>(command.scalar_param0) *
                    command.tensor_shape.features
              : tensor_elements;
  const uint64_t rhs_elements =
      command.operation == xopennpux::Operation::kTmma
          ? static_cast<uint64_t>(command.shape.k) * command.shape.n
          : command.operation == xopennpux::Operation::kTconv
              ? static_cast<uint64_t>(conv_output_channels) * conv_kernel_h *
                    conv_kernel_w *
                    (command.tensor_shape.features / conv_groups)
          : command.operation == xopennpux::Operation::kTdequant
              ? 0
              : command.operation == xopennpux::Operation::kTrecurrent
              ? static_cast<uint64_t>(command.tensor_shape.rows) *
                    (command.recurrent_heads >> 16)
              : command.operation == xopennpux::Operation::kTrmsnorm
              ? command.tensor_shape.features
              : command.operation == xopennpux::Operation::kTattention
              ? static_cast<uint64_t>(2) * command.attention_kv_length *
                    (command.attention_heads >> 16) *
                    (command.attention_head_dim_flags & 0xffffu)
              : command.operation == xopennpux::Operation::kTcausalconv
              ? static_cast<uint64_t>(command.tensor_shape.features) *
                    (command.scalar_param0 & 0xffffu)
              : command.operation == xopennpux::Operation::kTrope
                  ? tensor_elements * 2
              : command.operation == xopennpux::Operation::kTgather
                  ? command.tensor_shape.rows
              : command.operation == xopennpux::Operation::kTrowScale
                  ? command.tensor_shape.rows
              : command.operation == xopennpux::Operation::kTsoftmax ||
                        command.operation == xopennpux::Operation::kTsilu ||
                        command.operation == xopennpux::Operation::kTsigmoid ||
                        command.operation == xopennpux::Operation::kTdma ||
                        command.operation == xopennpux::Operation::kTtopk
                    ? 0
                    : tensor_elements;
  const uint64_t dst_elements =
      command.operation == xopennpux::Operation::kTmma
          ? static_cast<uint64_t>(command.shape.m) * command.shape.n
          : command.operation == xopennpux::Operation::kTconv
              ? static_cast<uint64_t>(command.tensor_shape.rows) *
                    conv_output_h * conv_output_w * conv_output_channels
          : command.operation == xopennpux::Operation::kTdequant
              ? static_cast<uint64_t>(command.shape.k) * command.shape.n
          : command.operation == xopennpux::Operation::kTrecurrent
              ? static_cast<uint64_t>(command.tensor_shape.rows) *
                    (command.recurrent_heads >> 16) *
                    (command.recurrent_dims >> 16)
          : command.operation == xopennpux::Operation::kTtopk
              ? static_cast<uint64_t>(command.tensor_shape.rows) *
                    command.scalar_param0 *
                    (command.tensor_aux_destination_address == 0 ? 2 : 1)
          : tensor_elements;
  uint64_t dst_bytes = dst_elements * sizeof(float);
  if (command.operation == xopennpux::Operation::kTmma) {
    const bool transpose_rhs =
        (command.mma_flags & xopennpux::kMmaFlagTransposeRhs) != 0;
    const uint32_t lhs_stride =
        command.mma_lhs_stride == 0
            ? command.shape.k * sizeof(float)
            : command.mma_lhs_stride;
    const uint32_t rhs_stride =
        command.mma_rhs_stride == 0
            ? (transpose_rhs ? command.shape.k : command.shape.n) *
                  sizeof(float)
            : command.mma_rhs_stride;
    const uint32_t dst_stride =
        command.mma_dst_stride == 0
            ? command.shape.n * sizeof(float)
            : command.mma_dst_stride;
    const uint32_t rhs_rows =
        transpose_rhs ? command.shape.n : command.shape.k;
    const uint32_t rhs_row_bytes =
        (transpose_rhs ? command.shape.k : command.shape.n) * sizeof(float);
    if (!StridedMatrixRangeValid(command.dispatch.rs1_value, command.shape.m,
                                 command.shape.k * sizeof(float), lhs_stride,
                                 memory_base, memory->size())) {
      completion->error = Gem5TmmaExecutionError::kAddress;
      completion->faulting_address = command.dispatch.rs1_value;
      return true;
    }
    if (!StridedMatrixRangeValid(command.dispatch.rs2_value, rhs_rows,
                                 rhs_row_bytes, rhs_stride, memory_base,
                                 memory->size())) {
      completion->error = Gem5TmmaExecutionError::kAddress;
      completion->faulting_address = command.dispatch.rs2_value;
      return true;
    }
    if (!StridedMatrixRangeValid(command.dispatch.rd_value, command.shape.m,
                                 command.shape.n * sizeof(float), dst_stride,
                                 memory_base, memory->size())) {
      completion->error = Gem5TmmaExecutionError::kAddress;
      completion->faulting_address = command.dispatch.rd_value;
      return true;
    }
    dst_bytes = static_cast<uint64_t>(command.shape.m - 1) * dst_stride +
                command.shape.n * sizeof(float);
  } else if (command.operation != xopennpux::Operation::kTdequant &&
      !MatrixRangeValid(command.dispatch.rs1_value, lhs_elements, memory_base,
                        memory->size())) {
    completion->error = Gem5TmmaExecutionError::kAddress;
    completion->faulting_address = command.dispatch.rs1_value;
    return true;
  }
  if (command.operation == xopennpux::Operation::kTattention &&
      (command.attention_head_dim_flags >> 16) != 0 &&
      !MatrixRangeValid(command.tensor_aux_source_address, tensor_elements,
                        memory_base, memory->size())) {
    completion->error = Gem5TmmaExecutionError::kAddress;
    completion->faulting_address = command.tensor_aux_source_address;
    return true;
  }
  if (command.operation != xopennpux::Operation::kTmma &&
      rhs_elements != 0 &&
      !MatrixRangeValid(command.dispatch.rs2_value, rhs_elements, memory_base,
                        memory->size())) {
    completion->error = Gem5TmmaExecutionError::kAddress;
    completion->faulting_address = command.dispatch.rs2_value;
    return true;
  }
  if (command.operation != xopennpux::Operation::kTmma &&
      !MatrixRangeValid(command.dispatch.rd_value, dst_elements, memory_base,
                        memory->size())) {
    completion->error = Gem5TmmaExecutionError::kAddress;
    completion->faulting_address = command.dispatch.rd_value;
    return true;
  }
  if (command.operation == xopennpux::Operation::kTcausalconv &&
      ((command.scalar_param0 >> 16) & 1u) != 0 &&
      (command.scalar_param0 & 0xffffu) > 1) {
    const uint64_t state_elements =
        static_cast<uint64_t>((command.scalar_param0 & 0xffffu) - 1) *
        command.tensor_shape.features;
    if (!MatrixRangeValid(command.tensor_aux_source_address, state_elements,
                          memory_base, memory->size()) ||
        !MatrixRangeValid(command.tensor_aux_destination_address,
                          state_elements, memory_base, memory->size())) {
      completion->error = Gem5TmmaExecutionError::kAddress;
      completion->faulting_address = command.tensor_aux_source_address;
      return true;
    }
  }
  if (command.operation == xopennpux::Operation::kTrecurrent) {
    const uint64_t rows = command.tensor_shape.rows;
    const uint64_t value_heads = command.recurrent_heads >> 16;
    const uint64_t key_dim = command.recurrent_dims & 0xffffu;
    const uint64_t value_dim = command.recurrent_dims >> 16;
    const uint64_t gates = rows * value_heads;
    const uint64_t state = value_heads * key_dim * value_dim;
    const struct {
      uint32_t address;
      uint64_t elements;
    } ranges[] = {
        {command.recurrent_beta_address, gates},
        {command.tensor_aux_destination_address, state},
        {command.recurrent_a_log_address, value_heads},
        {command.recurrent_dt_bias_address, value_heads},
    };
    for (const auto& range : ranges) {
      if (!MatrixRangeValid(range.address, range.elements, memory_base,
                            memory->size())) {
        completion->error = Gem5TmmaExecutionError::kAddress;
        completion->faulting_address = range.address;
        return true;
      }
    }
  }
  if (command.operation == xopennpux::Operation::kTconv &&
      command.conv_bias_address != 0 &&
      !MatrixRangeValid(command.conv_bias_address, conv_output_channels,
                        memory_base, memory->size())) {
    completion->error = Gem5TmmaExecutionError::kAddress;
    completion->faulting_address = command.conv_bias_address;
    return true;
  }
  if (command.operation == xopennpux::Operation::kTtopk &&
      command.tensor_aux_destination_address != 0 &&
      !MatrixRangeValid(
          command.tensor_aux_destination_address,
          static_cast<uint64_t>(command.tensor_shape.rows) *
              command.scalar_param0,
          memory_base, memory->size())) {
    completion->error = Gem5TmmaExecutionError::kAddress;
    completion->faulting_address = command.tensor_aux_destination_address;
    return true;
  }

  const size_t lhs_base = command.dispatch.rs1_value - memory_base;
  const size_t rhs_base = command.dispatch.rs2_value - memory_base;
  const size_t dst_base = command.dispatch.rd_value - memory_base;
  if (command.operation == xopennpux::Operation::kTdequant) {
    const xopennpux::QuantConfig config =
        xopennpux::DecodeQuantConfig(command.quant_config);
    const uint32_t group_base = command.quant_group_range & 0xffffu;
    const uint32_t group_count = command.quant_group_range >> 16;
    const uint32_t packed_k_rows = (command.shape.k + 7) / 8;
    const uint32_t zero_row_bytes = ((command.shape.n + 7) / 8) * 4;
    const uint32_t scale_element_bytes =
        config.scale_data_type == xopennpux::DataType::kFp32 ? 4 : 2;
    const uint64_t qweight_span =
        static_cast<uint64_t>(packed_k_rows - 1) *
            command.quant_qweight_stride + command.shape.n * 4;
    const uint64_t qzeros_span = static_cast<uint64_t>(group_count - 1) *
                                     command.quant_qzeros_stride +
                                 zero_row_bytes;
    const uint64_t scales_span = static_cast<uint64_t>(group_count - 1) *
                                     command.quant_scales_stride +
                                 command.shape.n * scale_element_bytes;
    if (!ByteRangeValid(command.dispatch.rs1_value, qweight_span, 4,
                        memory_base, memory->size()) ||
        !ByteRangeValid(command.quant_qzeros_address, qzeros_span, 4,
                        memory_base, memory->size()) ||
        !ByteRangeValid(command.quant_scales_address, scales_span,
                        scale_element_bytes, memory_base, memory->size()) ||
        (config.has_g_idx &&
         !ByteRangeValid(command.quant_g_idx_address,
                         static_cast<uint64_t>(command.shape.k) * 4, 4,
                         memory_base, memory->size()))) {
      completion->error = Gem5TmmaExecutionError::kAddress;
      return true;
    }
    const size_t qweight_base = command.dispatch.rs1_value - memory_base;
    const size_t qzeros_base = command.quant_qzeros_address - memory_base;
    const size_t scales_base = command.quant_scales_address - memory_base;
    const size_t g_idx_base = command.quant_g_idx_address - memory_base;
    for (uint32_t k = 0; k < command.shape.k; ++k) {
      uint32_t group = group_base + k / config.group_size;
      if (config.has_g_idx) {
        group = LoadUint32(*memory, g_idx_base + k * 4);
      }
      if (group >= group_count) {
        completion->error = Gem5TmmaExecutionError::kAddress;
        completion->faulting_address = config.has_g_idx
                                           ? command.quant_g_idx_address + k * 4
                                           : command.dispatch.rs1_value;
        return true;
      }
      for (uint32_t column = 0; column < command.shape.n; ++column) {
        const uint32_t packed_weight = LoadUint32(
            *memory, qweight_base + (k / 8) * command.quant_qweight_stride +
                         column * 4);
        const uint32_t quantized = (packed_weight >> (4 * (k % 8))) & 0xf;
        const uint32_t packed_zero = LoadUint32(
            *memory, qzeros_base + group * command.quant_qzeros_stride +
                         (column / 8) * 4);
        const uint32_t zero =
            ((packed_zero >> (4 * (column % 8))) & 0xf) + config.zero_bias;
        const float scale = LoadScale(
            *memory, scales_base + group * command.quant_scales_stride +
                         column * scale_element_bytes,
            config.scale_data_type);
        if (!std::isfinite(scale)) {
          completion->error = Gem5TmmaExecutionError::kInvalidData;
          completion->faulting_address =
              command.quant_scales_address +
              group * command.quant_scales_stride +
              column * scale_element_bytes;
          return true;
        }
        StoreFloat(memory, dst_base +
                               (static_cast<size_t>(k) * command.shape.n +
                                column) *
                                   sizeof(float),
                   (static_cast<int32_t>(quantized) -
                    static_cast<int32_t>(zero)) * scale);
      }
    }
  } else if (command.operation == xopennpux::Operation::kTmma) {
    const bool transpose_rhs =
        (command.mma_flags & xopennpux::kMmaFlagTransposeRhs) != 0;
    const bool accumulate =
        (command.mma_flags & xopennpux::kMmaFlagAccumulate) != 0;
    const uint32_t lhs_stride =
        command.mma_lhs_stride == 0
            ? command.shape.k * sizeof(float)
            : command.mma_lhs_stride;
    const uint32_t rhs_stride =
        command.mma_rhs_stride == 0
            ? (transpose_rhs ? command.shape.k : command.shape.n) *
                  sizeof(float)
            : command.mma_rhs_stride;
    const uint32_t dst_stride =
        command.mma_dst_stride == 0
            ? command.shape.n * sizeof(float)
            : command.mma_dst_stride;
    for (uint32_t row = 0; row < command.shape.m; ++row) {
      for (uint32_t column = 0; column < command.shape.n; ++column) {
        const size_t dst_offset =
            dst_base + static_cast<size_t>(row) * dst_stride +
            column * sizeof(float);
        float accumulator = accumulate ? LoadFloat(*memory, dst_offset) : 0.0f;
        for (uint32_t inner = 0; inner < command.shape.k; ++inner) {
          const size_t lhs_offset =
              lhs_base + static_cast<size_t>(row) * lhs_stride +
              inner * sizeof(float);
          const size_t rhs_offset = transpose_rhs
                                        ? rhs_base +
                                              static_cast<size_t>(column) *
                                                  rhs_stride +
                                              inner * sizeof(float)
                                        : rhs_base +
                                              static_cast<size_t>(inner) *
                                                  rhs_stride +
                                              column * sizeof(float);
          accumulator += LoadFloat(*memory, lhs_offset) *
                         LoadFloat(*memory, rhs_offset);
        }
        StoreFloat(memory, dst_offset, accumulator);
      }
    }
  } else if (command.operation == xopennpux::Operation::kTconv) {
    const uint32_t stride_h = command.conv_stride_hw & 0xffffu;
    const uint32_t stride_w = command.conv_stride_hw >> 16;
    const uint32_t pad_top = command.conv_padding_tl & 0xffffu;
    const uint32_t pad_left = command.conv_padding_tl >> 16;
    const uint32_t dilation_h = command.conv_dilation_hw & 0xffffu;
    const uint32_t dilation_w = command.conv_dilation_hw >> 16;
    const uint32_t input_channels = command.tensor_shape.features;
    const uint32_t channels_per_group = input_channels / conv_groups;
    const uint32_t outputs_per_group = conv_output_channels / conv_groups;
    const size_t bias_base = command.conv_bias_address == 0
                                 ? 0
                                 : command.conv_bias_address - memory_base;
    for (uint32_t batch = 0; batch < command.tensor_shape.rows; ++batch) {
      for (uint32_t output_y = 0; output_y < conv_output_h; ++output_y) {
        for (uint32_t output_x = 0; output_x < conv_output_w; ++output_x) {
          for (uint32_t output_channel = 0;
               output_channel < conv_output_channels; ++output_channel) {
            float accumulator = command.conv_bias_address == 0
                                    ? 0.0f
                                    : LoadFloat(
                                          *memory,
                                          bias_base + output_channel *
                                                          sizeof(float));
            const uint32_t group = output_channel / outputs_per_group;
            for (uint32_t kernel_y = 0; kernel_y < conv_kernel_h;
                 ++kernel_y) {
              const int64_t input_y =
                  static_cast<int64_t>(output_y) * stride_h +
                  static_cast<int64_t>(kernel_y) * dilation_h - pad_top;
              if (input_y < 0 || input_y >= conv_input_h) continue;
              for (uint32_t kernel_x = 0; kernel_x < conv_kernel_w;
                   ++kernel_x) {
                const int64_t input_x =
                    static_cast<int64_t>(output_x) * stride_w +
                    static_cast<int64_t>(kernel_x) * dilation_w - pad_left;
                if (input_x < 0 || input_x >= conv_input_w) continue;
                for (uint32_t local_channel = 0;
                     local_channel < channels_per_group; ++local_channel) {
                  const uint32_t input_channel =
                      group * channels_per_group + local_channel;
                  const size_t input_index =
                      (((static_cast<size_t>(batch) * conv_input_h + input_y) *
                            conv_input_w +
                        input_x) *
                           input_channels +
                       input_channel);
                  const size_t weight_index =
                      (((static_cast<size_t>(output_channel) * conv_kernel_h +
                         kernel_y) *
                            conv_kernel_w +
                        kernel_x) *
                           channels_per_group +
                       local_channel);
                  accumulator +=
                      LoadFloat(*memory,
                                lhs_base + input_index * sizeof(float)) *
                      LoadFloat(*memory,
                                rhs_base + weight_index * sizeof(float));
                }
              }
            }
            const size_t output_index =
                (((static_cast<size_t>(batch) * conv_output_h + output_y) *
                      conv_output_w +
                  output_x) *
                     conv_output_channels +
                 output_channel);
            StoreFloat(memory, dst_base + output_index * sizeof(float),
                       accumulator);
          }
        }
      }
    }
  } else if (command.operation == xopennpux::Operation::kTadd) {
    for (uint64_t index = 0; index < tensor_elements; ++index) {
      const size_t offset = static_cast<size_t>(index) * sizeof(float);
      StoreFloat(memory, dst_base + offset,
                 LoadFloat(*memory, lhs_base + offset) +
                     LoadFloat(*memory, rhs_base + offset));
    }
  } else if (command.operation == xopennpux::Operation::kTmul) {
    for (uint64_t index = 0; index < tensor_elements; ++index) {
      const size_t offset = static_cast<size_t>(index) * sizeof(float);
      StoreFloat(memory, dst_base + offset,
                 LoadFloat(*memory, lhs_base + offset) *
                     LoadFloat(*memory, rhs_base + offset));
    }
  } else if (command.operation == xopennpux::Operation::kTrowScale) {
    for (uint32_t row = 0; row < command.tensor_shape.rows; ++row) {
      const float scale =
          LoadFloat(*memory, rhs_base + row * sizeof(float));
      const size_t row_base =
          static_cast<size_t>(row) * command.tensor_shape.features;
      for (uint32_t feature = 0; feature < command.tensor_shape.features;
           ++feature) {
        const size_t offset = (row_base + feature) * sizeof(float);
        StoreFloat(memory, dst_base + offset,
                   LoadFloat(*memory, lhs_base + offset) * scale);
      }
    }
  } else if (command.operation == xopennpux::Operation::kTrmsnorm) {
    const float epsilon = DecodeFloat(command.scalar_param0);
    const bool weight_offset =
        (command.tensor_flags & xopennpux::kTensorFlagNormWeightOffset) != 0;
    const bool bfloat16_input =
        (command.tensor_flags & xopennpux::kTensorFlagBfloat16Input) != 0;
    for (uint32_t row = 0; row < command.tensor_shape.rows; ++row) {
      float sum_squares = 0.0f;
      const size_t row_base =
          static_cast<size_t>(row) * command.tensor_shape.features;
      for (uint32_t feature = 0; feature < command.tensor_shape.features;
           ++feature) {
        const float source =
            LoadFloat(*memory, lhs_base +
                                   (row_base + feature) * sizeof(float));
        const float value =
            bfloat16_input ? RoundBfloat16(source) : source;
        sum_squares += value * value;
      }
      const float inverse_rms =
          1.0f / std::sqrt(sum_squares / command.tensor_shape.features +
                           epsilon);
      for (uint32_t feature = 0; feature < command.tensor_shape.features;
           ++feature) {
        const float source =
            LoadFloat(*memory, lhs_base +
                                   (row_base + feature) * sizeof(float));
        const float value =
            bfloat16_input ? RoundBfloat16(source) : source;
        const float weight =
            LoadFloat(*memory, rhs_base + feature * sizeof(float)) +
            (weight_offset ? 1.0f : 0.0f);
        StoreFloat(memory,
                   dst_base + (row_base + feature) * sizeof(float),
                   value * inverse_rms * weight);
      }
    }
  } else if (command.operation == xopennpux::Operation::kTsoftmax) {
    for (uint32_t row = 0; row < command.tensor_shape.rows; ++row) {
      const size_t row_base =
          static_cast<size_t>(row) * command.tensor_shape.features;
      float maximum = LoadFloat(*memory, lhs_base + row_base * sizeof(float));
      for (uint32_t feature = 1; feature < command.tensor_shape.features;
           ++feature) {
        maximum = std::max(
            maximum,
            LoadFloat(*memory,
                      lhs_base + (row_base + feature) * sizeof(float)));
      }
      float sum = 0.0f;
      for (uint32_t feature = 0; feature < command.tensor_shape.features;
           ++feature) {
        const float value = std::exp(
            LoadFloat(*memory,
                      lhs_base + (row_base + feature) * sizeof(float)) -
            maximum);
        StoreFloat(memory,
                   dst_base + (row_base + feature) * sizeof(float), value);
        sum += value;
      }
      for (uint32_t feature = 0; feature < command.tensor_shape.features;
           ++feature) {
        const size_t offset = (row_base + feature) * sizeof(float);
        StoreFloat(memory, dst_base + offset,
                   LoadFloat(*memory, dst_base + offset) / sum);
      }
    }
  } else if (command.operation == xopennpux::Operation::kTrope) {
    const bool half_split = command.scalar_param0 == 1;
    const size_t sin_base = rhs_base + tensor_elements * sizeof(float);
    std::vector<float> row_input(command.tensor_shape.features);
    for (uint32_t row = 0; row < command.tensor_shape.rows; ++row) {
      const size_t row_base =
          static_cast<size_t>(row) * command.tensor_shape.features;
      for (uint32_t feature = 0; feature < command.tensor_shape.features;
           ++feature) {
        row_input[feature] =
            LoadFloat(*memory,
                      lhs_base + (row_base + feature) * sizeof(float));
      }
      const uint32_t half = command.tensor_shape.features / 2;
      for (uint32_t feature = 0; feature < command.tensor_shape.features;
           ++feature) {
        uint32_t rotated_feature = 0;
        float sign = 1.0f;
        if (half_split) {
          rotated_feature = feature < half ? feature + half : feature - half;
          sign = feature < half ? -1.0f : 1.0f;
        } else {
          rotated_feature = feature ^ 1u;
          sign = (feature & 1u) == 0 ? -1.0f : 1.0f;
        }
        const size_t offset = (row_base + feature) * sizeof(float);
        const float cosine = LoadFloat(*memory, rhs_base + offset);
        const float sine = LoadFloat(*memory, sin_base + offset);
        StoreFloat(memory, dst_base + offset,
                   row_input[feature] * cosine +
                       sign * row_input[rotated_feature] * sine);
      }
    }
  } else if (command.operation == xopennpux::Operation::kTsilu) {
    for (uint64_t index = 0; index < tensor_elements; ++index) {
      const size_t offset = static_cast<size_t>(index) * sizeof(float);
      const float value = LoadFloat(*memory, lhs_base + offset);
      StoreFloat(memory, dst_base + offset,
                 value / (1.0f + std::exp(-value)));
    }
  } else if (command.operation == xopennpux::Operation::kTsigmoid) {
    for (uint64_t index = 0; index < tensor_elements; ++index) {
      const size_t offset = static_cast<size_t>(index) * sizeof(float);
      const float value = LoadFloat(*memory, lhs_base + offset);
      StoreFloat(memory, dst_base + offset, 1.0f / (1.0f + std::exp(-value)));
    }
  } else if (command.operation == xopennpux::Operation::kTgather) {
    for (uint32_t row = 0; row < command.tensor_shape.rows; ++row) {
      const uint32_t source_row =
          LoadUint32(*memory, rhs_base + row * sizeof(uint32_t));
      if (source_row >= command.scalar_param0) {
        completion->error = Gem5TmmaExecutionError::kAddress;
        completion->faulting_address = command.dispatch.rs2_value +
                                       row * sizeof(uint32_t);
        return true;
      }
    }
    for (uint32_t row = 0; row < command.tensor_shape.rows; ++row) {
      const uint32_t source_row =
          LoadUint32(*memory, rhs_base + row * sizeof(uint32_t));
      for (uint32_t feature = 0; feature < command.tensor_shape.features;
           ++feature) {
        const size_t source_offset =
            (static_cast<size_t>(source_row) * command.tensor_shape.features +
             feature) *
            sizeof(float);
        const size_t destination_offset =
            (static_cast<size_t>(row) * command.tensor_shape.features +
             feature) *
            sizeof(float);
        StoreFloat(memory, dst_base + destination_offset,
                   LoadFloat(*memory, lhs_base + source_offset));
      }
    }
  } else if (command.operation == xopennpux::Operation::kTdma) {
    std::memmove(memory->data() + dst_base, memory->data() + lhs_base,
                 static_cast<size_t>(tensor_elements) * sizeof(float));
  } else if (command.operation == xopennpux::Operation::kTcausalconv) {
    const uint32_t kernel_width = command.scalar_param0 & 0xffffu;
    const uint32_t flags = command.scalar_param0 >> 16;
    const bool stateful = (flags & 1u) != 0 && kernel_width > 1;
    const bool silu = (flags & 2u) != 0;
    std::vector<float> history;
    if (stateful) {
      const size_t state_elements =
          static_cast<size_t>(kernel_width - 1) *
          command.tensor_shape.features;
      const size_t state_base =
          command.tensor_aux_source_address - memory_base;
      history.resize(state_elements);
      for (size_t index = 0; index < state_elements; ++index) {
        history[index] =
            LoadFloat(*memory, state_base + index * sizeof(float));
      }
    }
    for (uint32_t row = 0; row < command.tensor_shape.rows; ++row) {
      for (uint32_t feature = 0; feature < command.tensor_shape.features;
           ++feature) {
        float sum = LoadFloat(
                        *memory,
                        lhs_base +
                            (static_cast<size_t>(row) *
                                 command.tensor_shape.features +
                             feature) *
                                sizeof(float)) *
                    LoadFloat(*memory,
                              rhs_base +
                                  (static_cast<size_t>(feature) * kernel_width +
                                   kernel_width - 1) *
                                      sizeof(float));
        for (uint32_t tap = 1; tap < kernel_width; ++tap) {
          float sample = 0.0f;
          if (tap <= row) {
            sample = LoadFloat(
                *memory,
                lhs_base +
                    (static_cast<size_t>(row - tap) *
                         command.tensor_shape.features +
                     feature) *
                        sizeof(float));
          } else if (stateful) {
            const size_t history_row = kernel_width - 1 + row - tap;
            sample = history[history_row * command.tensor_shape.features +
                             feature];
          }
          sum += sample *
                 LoadFloat(*memory,
                           rhs_base +
                               (static_cast<size_t>(feature) * kernel_width +
                                kernel_width - 1 - tap) *
                                   sizeof(float));
        }
        if (silu) {
          sum /= 1.0f + std::exp(-sum);
        }
        StoreFloat(memory,
                   dst_base +
                       (static_cast<size_t>(row) *
                            command.tensor_shape.features +
                        feature) *
                           sizeof(float),
                   sum);
      }
    }
    if (stateful) {
      const size_t state_rows = kernel_width - 1;
      const size_t next_base =
          command.tensor_aux_destination_address - memory_base;
      for (size_t state_row = 0; state_row < state_rows; ++state_row) {
        const int64_t input_row =
            static_cast<int64_t>(command.tensor_shape.rows) - state_rows +
            state_row;
        for (uint32_t feature = 0; feature < command.tensor_shape.features;
             ++feature) {
          float value = 0.0f;
          if (input_row >= 0) {
            value = LoadFloat(
                *memory,
                lhs_base +
                    (static_cast<size_t>(input_row) *
                         command.tensor_shape.features +
                     feature) *
                        sizeof(float));
          } else {
            value = history[(state_rows + input_row) *
                                command.tensor_shape.features +
                            feature];
          }
          StoreFloat(memory,
                     next_base +
                         (state_row * command.tensor_shape.features + feature) *
                             sizeof(float),
                     value);
        }
      }
    }
  } else if (command.operation == xopennpux::Operation::kTattention) {
    const uint32_t rows = command.tensor_shape.rows;
    const uint32_t heads = command.attention_heads & 0xffffu;
    const uint32_t kv_heads = command.attention_heads >> 16;
    const uint32_t head_dim = command.attention_head_dim_flags & 0xffffu;
    const uint32_t kv_length = command.attention_kv_length;
    const uint32_t query_start = kv_length - rows;
    const uint32_t heads_per_kv_head = heads / kv_heads;
    const uint64_t kv_plane_elements =
        static_cast<uint64_t>(kv_length) * kv_heads * head_dim;
    const size_t values_base =
        rhs_base + static_cast<size_t>(kv_plane_elements) * sizeof(float);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> scores(kv_length);
    for (uint32_t row = 0; row < rows; ++row) {
      const uint32_t visible = query_start + row + 1;
      for (uint32_t head = 0; head < heads; ++head) {
        const uint32_t kv_head = head / heads_per_kv_head;
        const size_t query_index =
            (static_cast<size_t>(row) * heads + head) * head_dim;
        float maximum = -INFINITY;
        for (uint32_t position = 0; position < visible; ++position) {
          const size_t key_index =
              (static_cast<size_t>(position) * kv_heads + kv_head) * head_dim;
          float dot = 0.0f;
          for (uint32_t lane = 0; lane < head_dim; ++lane) {
            dot += LoadFloat(*memory, lhs_base + (query_index + lane) * 4) *
                   LoadFloat(*memory, rhs_base + (key_index + lane) * 4);
          }
          scores[position] = dot * scale;
          maximum = std::max(maximum, scores[position]);
        }
        float denominator = 0.0f;
        for (uint32_t position = 0; position < visible; ++position) {
          scores[position] = std::exp(scores[position] - maximum);
          denominator += scores[position];
        }
        for (uint32_t lane = 0; lane < head_dim; ++lane) {
          float sum = 0.0f;
          for (uint32_t position = 0; position < visible; ++position) {
            const size_t value_index =
                (static_cast<size_t>(position) * kv_heads + kv_head) *
                    head_dim +
                lane;
            sum += (scores[position] / denominator) *
                   LoadFloat(*memory, values_base + value_index * 4);
          }
          if ((command.attention_head_dim_flags >> 16) != 0) {
            const float gate = LoadFloat(
                *memory, static_cast<size_t>(command.tensor_aux_source_address -
                                             memory_base) +
                             (query_index + lane) * sizeof(float));
            sum *= 1.0f / (1.0f + std::exp(-gate));
          }
          StoreFloat(memory, dst_base + (query_index + lane) * 4, sum);
        }
      }
    }
  } else if (command.operation == xopennpux::Operation::kTrecurrent) {
    const uint32_t rows = command.tensor_shape.rows;
    const uint32_t key_heads = command.recurrent_heads & 0xffffu;
    const uint32_t value_heads = command.recurrent_heads >> 16;
    const uint32_t key_dim = command.recurrent_dims & 0xffffu;
    const uint32_t value_dim = command.recurrent_dims >> 16;
    const uint32_t repeat = value_heads / key_heads;
    const size_t key_features = static_cast<size_t>(key_heads) * key_dim;
    const size_t value_features = static_cast<size_t>(value_heads) * value_dim;
    const size_t qkv_features = key_features * 2 + value_features;
    const size_t state_per_head = static_cast<size_t>(key_dim) * value_dim;
    const size_t beta_base = command.recurrent_beta_address - memory_base;
    const size_t state_base =
        command.tensor_aux_destination_address - memory_base;
    const size_t a_log_base = command.recurrent_a_log_address - memory_base;
    const size_t dt_bias_base =
        command.recurrent_dt_bias_address - memory_base;
    const float query_scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
    std::vector<float> normalized_q(key_dim);
    std::vector<float> normalized_k(key_dim);
    std::vector<float> delta(value_dim);
    for (uint32_t row = 0; row < rows; ++row) {
      const size_t row_base = lhs_base + row * qkv_features * sizeof(float);
      const size_t key_base = row_base + key_features * sizeof(float);
      const size_t value_base = key_base + key_features * sizeof(float);
      for (uint32_t value_head = 0; value_head < value_heads; ++value_head) {
        const uint32_t key_head = value_head / repeat;
        double q_sum = 0.0;
        double k_sum = 0.0;
        for (uint32_t index = 0; index < key_dim; ++index) {
          const float q = LoadFloat(
              *memory, row_base +
                           (static_cast<size_t>(key_head) * key_dim + index) *
                               sizeof(float));
          const float k = LoadFloat(
              *memory, key_base +
                           (static_cast<size_t>(key_head) * key_dim + index) *
                               sizeof(float));
          q_sum += static_cast<double>(q) * q;
          k_sum += static_cast<double>(k) * k;
        }
        const float q_norm =
            1.0f / std::sqrt(static_cast<float>(q_sum) + 1e-6f);
        const float k_norm =
            1.0f / std::sqrt(static_cast<float>(k_sum) + 1e-6f);
        for (uint32_t index = 0; index < key_dim; ++index) {
          normalized_q[index] =
              LoadFloat(*memory,
                        row_base +
                            (static_cast<size_t>(key_head) * key_dim + index) *
                                sizeof(float)) *
              q_norm * query_scale;
          normalized_k[index] =
              LoadFloat(*memory,
                        key_base +
                            (static_cast<size_t>(key_head) * key_dim + index) *
                                sizeof(float)) *
              k_norm;
        }
        const size_t gate_index =
            static_cast<size_t>(row) * value_heads + value_head;
        const float beta = LoadFloat(
            *memory, beta_base + gate_index * sizeof(float));
        const float beta_value = 1.0f / (1.0f + std::exp(-beta));
        const float alpha_value =
            LoadFloat(*memory, rhs_base + gate_index * sizeof(float)) +
            LoadFloat(*memory,
                      dt_bias_base + value_head * sizeof(float));
        const float softplus = alpha_value > 20.0f
                                   ? alpha_value
                                   : std::log1p(std::exp(alpha_value));
        const float decay = std::exp(
            -std::exp(LoadFloat(*memory,
                                a_log_base + value_head * sizeof(float))) *
            softplus);
        const size_t head_state =
            state_base + value_head * state_per_head * sizeof(float);
        for (uint32_t value_index = 0; value_index < value_dim;
             ++value_index) {
          float memory_value = 0.0f;
          for (uint32_t key_index = 0; key_index < key_dim; ++key_index) {
            const size_t cell =
                head_state +
                (static_cast<size_t>(key_index) * value_dim + value_index) *
                    sizeof(float);
            const float decayed = LoadFloat(*memory, cell) * decay;
            StoreFloat(memory, cell, decayed);
            memory_value += decayed * normalized_k[key_index];
          }
          const float value = LoadFloat(
              *memory, value_base +
                           (static_cast<size_t>(value_head) * value_dim +
                            value_index) *
                               sizeof(float));
          delta[value_index] = (value - memory_value) * beta_value;
        }
        for (uint32_t key_index = 0; key_index < key_dim; ++key_index) {
          for (uint32_t value_index = 0; value_index < value_dim;
               ++value_index) {
            const size_t cell =
                head_state +
                (static_cast<size_t>(key_index) * value_dim + value_index) *
                    sizeof(float);
            StoreFloat(memory, cell,
                       LoadFloat(*memory, cell) +
                           normalized_k[key_index] * delta[value_index]);
          }
        }
        for (uint32_t value_index = 0; value_index < value_dim;
             ++value_index) {
          float value = 0.0f;
          for (uint32_t key_index = 0; key_index < key_dim; ++key_index) {
            value += LoadFloat(
                         *memory,
                         head_state +
                             (static_cast<size_t>(key_index) * value_dim +
                              value_index) *
                                 sizeof(float)) *
                     normalized_q[key_index];
          }
          StoreFloat(memory,
                     dst_base +
                         (static_cast<size_t>(row) * value_features +
                          static_cast<size_t>(value_head) * value_dim +
                          value_index) *
                             sizeof(float),
                     value);
        }
      }
    }
  } else if (command.operation == xopennpux::Operation::kTtopk) {
    const uint32_t k = command.scalar_param0;
    const size_t value_count =
        static_cast<size_t>(command.tensor_shape.rows) * k;
    const size_t indices_base = command.tensor_aux_destination_address == 0
                                    ? dst_base + value_count * sizeof(float)
                                    : command.tensor_aux_destination_address -
                                          memory_base;
    std::vector<std::pair<float, uint32_t>> candidates(
        command.tensor_shape.features);
    for (uint32_t row = 0; row < command.tensor_shape.rows; ++row) {
      const size_t row_base =
          static_cast<size_t>(row) * command.tensor_shape.features;
      for (uint32_t feature = 0; feature < command.tensor_shape.features;
           ++feature) {
        candidates[feature] = {
            LoadFloat(*memory,
                      lhs_base + (row_base + feature) * sizeof(float)),
            feature};
      }
      std::partial_sort(
          candidates.begin(), candidates.begin() + k, candidates.end(),
          [](const auto& lhs, const auto& rhs) {
            const bool lhs_nan = std::isnan(lhs.first);
            const bool rhs_nan = std::isnan(rhs.first);
            if (lhs_nan != rhs_nan) {
              return !lhs_nan;
            }
            return lhs.first > rhs.first ||
                   ((lhs.first == rhs.first || (lhs_nan && rhs_nan)) &&
                    lhs.second < rhs.second);
          });
      for (uint32_t rank = 0; rank < k; ++rank) {
        const size_t output = static_cast<size_t>(row) * k + rank;
        StoreFloat(memory, dst_base + output * sizeof(float),
                   candidates[rank].first);
        StoreUint32(memory,
                    indices_base + output * sizeof(uint32_t),
                    candidates[rank].second);
      }
    }
  } else {
    completion->error = Gem5TmmaExecutionError::kUnsupportedDataType;
    return true;
  }
  completion->destination_bytes = static_cast<uint32_t>(dst_bytes);
  completion->destination_checksum =
      Fnv1a(memory->data() + dst_base, completion->destination_bytes);
  const size_t words = std::min<size_t>(
      completion->destination_words.size(), dst_elements);
  for (size_t index = 0; index < words; ++index) {
    std::memcpy(&completion->destination_words[index],
                memory->data() + dst_base + index * sizeof(uint32_t),
                sizeof(uint32_t));
  }
  return true;
}
