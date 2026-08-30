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
      operation != xopennpux::Operation::kTrmsnorm &&
      operation != xopennpux::Operation::kTsoftmax &&
      operation != xopennpux::Operation::kTrope &&
      operation != xopennpux::Operation::kTsilu &&
      operation != xopennpux::Operation::kTgather &&
      operation != xopennpux::Operation::kTdequant &&
      operation != xopennpux::Operation::kTdma &&
      operation != xopennpux::Operation::kTcausalconv &&
      operation != xopennpux::Operation::kTattention &&
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
    if (heads == 0 || kv_heads == 0 || head_dim == 0 || kv_length == 0 ||
        flags != 0 || heads % kv_heads != 0 ||
        tensor_shape.rows > kv_length ||
        tensor_shape.features != heads * head_dim) {
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
    completion->element_operations = visible_positions * heads * head_dim * 4;
    completion->modeled_cycles = completion->element_operations;
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
          : command.operation == xopennpux::Operation::kTgather
              ? static_cast<uint64_t>(command.scalar_param0) *
                    command.tensor_shape.features
              : tensor_elements;
  const uint64_t rhs_elements =
      command.operation == xopennpux::Operation::kTmma
          ? static_cast<uint64_t>(command.shape.k) * command.shape.n
          : command.operation == xopennpux::Operation::kTdequant
              ? 0
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
              : command.operation == xopennpux::Operation::kTsoftmax ||
                        command.operation == xopennpux::Operation::kTsilu ||
                        command.operation == xopennpux::Operation::kTdma ||
                        command.operation == xopennpux::Operation::kTtopk
                    ? 0
                    : tensor_elements;
  const uint64_t dst_elements =
      command.operation == xopennpux::Operation::kTmma
          ? static_cast<uint64_t>(command.shape.m) * command.shape.n
          : command.operation == xopennpux::Operation::kTdequant
              ? static_cast<uint64_t>(command.shape.k) * command.shape.n
          : command.operation == xopennpux::Operation::kTtopk
              ? static_cast<uint64_t>(command.tensor_shape.rows) *
                    command.scalar_param0 * 2
          : tensor_elements;
  const uint64_t dst_bytes = dst_elements * sizeof(float);
  if (command.operation != xopennpux::Operation::kTdequant &&
      !MatrixRangeValid(command.dispatch.rs1_value, lhs_elements, memory_base,
                        memory->size())) {
    completion->error = Gem5TmmaExecutionError::kAddress;
    completion->faulting_address = command.dispatch.rs1_value;
    return true;
  }
  if (rhs_elements != 0 &&
      !MatrixRangeValid(command.dispatch.rs2_value, rhs_elements, memory_base,
                        memory->size())) {
    completion->error = Gem5TmmaExecutionError::kAddress;
    completion->faulting_address = command.dispatch.rs2_value;
    return true;
  }
  if (!MatrixRangeValid(command.dispatch.rd_value, dst_elements, memory_base,
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
    for (uint32_t row = 0; row < command.shape.m; ++row) {
      for (uint32_t column = 0; column < command.shape.n; ++column) {
        float accumulator = 0.0f;
        for (uint32_t inner = 0; inner < command.shape.k; ++inner) {
          const size_t lhs_offset =
              lhs_base + (row * command.shape.k + inner) * sizeof(float);
          const size_t rhs_offset =
              rhs_base + (inner * command.shape.n + column) * sizeof(float);
          accumulator += LoadFloat(*memory, lhs_offset) *
                         LoadFloat(*memory, rhs_offset);
        }
        const size_t dst_offset =
            dst_base + (row * command.shape.n + column) * sizeof(float);
        StoreFloat(memory, dst_offset, accumulator);
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
  } else if (command.operation == xopennpux::Operation::kTrmsnorm) {
    const float epsilon = DecodeFloat(command.scalar_param0);
    for (uint32_t row = 0; row < command.tensor_shape.rows; ++row) {
      float sum_squares = 0.0f;
      const size_t row_base =
          static_cast<size_t>(row) * command.tensor_shape.features;
      for (uint32_t feature = 0; feature < command.tensor_shape.features;
           ++feature) {
        const float value =
            LoadFloat(*memory, lhs_base +
                                   (row_base + feature) * sizeof(float));
        sum_squares += value * value;
      }
      const float inverse_rms =
          1.0f / std::sqrt(sum_squares / command.tensor_shape.features +
                           epsilon);
      for (uint32_t feature = 0; feature < command.tensor_shape.features;
           ++feature) {
        const float value =
            LoadFloat(*memory, lhs_base +
                                   (row_base + feature) * sizeof(float));
        const float weight =
            LoadFloat(*memory, rhs_base + feature * sizeof(float));
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
          StoreFloat(memory, dst_base + (query_index + lane) * 4, sum);
        }
      }
    }
  } else if (command.operation == xopennpux::Operation::kTtopk) {
    const uint32_t k = command.scalar_param0;
    const size_t value_count =
        static_cast<size_t>(command.tensor_shape.rows) * k;
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
                    dst_base + (value_count + output) * sizeof(uint32_t),
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
