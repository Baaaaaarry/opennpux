#include "hw_sim/gem5_bridge/gem5_tmma_coprocessor.h"

#include <algorithm>
#include <cstring>

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

float LoadFloat(const std::vector<uint8_t>& memory, size_t offset) {
  float value = 0.0f;
  std::memcpy(&value, memory.data() + offset, sizeof(value));
  return value;
}

void StoreFloat(std::vector<uint8_t>* memory, size_t offset, float value) {
  std::memcpy(memory->data() + offset, &value, sizeof(value));
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
      operation != xopennpux::Operation::kTmul) {
    return Gem5TmmaSubmitResult::kIllegalInstruction;
  }
  if (!ready()) {
    return Gem5TmmaSubmitResult::kBackpressure;
  }

  const uint32_t shape_csr = packet.csr_epoch == 0 ? mma_shape_
                                                    : packet.mma_shape;
  const uint32_t data_type_csr = packet.csr_epoch == 0
                                     ? mma_data_type_
                                     : packet.mma_data_type;
  const xopennpux::MmaShape shape = xopennpux::DecodeMmaShape(shape_csr);
  const xopennpux::MmaDataTypes data_types =
      xopennpux::DecodeMmaDataTypes(data_type_csr);
  constexpr uint32_t kShapeReservedMask = 0xc0000000;
  constexpr uint32_t kDataTypeReservedMask = 0xfffff000;
  if (shape.m == 0 || shape.n == 0 || shape.k == 0 ||
      (shape_csr & kShapeReservedMask) != 0 ||
      (data_type_csr & kDataTypeReservedMask) != 0 ||
      data_types.src1 != xopennpux::DataType::kFp32 ||
      data_types.src2 != xopennpux::DataType::kFp32 ||
      data_types.dst != xopennpux::DataType::kFp32) {
    return Gem5TmmaSubmitResult::kInvalidCsrState;
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
  queue_[tail].csr_epoch = packet.csr_epoch == 0 ? csr_epoch_
                                                  : packet.csr_epoch;
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
  const uint64_t tensor_elements = static_cast<uint64_t>(command.shape.m) *
                                   command.shape.n * command.shape.k;
  if (command.operation == xopennpux::Operation::kTmma) {
    completion->mac_operations = tensor_elements;
    completion->modeled_cycles = completion->mac_operations;
  } else {
    completion->element_operations = tensor_elements;
    completion->modeled_cycles = completion->element_operations;
  }

  if (command.data_types.src1 != xopennpux::DataType::kFp32 ||
      command.data_types.src2 != xopennpux::DataType::kFp32 ||
      command.data_types.dst != xopennpux::DataType::kFp32) {
    completion->error = Gem5TmmaExecutionError::kUnsupportedDataType;
    return true;
  }

  const uint64_t lhs_elements =
      command.operation == xopennpux::Operation::kTmma
          ? static_cast<uint64_t>(command.shape.m) * command.shape.k
          : tensor_elements;
  const uint64_t rhs_elements =
      command.operation == xopennpux::Operation::kTmma
          ? static_cast<uint64_t>(command.shape.k) * command.shape.n
          : tensor_elements;
  const uint64_t dst_elements =
      command.operation == xopennpux::Operation::kTmma
          ? static_cast<uint64_t>(command.shape.m) * command.shape.n
          : tensor_elements;
  const uint64_t dst_bytes = dst_elements * sizeof(float);
  if (!MatrixRangeValid(command.dispatch.rs1_value, lhs_elements, memory_base,
                        memory->size())) {
    completion->error = Gem5TmmaExecutionError::kAddress;
    completion->faulting_address = command.dispatch.rs1_value;
    return true;
  }
  if (!MatrixRangeValid(command.dispatch.rs2_value, rhs_elements, memory_base,
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

  const size_t lhs_base = command.dispatch.rs1_value - memory_base;
  const size_t rhs_base = command.dispatch.rs2_value - memory_base;
  const size_t dst_base = command.dispatch.rd_value - memory_base;
  if (command.operation == xopennpux::Operation::kTmma) {
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
