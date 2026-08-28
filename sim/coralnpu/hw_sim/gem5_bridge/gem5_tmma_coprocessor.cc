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
      operation != xopennpux::Operation::kTtopk) {
    return Gem5TmmaSubmitResult::kIllegalInstruction;
  }
  if (!ready()) {
    return Gem5TmmaSubmitResult::kBackpressure;
  }

  const bool is_mma = operation == xopennpux::Operation::kTmma;
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
      data_types.src1 != xopennpux::DataType::kFp32 ||
      data_types.src2 != xopennpux::DataType::kFp32 ||
      data_types.dst != xopennpux::DataType::kFp32) {
    return Gem5TmmaSubmitResult::kInvalidCsrState;
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
  } else {
    completion->element_operations = tensor_elements;
    completion->modeled_cycles = completion->element_operations;
  }

  const xopennpux::MmaDataTypes data_types =
      command.operation == xopennpux::Operation::kTmma
          ? command.data_types
          : command.tensor_data_types;
  if (data_types.src1 != xopennpux::DataType::kFp32 ||
      data_types.src2 != xopennpux::DataType::kFp32 ||
      data_types.dst != xopennpux::DataType::kFp32) {
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
          : command.operation == xopennpux::Operation::kTrmsnorm
              ? command.tensor_shape.features
              : command.operation == xopennpux::Operation::kTrope
                  ? tensor_elements * 2
              : command.operation == xopennpux::Operation::kTgather
                  ? command.tensor_shape.rows
              : command.operation == xopennpux::Operation::kTsoftmax ||
                        command.operation == xopennpux::Operation::kTsilu ||
                        command.operation == xopennpux::Operation::kTtopk
                    ? 0
                    : tensor_elements;
  const uint64_t dst_elements =
      command.operation == xopennpux::Operation::kTmma
          ? static_cast<uint64_t>(command.shape.m) * command.shape.n
          : command.operation == xopennpux::Operation::kTtopk
              ? static_cast<uint64_t>(command.tensor_shape.rows) *
                    command.scalar_param0 * 2
          : tensor_elements;
  const uint64_t dst_bytes = dst_elements * sizeof(float);
  if (!MatrixRangeValid(command.dispatch.rs1_value, lhs_elements, memory_base,
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
