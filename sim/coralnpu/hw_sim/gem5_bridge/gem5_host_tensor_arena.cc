#include "hw_sim/gem5_bridge/gem5_host_tensor_arena.h"

#include <algorithm>
#include <cstring>
#include <limits>

namespace {

size_t ElementBytes(uint32_t data_type) {
  switch (data_type) {
    case 2:
      return 1;
    case 3:
    case 6:
      return 4;
    case 4:
    case 5:
      return 2;
    default:
      return 0;
  }
}

bool ResolveDimensions(const opennpux_npu_tensor_plan_tensor& tensor,
                       const opennpux_npu_tensor_plan_runtime& runtime,
                       uint32_t* dimensions) {
  for (uint32_t index = 0; index < tensor.rank; ++index) {
    const uint32_t symbol =
        (tensor.dimension_symbols >> (index * 4)) & UINT32_C(0xf);
    switch (symbol) {
      case OPENNPUX_NPU_DIMENSION_STATIC:
        dimensions[index] = tensor.dimensions[index];
        break;
      case OPENNPUX_NPU_DIMENSION_BATCH:
        dimensions[index] = runtime.batch_size;
        break;
      case OPENNPUX_NPU_DIMENSION_SEQUENCE:
        dimensions[index] = runtime.sequence_length;
        break;
      case OPENNPUX_NPU_DIMENSION_KV:
        dimensions[index] = runtime.kv_length;
        break;
      case OPENNPUX_NPU_DIMENSION_ACTIVE_EXPERTS:
        dimensions[index] = runtime.active_experts;
        break;
      default:
        return false;
    }
    if (dimensions[index] == 0) return false;
  }
  return true;
}

}  // namespace

Gem5HostTensorArena::~Gem5HostTensorArena() {
  opennpux_npu_tensor_plan_unload(&plan_);
}

bool Gem5HostTensorArena::Load(const std::string& tensor_plan_path) {
  opennpux_npu_tensor_plan_unload(&plan_);
  Reset();
  return !tensor_plan_path.empty() &&
         opennpux_npu_tensor_plan_load(tensor_plan_path.c_str(), &plan_) == 0;
}

bool Gem5HostTensorArena::Configure(
    const opennpux_npu_tensor_plan_runtime& runtime, uint32_t arena_base) {
  if (!loaded() || arena_base == 0) {
    return false;
  }
  opennpux_npu_tensor_plan_memory memory = {};
  uint64_t required = 0;
  if (opennpux_npu_tensor_plan_memory_layout(
          &plan_, &runtime, arena_base, UINT32_MAX - arena_base, &memory,
          &required) != 0 ||
      required == 0 || required > std::numeric_limits<size_t>::max() ||
      required > UINT32_MAX - arena_base) {
    return false;
  }
  try {
    storage_.assign(static_cast<size_t>(required), 0);
  } catch (...) {
    Reset();
    return false;
  }
  runtime_ = runtime;
  memory_ = memory;
  arena_base_ = arena_base;
  return true;
}

bool Gem5HostTensorArena::ConfigurePreservingPersistent(
    const opennpux_npu_tensor_plan_runtime& runtime, uint32_t arena_base) {
  if (!loaded() || storage_.empty()) {
    return Configure(runtime, arena_base);
  }
  const auto old_runtime = runtime_;
  const auto old_memory = memory_;
  const uint32_t old_base = arena_base_;
  std::vector<uint8_t> old_storage = storage_;
  if (!Configure(runtime, arena_base)) return false;

  for (uint32_t id = 0; id < plan_.header->tensor_count; ++id) {
    const auto& tensor = plan_.tensors[id];
    if (tensor.storage != OPENNPUX_NPU_TENSOR_PERSISTENT) continue;
    uint64_t old_address = 0;
    uint64_t old_size = 0;
    uint64_t new_address = 0;
    uint64_t new_size = 0;
    uint32_t old_dimensions[OPENNPUX_NPU_TENSOR_PLAN_MAX_RANK] = {};
    uint32_t new_dimensions[OPENNPUX_NPU_TENSOR_PLAN_MAX_RANK] = {};
    const size_t element_bytes = ElementBytes(tensor.data_type);
    if (element_bytes == 0 ||
        !ResolveDimensions(tensor, old_runtime, old_dimensions) ||
        !ResolveDimensions(tensor, runtime, new_dimensions) ||
        opennpux_npu_tensor_plan_resolve(&plan_, id, &old_runtime,
                                         &old_memory, &old_address,
                                         &old_size) != 0 ||
        opennpux_npu_tensor_plan_resolve(&plan_, id, &runtime, &memory_,
                                         &new_address, &new_size) != 0 ||
        old_address < old_base || old_address - old_base > old_storage.size() ||
        old_size > old_storage.size() - (old_address - old_base)) {
      return false;
    }
    auto* destination = Translate(new_address, new_size);
    if (destination == nullptr) return false;
    const auto* source = old_storage.data() + (old_address - old_base);
    if (tensor.rank == 0) {
      std::memcpy(destination, source,
                  static_cast<size_t>(std::min(old_size, new_size)));
      continue;
    }
    uint64_t old_strides[OPENNPUX_NPU_TENSOR_PLAN_MAX_RANK] = {};
    uint64_t new_strides[OPENNPUX_NPU_TENSOR_PLAN_MAX_RANK] = {};
    old_strides[tensor.rank - 1] = 1;
    new_strides[tensor.rank - 1] = 1;
    for (uint32_t reverse = 1; reverse < tensor.rank; ++reverse) {
      const uint32_t index = tensor.rank - 1 - reverse;
      old_strides[index] = old_strides[index + 1] * old_dimensions[index + 1];
      new_strides[index] = new_strides[index + 1] * new_dimensions[index + 1];
    }
    uint64_t prefix_count = 1;
    for (uint32_t index = 0; index + 1 < tensor.rank; ++index) {
      prefix_count *= std::min(old_dimensions[index], new_dimensions[index]);
    }
    const size_t run_elements =
        std::min(old_dimensions[tensor.rank - 1],
                 new_dimensions[tensor.rank - 1]);
    for (uint64_t linear = 0; linear < prefix_count; ++linear) {
      uint64_t remaining = linear;
      uint64_t old_offset = 0;
      uint64_t new_offset = 0;
      for (uint32_t reverse = 1; reverse < tensor.rank; ++reverse) {
        const uint32_t index = tensor.rank - 1 - reverse;
        const uint32_t extent =
            std::min(old_dimensions[index], new_dimensions[index]);
        const uint32_t coordinate = remaining % extent;
        remaining /= extent;
        old_offset += coordinate * old_strides[index];
        new_offset += coordinate * new_strides[index];
      }
      std::memcpy(destination + new_offset * element_bytes,
                  source + old_offset * element_bytes,
                  run_elements * element_bytes);
    }
  }
  return true;
}

bool Gem5HostTensorArena::ResolveCommand(
    uint32_t command_id, opennpux_npu_command_tensor_views* views) const {
  return loaded() && !storage_.empty() && views != nullptr &&
         opennpux_npu_tensor_plan_resolve_command(
             &plan_, command_id, &runtime_, &memory_, views) == 0;
}

uint8_t* Gem5HostTensorArena::Translate(uint64_t address, uint64_t size) {
  return const_cast<uint8_t*>(
      static_cast<const Gem5HostTensorArena*>(this)->Translate(address, size));
}

const uint8_t* Gem5HostTensorArena::Translate(uint64_t address,
                                               uint64_t size) const {
  if (storage_.empty() || address < arena_base_) {
    return nullptr;
  }
  const uint64_t offset = address - arena_base_;
  if (offset > storage_.size() || size > storage_.size() - offset) {
    return nullptr;
  }
  return storage_.data() + offset;
}

void Gem5HostTensorArena::Reset() {
  runtime_ = {};
  memory_ = {};
  storage_.clear();
  arena_base_ = 0;
}
