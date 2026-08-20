#include "hw_sim/gem5_bridge/gem5_host_tensor_arena.h"

#include <limits>

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
