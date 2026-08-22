#ifndef HW_SIM_GEM5_BRIDGE_GEM5_HOST_TENSOR_ARENA_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_HOST_TENSOR_ARENA_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "opennpux/npu_tensor_plan.h"

// Owns the host-side tensor storage used by graph-level functional execution.
// Addresses remain 32-bit NPU addresses so materialized requests can use the
// same ABI as RTL, while bytes are backed by a private host vector.
class Gem5HostTensorArena {
 public:
  Gem5HostTensorArena() = default;
  ~Gem5HostTensorArena();

  Gem5HostTensorArena(const Gem5HostTensorArena&) = delete;
  Gem5HostTensorArena& operator=(const Gem5HostTensorArena&) = delete;

  bool Load(const std::string& tensor_plan_path);
  bool Configure(const opennpux_npu_tensor_plan_runtime& runtime,
                 uint32_t arena_base = UINT32_C(0x30000000));
  bool ConfigurePreservingPersistent(
      const opennpux_npu_tensor_plan_runtime& runtime,
      uint32_t arena_base = UINT32_C(0x30000000));
  bool ResolveCommand(uint32_t command_id,
                      opennpux_npu_command_tensor_views* views) const;
  uint8_t* Translate(uint64_t address, uint64_t size);
  const uint8_t* Translate(uint64_t address, uint64_t size) const;
  void Reset();

  bool loaded() const { return plan_.header != nullptr; }
  uint32_t command_count() const {
    return loaded() ? plan_.header->command_count : 0;
  }
  uint32_t tensor_count() const {
    return loaded() ? plan_.header->tensor_count : 0;
  }
  uint32_t base() const { return arena_base_; }
  uint8_t* data() { return storage_.data(); }
  const uint8_t* data() const { return storage_.data(); }
  size_t size() const { return storage_.size(); }
  const opennpux_npu_tensor_plan& plan() const { return plan_; }
  const opennpux_npu_tensor_plan_memory& memory() const { return memory_; }
  const opennpux_npu_tensor_plan_runtime& runtime() const { return runtime_; }

 private:
  opennpux_npu_tensor_plan plan_ = {};
  opennpux_npu_tensor_plan_runtime runtime_ = {};
  opennpux_npu_tensor_plan_memory memory_ = {};
  std::vector<uint8_t> storage_;
  uint32_t arena_base_ = 0;
};

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_HOST_TENSOR_ARENA_H_
