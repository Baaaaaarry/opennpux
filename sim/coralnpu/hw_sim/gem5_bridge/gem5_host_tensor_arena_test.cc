#include "hw_sim/gem5_bridge/gem5_host_tensor_arena.h"

#include <cassert>
#include <cstring>
#include <cstdio>

int main(int argc, char** argv) {
  assert(argc == 2);
  Gem5HostTensorArena arena;
  assert(arena.Load(argv[1]));
  assert(arena.command_count() == 30);
  assert(arena.tensor_count() >= 40);
  const opennpux_npu_tensor_plan_runtime runtime = {2, 4, 7, 2};
  assert(arena.Configure(runtime));
  assert(arena.size() != 0);

  for (uint32_t command = 0; command < arena.command_count(); ++command) {
    opennpux_npu_command_tensor_views views = {};
    assert(arena.ResolveCommand(command, &views));
    assert(views.command_id == command);
    for (uint32_t index = 0; index < views.input_count; ++index) {
      assert(views.inputs[index].address >= arena.base());
      assert(views.inputs[index].address + views.inputs[index].size <=
             arena.base() + arena.size());
      assert(arena.Translate(views.inputs[index].address,
                             views.inputs[index].size) != nullptr);
    }
    for (uint32_t index = 0; index < views.output_count; ++index) {
      assert(views.outputs[index].address >= arena.base());
      assert(views.outputs[index].address + views.outputs[index].size <=
             arena.base() + arena.size());
      assert(arena.Translate(views.outputs[index].address,
                             views.outputs[index].size) != nullptr);
    }
  }
  uint32_t persistent_id = UINT32_MAX;
  for (uint32_t id = 0; id < arena.tensor_count(); ++id) {
    if (arena.plan().tensors[id].storage ==
        OPENNPUX_NPU_TENSOR_PERSISTENT) {
      persistent_id = id;
      break;
    }
  }
  assert(persistent_id != UINT32_MAX);
  uint64_t persistent_address = 0;
  uint64_t persistent_size = 0;
  assert(opennpux_npu_tensor_plan_resolve(
             &arena.plan(), persistent_id, &arena.runtime(), &arena.memory(),
             &persistent_address, &persistent_size) == 0);
  assert(persistent_size >= sizeof(uint32_t));
  const uint32_t marker = UINT32_C(0x4e505558);
  std::memcpy(arena.Translate(persistent_address, sizeof(marker)), &marker,
              sizeof(marker));
  const opennpux_npu_tensor_plan_runtime decode_runtime = {2, 1, 8, 2};
  assert(arena.ConfigurePreservingPersistent(decode_runtime));
  assert(opennpux_npu_tensor_plan_resolve(
             &arena.plan(), persistent_id, &arena.runtime(), &arena.memory(),
             &persistent_address, &persistent_size) == 0);
  uint32_t restored = 0;
  std::memcpy(&restored, arena.Translate(persistent_address, sizeof(restored)),
              sizeof(restored));
  assert(restored == marker);
  assert(arena.Translate(arena.base() - 1, 1) == nullptr);
  assert(arena.Translate(arena.base() + arena.size(), 1) == nullptr);
  assert(arena.Translate(arena.base(), arena.size() + 1) == nullptr);
  arena.Reset();
  assert(arena.size() == 0);
  std::puts("gem5_host_tensor_arena=PASS");
  return 0;
}
