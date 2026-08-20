#include "hw_sim/gem5_bridge/gem5_host_tensor_arena.h"

#include <cassert>
#include <cstdio>

int main(int argc, char** argv) {
  assert(argc == 2);
  Gem5HostTensorArena arena;
  assert(arena.Load(argv[1]));
  assert(arena.command_count() == 30);
  assert(arena.tensor_count() == 39);
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
  assert(arena.Translate(arena.base() - 1, 1) == nullptr);
  assert(arena.Translate(arena.base() + arena.size(), 1) == nullptr);
  assert(arena.Translate(arena.base(), arena.size() + 1) == nullptr);
  arena.Reset();
  assert(arena.size() == 0);
  std::puts("gem5_host_tensor_arena=PASS");
  return 0;
}
