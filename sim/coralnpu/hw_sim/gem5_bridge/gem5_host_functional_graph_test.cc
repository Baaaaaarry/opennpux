#include "hw_sim/gem5_bridge/gem5_host_functional_graph.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "opennpux/npu_executable.h"

namespace {

const opennpux_npu_functional_operand* FindOperand(
    const opennpux_npu_functional_request& request, uint32_t role) {
  for (uint32_t index = 0; index < request.operand_count; ++index) {
    if (request.operands[index].role == role) {
      return &request.operands[index];
    }
  }
  return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
  assert(argc == 3);
  opennpux_npu_executable executable = {};
  assert(opennpux_npu_executable_load(argv[1], &executable) == 0);
  opennpux_npu_tensor_binding bindings[5] = {};
  for (uint32_t index = 0; index < 5; ++index) {
    bindings[index].tensor_id = index;
    bindings[index].flags = OPENNPUX_NPU_BIND_READ;
    bindings[index].data_type = OPENNPUX_NPU_DTYPE_FLOAT32;
    bindings[index].rank = 2;
    bindings[index].device_address = UINT32_C(0x10000000) + index * 0x100000;
    bindings[index].byte_size = 0x100000;
    bindings[index].dimensions[0] = 1;
    bindings[index].dimensions[1] = 18;
    bindings[index].memory_object = index + 1;
  }
  bindings[1].flags = OPENNPUX_NPU_BIND_WRITE;
  bindings[2].flags |= OPENNPUX_NPU_BIND_WEIGHT;
  bindings[3].flags |= OPENNPUX_NPU_BIND_PERSISTENT |
                       OPENNPUX_NPU_BIND_WRITE;
  bindings[4].flags |= OPENNPUX_NPU_BIND_WRITE;

  constexpr size_t kCapacity = 1024 * 1024;
  void* submission =
      std::aligned_alloc(OPENNPUX_NPU_RECORD_ALIGNMENT, kCapacity);
  assert(submission != nullptr);
  const opennpux_npu_invocation_parameters invocation = {1, 2, 3, 2};
  size_t submission_size = 0;
  assert(opennpux_npu_executable_instantiate_with_parameters(
             &executable, OPENNPUX_NPU_ENTRY_DECODE, 1, 1, &invocation,
             bindings, 5, submission, kCapacity, &submission_size) == 0);

  Gem5HostFunctionalGraph graph;
  assert(graph.LoadTensorPlan(argv[2]));
  assert(graph.Configure(submission, submission_size, UINT32_C(0x24000000)));
  uint32_t add_index = UINT32_MAX;
  for (uint32_t index = 0; index < graph.command_count(); ++index) {
    if (graph.command(index)->opcode == OPENNPUX_NPU_OP_ADD) {
      add_index = index;
      break;
    }
  }
  assert(add_index != UINT32_MAX);
  opennpux_npu_functional_request request = {};
  assert(graph.Materialize(add_index, nullptr, 0, &request));
  const auto* lhs = FindOperand(request, OPENNPUX_NPU_OPERAND_INPUT);
  const auto* rhs = FindOperand(request, OPENNPUX_NPU_OPERAND_SECONDARY);
  const auto* output = FindOperand(request, OPENNPUX_NPU_OPERAND_OUTPUT);
  assert(lhs != nullptr && rhs != nullptr && output != nullptr);
  assert(lhs->byte_size == rhs->byte_size && lhs->byte_size == output->byte_size);
  assert(lhs->byte_size % sizeof(float) == 0);
  auto* lhs_data = reinterpret_cast<float*>(
      graph.arena().Translate(lhs->address, lhs->byte_size));
  auto* rhs_data = reinterpret_cast<float*>(
      graph.arena().Translate(rhs->address, rhs->byte_size));
  auto* output_data = reinterpret_cast<float*>(
      graph.arena().Translate(output->address, output->byte_size));
  assert(lhs_data != nullptr && rhs_data != nullptr && output_data != nullptr);
  const size_t count = lhs->byte_size / sizeof(float);
  for (size_t index = 0; index < count; ++index) {
    lhs_data[index] = static_cast<float>(index + 1);
    rhs_data[index] = static_cast<float>((index + 1) * 2);
  }
  assert(graph.Execute(&request));
  for (size_t index = 0; index < count; ++index) {
    assert(output_data[index] == static_cast<float>((index + 1) * 3));
  }
  assert(graph.stats().completed_commands == 1 &&
         graph.stats().operations == count);
  std::printf("functional_graph_add_elements=%zu\n", count);
  std::puts("gem5_host_functional_graph=PASS");
  std::free(submission);
  opennpux_npu_executable_unload(&executable);
  return 0;
}
