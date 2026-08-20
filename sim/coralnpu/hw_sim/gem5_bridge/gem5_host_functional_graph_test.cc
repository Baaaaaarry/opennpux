#include "hw_sim/gem5_bridge/gem5_host_functional_graph.h"
#include "hw_sim/gem5_bridge/gem5_host_weight_provider.h"

#include <cassert>
#include <cmath>
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
  assert(argc == 5);
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
  uint32_t matmul_index = UINT32_MAX;
  for (uint32_t index = 0; index < graph.command_count(); ++index) {
    if (graph.command(index)->opcode == OPENNPUX_NPU_OP_MATMUL) {
      matmul_index = index;
      break;
    }
  }
  assert(matmul_index != UINT32_MAX);
  opennpux_npu_functional_request matmul = {};
  assert(graph.Materialize(matmul_index, nullptr, 0, &matmul));
  const auto* matmul_input = FindOperand(matmul, OPENNPUX_NPU_OPERAND_INPUT);
  const auto* matmul_output = FindOperand(matmul, OPENNPUX_NPU_OPERAND_OUTPUT);
  assert(matmul_input != nullptr && matmul_output != nullptr);
  auto* matmul_input_data = reinterpret_cast<float*>(
      graph.arena().Translate(matmul_input->address, matmul_input->byte_size));
  auto* matmul_output_data = reinterpret_cast<float*>(
      graph.arena().Translate(matmul_output->address, matmul_output->byte_size));
  assert(matmul_input_data != nullptr && matmul_output_data != nullptr);
  for (size_t index = 0; index < matmul_input->byte_size / sizeof(float);
       ++index) {
    matmul_input_data[index] = 1.0f;
  }
  Gem5HostWeightProvider weights;
  assert(weights.Load(argv[3], argv[4], 4096));
  assert(graph.ExecuteGptqQkv(matmul_index, &weights));
  for (size_t index = 0; index < matmul_output->byte_size / sizeof(float);
       ++index) {
    assert(std::isfinite(matmul_output_data[index]));
  }
  assert(graph.stats().completed_commands == 2);
  uint32_t expert_index = UINT32_MAX;
  for (uint32_t index = 0; index < graph.command_count(); ++index) {
    if (graph.command(index)->opcode == OPENNPUX_NPU_OP_EXPERT) {
      expert_index = index;
      break;
    }
  }
  assert(expert_index != UINT32_MAX);
  opennpux_npu_functional_request expert = {};
  assert(graph.Materialize(expert_index, nullptr, 0, &expert));
  const auto* expert_input = FindOperand(expert, OPENNPUX_NPU_OPERAND_INPUT);
  const auto* expert_ids = FindOperand(expert, OPENNPUX_NPU_OPERAND_SECONDARY);
  const auto* expert_routes =
      FindOperand(expert, OPENNPUX_NPU_OPERAND_INPUT_TERTIARY);
  const auto* expert_output = FindOperand(expert, OPENNPUX_NPU_OPERAND_OUTPUT);
  assert(expert_input != nullptr && expert_ids != nullptr &&
         expert_routes != nullptr && expert_output != nullptr);
  auto* expert_input_data = reinterpret_cast<float*>(graph.arena().Translate(
      expert_input->address, expert_input->byte_size));
  auto* expert_id_data = reinterpret_cast<uint32_t*>(
      graph.arena().Translate(expert_ids->address, expert_ids->byte_size));
  auto* expert_route_data = reinterpret_cast<float*>(
      graph.arena().Translate(expert_routes->address,
                              expert_routes->byte_size));
  auto* expert_output_data = reinterpret_cast<float*>(graph.arena().Translate(
      expert_output->address, expert_output->byte_size));
  assert(expert_input_data != nullptr && expert_id_data != nullptr &&
         expert_route_data != nullptr && expert_output_data != nullptr);
  for (size_t index = 0; index < expert_input->byte_size / sizeof(float);
       ++index) {
    expert_input_data[index] = 1.0f;
  }
  for (size_t index = 0; index < expert_ids->byte_size / sizeof(uint32_t);
       ++index) {
    expert_id_data[index] = 0;
    expert_route_data[index] =
        index % graph.arena().runtime().active_experts == 0 ? 1.0f : 0.0f;
  }
  assert(graph.ExecuteRoutedExpert(expert_index, &weights));
  for (size_t index = 0; index < expert_output->byte_size / sizeof(float);
       ++index) {
    assert(std::isfinite(expert_output_data[index]));
  }
  assert(graph.stats().completed_commands == 3);
  std::printf("functional_graph_add_elements=%zu\n", count);
  std::puts("functional_graph_gptq_projection=PASS");
  std::puts("functional_graph_routed_expert=PASS");
  std::puts("gem5_host_functional_graph=PASS");
  std::free(submission);
  opennpux_npu_executable_unload(&executable);
  return 0;
}
