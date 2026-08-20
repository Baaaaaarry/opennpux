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
  const Gem5HostWeightBinding direct_expert = {
      OPENNPUX_NPU_WEIGHT_ROLE_ROUTED_EXPERT, 0,
      OPENNPUX_NPU_WEIGHT_SLOT_GATE_PROJ};
  assert(graph.ExecuteGptqExpert(expert_index, &weights, direct_expert));
  assert(graph.ExecuteCommand(expert_index, &weights));
  uint32_t router_index = UINT32_MAX;
  for (uint32_t index = 0; index < graph.command_count(); ++index) {
    if (graph.command(index)->opcode == OPENNPUX_NPU_OP_ROUTER) {
      router_index = index;
    }
  }
  assert(router_index != UINT32_MAX);
  opennpux_npu_functional_request router = {};
  assert(graph.Materialize(router_index, nullptr, 0, &router));
  const auto* router_input = FindOperand(router, OPENNPUX_NPU_OPERAND_INPUT);
  const auto* router_ids =
      FindOperand(router, OPENNPUX_NPU_OPERAND_OUTPUT_INDICES);
  const auto* router_weights =
      FindOperand(router, OPENNPUX_NPU_OPERAND_OUTPUT);
  assert(router_input != nullptr && router_ids != nullptr &&
         router_weights != nullptr);
  auto* router_input_data = reinterpret_cast<float*>(graph.arena().Translate(
      router_input->address, router_input->byte_size));
  auto* router_id_data = reinterpret_cast<uint32_t*>(graph.arena().Translate(
      router_ids->address, router_ids->byte_size));
  auto* router_weight_data = reinterpret_cast<float*>(graph.arena().Translate(
      router_weights->address, router_weights->byte_size));
  assert(router_input_data != nullptr && router_id_data != nullptr &&
         router_weight_data != nullptr);
  for (size_t index = 0; index < router_input->byte_size / sizeof(float);
       ++index) {
    router_input_data[index] = static_cast<float>(index % 7) / 7.0f;
  }
  assert(graph.ExecuteGptqRouter(router_index, &weights));
  const size_t route_count =
      static_cast<size_t>(router.rows) * router.top_k;
  for (size_t row = 0; row < router.rows; ++row) {
    float sum = 0.0f;
    for (size_t route = 0; route < router.top_k; ++route) {
      const size_t index = row * router.top_k + route;
      assert(index < route_count && router_id_data[index] < 8);
      assert(std::isfinite(router_weight_data[index]) &&
             router_weight_data[index] >= 0.0f);
      sum += router_weight_data[index];
    }
    assert(std::fabs(sum - 1.0f) < 1.0e-5f);
  }
  assert(graph.stats().completed_commands == 6);
  assert(graph.ExecuteCommand(add_index, &weights));
  assert(graph.ExecuteCommand(matmul_index, &weights));
  assert(graph.ExecuteCommand(router_index, &weights));
  uint32_t rope_index = UINT32_MAX;
  for (uint32_t index = 0; index < graph.command_count(); ++index) {
    if (graph.command(index)->opcode == OPENNPUX_NPU_OP_ROPE) {
      rope_index = index;
      break;
    }
  }
  assert(rope_index != UINT32_MAX);
  assert(graph.ExecuteCommand(rope_index, &weights));
  assert(graph.stats().completed_commands == 10);
  uint32_t failed_command = UINT32_MAX;
  assert(!graph.ExecuteProgram(&weights, &failed_command));
  assert(failed_command == 0);
  const uint32_t input_tokens[] = {0, 1};
  assert(graph.SetInputTokenIds(input_tokens, 2));
  uint32_t topk_index = UINT32_MAX;
  for (uint32_t index = 0; index < graph.command_count(); ++index) {
    if (graph.command(index)->opcode == OPENNPUX_NPU_OP_TOPK) {
      topk_index = index;
    }
  }
  assert(topk_index != UINT32_MAX);
  opennpux_npu_functional_request topk = {};
  assert(graph.Materialize(topk_index, nullptr, 0, &topk));
  const auto* topk_input = FindOperand(topk, OPENNPUX_NPU_OPERAND_INPUT);
  assert(topk_input != nullptr);
  auto* logits = reinterpret_cast<float*>(graph.arena().Translate(
      topk_input->address, topk_input->byte_size));
  assert(logits != nullptr);
  for (size_t index = 0; index < topk_input->byte_size / sizeof(float);
       ++index) {
    logits[index] = static_cast<float>(index);
  }
  assert(graph.ExecuteCommand(topk_index, &weights));
  uint32_t next_token = 0;
  assert(graph.ReadNextToken(&next_token));
  assert(next_token == 31);
  std::printf("functional_graph_add_elements=%zu\n", count);
  std::puts("functional_graph_gptq_projection=PASS");
  std::puts("functional_graph_routed_expert=PASS");
  std::puts("functional_graph_direct_expert=PASS");
  std::puts("functional_graph_gptq_router=PASS");
  std::puts("functional_graph_auto_dispatch=PASS");
  std::puts("functional_graph_program_failure_location=PASS");
  std::puts("functional_graph_token_io=PASS");
  std::puts("gem5_host_functional_graph=PASS");
  std::free(submission);
  opennpux_npu_executable_unload(&executable);
  return 0;
}
