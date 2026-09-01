#include "hw_sim/gem5_bridge/gem5_host_functional_graph.h"
#include "hw_sim/gem5_bridge/gem5_host_xgraph_executor.h"
#include "hw_sim/gem5_bridge/gem5_host_weight_provider.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

class CapturingObserver final : public Gem5HostFunctionalRequestObserver {
 public:
  void Observe(Gem5HostFunctionalExecutionPath observed_path,
               const opennpux_npu_functional_request& request,
               const Gem5FunctionalMemoryRegion*, size_t region_count) override {
    ++count;
    path = observed_path;
    command_id = request.command_id;
    regions = region_count;
  }

  uint32_t count = 0;
  uint32_t command_id = UINT32_MAX;
  size_t regions = 0;
  Gem5HostFunctionalExecutionPath path =
      Gem5HostFunctionalExecutionPath::kHostFusedRoutedExpert;
};

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
  lhs_data[0] = 1.001f;
  rhs_data[0] = 0.0f;
  assert(setenv("OPENNPUX_HOST_FUNCTIONAL_PRECISION", "bf16", 1) == 0);
  assert(setenv("OPENNPUX_HOST_FUNCTIONAL_EXECUTION",
                "xopennpux-primitives", 1) == 0);
  CapturingObserver observer;
  graph.SetRequestObserver(&observer);
  assert(graph.Execute(&request));
  graph.SetRequestObserver(nullptr);
  assert(observer.count == 1 && observer.command_id == add_index &&
         observer.regions == 2 &&
         observer.path == Gem5HostFunctionalExecutionPath::kGenericRequest);
  assert(unsetenv("OPENNPUX_HOST_FUNCTIONAL_PRECISION") == 0);
  assert(unsetenv("OPENNPUX_HOST_FUNCTIONAL_EXECUTION") == 0);
  assert(output_data[0] == 1.0f);
  for (size_t index = 1; index < count; ++index) {
    assert(output_data[index] == static_cast<float>((index + 1) * 3));
  }
  float recurrent_values[] = {1.001f, 1.001f};
  opennpux_npu_functional_request recurrent_boundary = {};
  recurrent_boundary.opcode = OPENNPUX_NPU_OP_RECURRENT_UPDATE;
  recurrent_boundary.operand_count = 2;
  recurrent_boundary.operands[0] = {
      OPENNPUX_NPU_OPERAND_OUTPUT, UINT32_C(0x60000000), sizeof(float), 0};
  recurrent_boundary.operands[1] = {
      OPENNPUX_NPU_OPERAND_OUTPUT_SECONDARY, UINT32_C(0x60000004),
      sizeof(float), 0};
  const std::vector<Gem5FunctionalMemoryRegion> recurrent_regions = {{
      UINT32_C(0x60000000),
      reinterpret_cast<uint8_t*>(recurrent_values),
      sizeof(recurrent_values),
  }};
  ApplyGem5HostBfloat16OutputBoundaries(recurrent_boundary,
                                        recurrent_regions);
  assert(recurrent_values[0] == 1.0f);
  assert(recurrent_values[1] == 1.001f);
  assert(graph.stats().completed_commands == 1 &&
         graph.stats().operations == count &&
         graph.stats().xgraph_requests == 1 &&
         graph.stats().xgraph_commands == 1 &&
         graph.stats().xgraph_operations == count &&
         graph.stats().xgraph_modeled_cycles == count &&
         graph.stats().xgraph_fallback_requests == 0);
  std::printf("functional_graph_xopennpux_add=PASS\n");

  std::vector<float> norm_weights(count, 1.0f);
  for (size_t index = 0; index < count; ++index) {
    lhs_data[index] = static_cast<float>(index + 1);
    output_data[index] = 0.0f;
  }
  opennpux_npu_functional_request normalize = {};
  normalize.magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  normalize.version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  normalize.struct_size = sizeof(normalize);
  normalize.opcode = OPENNPUX_NPU_OP_NORMALIZE;
  normalize.rows = 1;
  normalize.features = static_cast<uint32_t>(count);
  normalize.epsilon = 1.0e-6f;
  normalize.operand_count = 3;
  normalize.operands[0] = {OPENNPUX_NPU_OPERAND_INPUT, lhs->address,
                           lhs->byte_size, 0};
  normalize.operands[1] = {OPENNPUX_NPU_OPERAND_OUTPUT, output->address,
                           output->byte_size, 0};
  normalize.operands[2] = {
      OPENNPUX_NPU_OPERAND_WEIGHT, UINT32_C(0x60000000),
      static_cast<uint32_t>(norm_weights.size() * sizeof(float)), 0};
  const Gem5FunctionalMemoryRegion norm_region = {
      UINT32_C(0x60000000),
      reinterpret_cast<uint8_t*>(norm_weights.data()),
      norm_weights.size() * sizeof(float)};
  opennpux_npu_operator_parameters norm_parameters = {};
  norm_parameters.magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  norm_parameters.version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  norm_parameters.struct_size = sizeof(norm_parameters);
  norm_parameters.opcode = OPENNPUX_NPU_OP_NORMALIZE;
  norm_parameters.input_features = static_cast<uint32_t>(count);
  norm_parameters.output_features = static_cast<uint32_t>(count);
  Gem5HostXGraphExecutionStats norm_stats = {};
  assert(ExecuteGem5HostXGraphRequest(
             normalize, norm_parameters, &norm_region, 1, &graph.arena(),
             &norm_stats) == Gem5HostXGraphExecutionOutcome::kExecuted);
  double square_sum = 0.0;
  for (size_t index = 0; index < count; ++index) {
    square_sum += static_cast<double>(index + 1) * (index + 1);
  }
  const double inverse_rms =
      1.0 / std::sqrt(square_sum / count + normalize.epsilon);
  for (size_t index = 0; index < count; ++index) {
    const float expected = static_cast<float>((index + 1) * inverse_rms);
    assert(std::fabs(output_data[index] - expected) < 1.0e-5f);
  }
  assert(norm_stats.commands == 1 && norm_stats.operations == count * 4 &&
         norm_stats.modeled_cycles == count * 4);
  std::puts("functional_graph_xopennpux_external_norm=PASS");

  assert(count % 2 == 0);
  const uint32_t conv_rows = 2;
  const uint32_t conv_features = static_cast<uint32_t>(count / conv_rows);
  const uint32_t conv_kernel = 3;
  std::vector<float> conv_weights(conv_features * conv_kernel, 1.0f);
  std::fill(lhs_data, lhs_data + count, 1.0f);
  std::fill(rhs_data, rhs_data + count, 0.0f);
  std::fill(output_data, output_data + count, 0.0f);
  opennpux_npu_functional_request convolution = {};
  convolution.magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  convolution.version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  convolution.struct_size = sizeof(convolution);
  convolution.opcode = OPENNPUX_NPU_OP_CAUSAL_CONVOLUTION;
  convolution.rows = conv_rows;
  convolution.features = conv_features;
  convolution.operand_count = 5;
  convolution.operands[0] = {OPENNPUX_NPU_OPERAND_INPUT, lhs->address,
                             lhs->byte_size, 0};
  convolution.operands[1] = {OPENNPUX_NPU_OPERAND_SECONDARY, rhs->address,
                             rhs->byte_size, 0};
  convolution.operands[2] = {OPENNPUX_NPU_OPERAND_OUTPUT, output->address,
                             output->byte_size, 0};
  convolution.operands[3] = {OPENNPUX_NPU_OPERAND_OUTPUT_SECONDARY,
                             rhs->address, rhs->byte_size, 0};
  convolution.operands[4] = {
      OPENNPUX_NPU_OPERAND_WEIGHT, UINT32_C(0x61000000),
      static_cast<uint32_t>(conv_weights.size() * sizeof(float)), 0};
  const Gem5FunctionalMemoryRegion conv_region = {
      UINT32_C(0x61000000),
      reinterpret_cast<uint8_t*>(conv_weights.data()),
      conv_weights.size() * sizeof(float)};
  opennpux_npu_operator_parameters conv_parameters = {};
  conv_parameters.magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  conv_parameters.version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  conv_parameters.struct_size = sizeof(conv_parameters);
  conv_parameters.opcode = OPENNPUX_NPU_OP_CAUSAL_CONVOLUTION;
  conv_parameters.intermediate_features = conv_kernel;
  Gem5HostXGraphExecutionStats conv_stats = {};
  assert(ExecuteGem5HostXGraphRequest(
             convolution, conv_parameters, &conv_region, 1, &graph.arena(),
             &conv_stats) == Gem5HostXGraphExecutionOutcome::kExecuted);
  for (uint32_t feature = 0; feature < conv_features; ++feature) {
    assert(output_data[feature] == 1.0f);
    assert(output_data[conv_features + feature] == 2.0f);
    assert(rhs_data[feature] == 1.0f);
    assert(rhs_data[conv_features + feature] == 1.0f);
  }
  const uint64_t conv_operations =
      static_cast<uint64_t>(count) * conv_kernel * 2;
  assert(conv_stats.commands == 1 &&
         conv_stats.operations == conv_operations &&
         conv_stats.modeled_cycles == conv_operations);
  std::puts("functional_graph_xopennpux_external_causal_conv=PASS");

  constexpr uint32_t recurrent_rows = 2;
  constexpr uint32_t recurrent_key_heads = 1;
  constexpr uint32_t recurrent_value_heads = 1;
  constexpr uint32_t recurrent_key_dim = 2;
  constexpr uint32_t recurrent_value_dim = 2;
  constexpr uint32_t recurrent_qkv_features = 6;
  const float recurrent_qkv[recurrent_rows * recurrent_qkv_features] = {
      0.1f, 0.2f, 0.3f, 0.4f, 1.0f, 2.0f,
      0.2f, 0.1f, 0.4f, 0.3f, 2.0f, 1.0f};
  std::copy(recurrent_qkv, recurrent_qkv + 12, lhs_data);
  lhs_data[12] = 0.5f;
  lhs_data[13] = 0.5f;
  lhs_data[14] = 0.5f;
  lhs_data[15] = 0.5f;
  std::fill(rhs_data, rhs_data + 4, 0.0f);
  std::fill(output_data, output_data + 4, 0.0f);
  float recurrent_a_log[] = {-1.0f};
  float recurrent_dt_bias[] = {0.0f};
  opennpux_npu_functional_request recurrent = {};
  recurrent.magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  recurrent.version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  recurrent.struct_size = sizeof(recurrent);
  recurrent.opcode = OPENNPUX_NPU_OP_RECURRENT_UPDATE;
  recurrent.rows = recurrent_rows;
  recurrent.features = recurrent_qkv_features;
  recurrent.heads = recurrent_key_heads;
  recurrent.kv_heads = recurrent_value_heads;
  recurrent.head_dim = recurrent_key_dim;
  recurrent.operand_count = 7;
  recurrent.operands[0] = {
      OPENNPUX_NPU_OPERAND_INPUT, lhs->address,
      recurrent_rows * recurrent_qkv_features * sizeof(float), 0};
  recurrent.operands[1] = {OPENNPUX_NPU_OPERAND_SECONDARY,
                           lhs->address + UINT32_C(48),
                           recurrent_rows * sizeof(float), 0};
  recurrent.operands[2] = {OPENNPUX_NPU_OPERAND_INPUT_TERTIARY,
                           lhs->address + UINT32_C(56),
                           recurrent_rows * sizeof(float), 0};
  recurrent.operands[3] = {OPENNPUX_NPU_OPERAND_OUTPUT, output->address,
                           recurrent_rows * recurrent_value_dim *
                               sizeof(float),
                           0};
  recurrent.operands[4] = {OPENNPUX_NPU_OPERAND_OUTPUT_SECONDARY,
                           rhs->address,
                           recurrent_key_dim * recurrent_value_dim *
                               sizeof(float),
                           0};
  recurrent.operands[5] = {OPENNPUX_NPU_OPERAND_LINEAR_A_LOG_WEIGHT,
                           UINT32_C(0x62000000), sizeof(recurrent_a_log), 0};
  recurrent.operands[6] = {OPENNPUX_NPU_OPERAND_LINEAR_DT_BIAS_WEIGHT,
                           UINT32_C(0x62000100), sizeof(recurrent_dt_bias), 0};
  const Gem5FunctionalMemoryRegion recurrent_weight_regions[] = {
      {UINT32_C(0x62000000),
       reinterpret_cast<uint8_t*>(recurrent_a_log), sizeof(recurrent_a_log)},
      {UINT32_C(0x62000100),
       reinterpret_cast<uint8_t*>(recurrent_dt_bias),
       sizeof(recurrent_dt_bias)}};
  opennpux_npu_operator_parameters recurrent_parameters = {};
  recurrent_parameters.magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  recurrent_parameters.version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  recurrent_parameters.struct_size = sizeof(recurrent_parameters);
  recurrent_parameters.opcode = OPENNPUX_NPU_OP_RECURRENT_UPDATE;
  recurrent_parameters.flags = OPENNPUX_NPU_PARAMETER_GATED_DELTA_NET;
  recurrent_parameters.output_features =
      recurrent_value_heads * recurrent_value_dim;
  Gem5HostXGraphExecutionStats recurrent_stats = {};
  assert(ExecuteGem5HostXGraphRequest(
             recurrent, recurrent_parameters, recurrent_weight_regions, 2,
             &graph.arena(), &recurrent_stats) ==
         Gem5HostXGraphExecutionOutcome::kExecuted);
  for (uint32_t index = 0; index < 4; ++index) {
    assert(std::isfinite(output_data[index]));
    assert(std::isfinite(rhs_data[index]));
  }
  const uint64_t recurrent_operations =
      recurrent_rows * recurrent_value_heads *
      (recurrent_key_dim * 4 +
       recurrent_value_dim * recurrent_key_dim * 6 +
       recurrent_value_dim * 3 + 8);
  assert(recurrent_stats.commands == 1 &&
         recurrent_stats.operations == recurrent_operations &&
         recurrent_stats.modeled_cycles == recurrent_operations);
  std::puts("functional_graph_xopennpux_external_recurrent=PASS");

  constexpr uint32_t attention_rows = 2;
  constexpr uint32_t attention_heads = 2;
  constexpr uint32_t attention_kv_heads = 1;
  constexpr uint32_t attention_head_dim = 2;
  constexpr uint32_t attention_kv_length = 2;
  std::fill(lhs_data, lhs_data + 8, 0.25f);
  std::fill(rhs_data, rhs_data + 8, 0.5f);
  std::fill(output_data, output_data + 8, 0.0f);
  opennpux_npu_functional_request attention = {};
  attention.magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  attention.version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  attention.struct_size = sizeof(attention);
  attention.opcode = OPENNPUX_NPU_OP_ATTENTION;
  attention.rows = attention_rows;
  attention.features = attention_heads * attention_head_dim;
  attention.heads = attention_heads;
  attention.kv_heads = attention_kv_heads;
  attention.head_dim = attention_head_dim;
  attention.kv_length = attention_kv_length;
  attention.operand_count = 3;
  attention.operands[0] = {OPENNPUX_NPU_OPERAND_INPUT, lhs->address,
                           UINT32_C(32), 0};
  attention.operands[1] = {OPENNPUX_NPU_OPERAND_SECONDARY, rhs->address,
                           UINT32_C(32), 0};
  attention.operands[2] = {OPENNPUX_NPU_OPERAND_OUTPUT, output->address,
                           UINT32_C(32), 0};
  opennpux_npu_operator_parameters attention_parameters = {};
  attention_parameters.magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  attention_parameters.version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  attention_parameters.struct_size = sizeof(attention_parameters);
  attention_parameters.opcode = OPENNPUX_NPU_OP_ATTENTION;
  Gem5HostXGraphExecutionStats attention_stats = {};
  const Gem5FunctionalMemoryRegion arena_region = {
      graph.arena().base(), graph.arena().data(), graph.arena().size()};
  assert(ExecuteGem5HostXGraphRequest(
             attention, attention_parameters, &arena_region, 1,
             &graph.arena(), &attention_stats) ==
         Gem5HostXGraphExecutionOutcome::kExecuted);
  for (uint32_t index = 0; index < 8; ++index) {
    assert(std::isfinite(output_data[index]));
  }
  const uint64_t visible_positions =
      attention_rows * (attention_kv_length - attention_rows + 1) +
      attention_rows * (attention_rows - 1) / 2;
  const uint64_t attention_operations =
      visible_positions * attention_heads * attention_head_dim * 4;
  assert(attention_stats.commands == 1 &&
         attention_stats.operations == attention_operations &&
         attention_stats.modeled_cycles == attention_operations);
  std::puts("functional_graph_xopennpux_attention=PASS");

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
  observer.count = 0;
  graph.SetRequestObserver(&observer);
  assert(graph.ExecuteRoutedExpert(expert_index, &weights));
  graph.SetRequestObserver(nullptr);
  assert(observer.count == 1 && observer.command_id == expert_index &&
         observer.regions == 2 &&
         observer.path == Gem5HostFunctionalExecutionPath::kGenericRequest);
  assert(graph.stats().routed_expert_commands == 1);
  assert(graph.stats().routed_expert_routes_issued ==
         graph.arena().runtime().batch_size *
             graph.arena().runtime().sequence_length *
             graph.arena().runtime().active_experts);
  assert(graph.stats().routed_expert_routes_completed ==
         graph.stats().routed_expert_routes_issued);
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
  const opennpux_npu_tensor_plan_runtime decode_runtime = {1, 3, 3, 2};
  const uint32_t decode_tokens[] = {0, 1, 2};
  assert(graph.ConfigureRuntime(submission, submission_size,
                                UINT32_C(0x24000000), decode_runtime));
  assert(graph.SetInputTokenIds(decode_tokens, 3));
  assert(graph.arena().runtime().sequence_length == 3);
  uint32_t linear_projection_index = UINT32_MAX;
  for (uint32_t index = 0; index < graph.command_count(); ++index) {
    std::vector<Gem5HostWeightBinding> floating;
    if (weights.FindFloatBindings(index, &floating) && floating.size() == 3) {
      linear_projection_index = index;
      break;
    }
  }
  assert(linear_projection_index != UINT32_MAX);
  assert(graph.ExecuteCommand(linear_projection_index, &weights));
  uint32_t linear_recurrent_index = UINT32_MAX;
  for (uint32_t index = 0; index < graph.command_count(); ++index) {
    std::vector<Gem5HostWeightBinding> floating;
    if (weights.FindFloatBindings(index, &floating) && floating.size() == 2 &&
        std::all_of(floating.begin(), floating.end(), [](const auto& binding) {
          return binding.role_id ==
                 OPENNPUX_NPU_WEIGHT_ROLE_LINEAR_DECAY;
        })) {
      linear_recurrent_index = index;
      break;
    }
  }
  assert(linear_recurrent_index != UINT32_MAX);
  assert(graph.ExecuteCommand(linear_recurrent_index, &weights));
  uint32_t linear_gate_norm_index = UINT32_MAX;
  for (uint32_t index = 0; index < graph.command_count(); ++index) {
    std::vector<Gem5HostWeightBinding> floating;
    const bool gate_norm = weights.FindFloatBindings(index, &floating) &&
        floating.size() == 2 &&
        std::any_of(floating.begin(), floating.end(), [](const auto& binding) {
          return binding.role_id == OPENNPUX_NPU_WEIGHT_ROLE_LINEAR_GATE;
        }) &&
        std::any_of(floating.begin(), floating.end(), [](const auto& binding) {
          return binding.role_id == OPENNPUX_NPU_WEIGHT_ROLE_LINEAR_NORM;
        });
    if (gate_norm) {
      linear_gate_norm_index = index;
      break;
    }
  }
  assert(linear_gate_norm_index != UINT32_MAX);
  assert(graph.ExecuteCommand(linear_gate_norm_index, &weights));
  std::vector<float> linear_gate_projection;
  assert(graph.ComputeLinearAttentionGateProjection(
      linear_gate_norm_index, &weights, &linear_gate_projection));
  assert(!linear_gate_projection.empty());
  assert(std::all_of(linear_gate_projection.begin(),
                     linear_gate_projection.end(),
                     [](float value) { return std::isfinite(value); }));
  uint32_t shared_expert_index = UINT32_MAX;
  for (uint32_t index = 0; index < graph.command_count(); ++index) {
    std::vector<Gem5HostWeightBinding> floating;
    if (!weights.FindFloatBindings(index, &floating) ||
        floating.size() != 4) {
      continue;
    }
    const bool shared =
        std::all_of(floating.begin(), floating.end(), [](const auto& binding) {
          return binding.role_id ==
                 OPENNPUX_NPU_WEIGHT_ROLE_SHARED_EXPERT;
        });
    if (shared) {
      shared_expert_index = index;
      break;
    }
  }
  assert(shared_expert_index != UINT32_MAX);
  assert(graph.ExecuteCommand(shared_expert_index, &weights));
  opennpux_npu_functional_request shared_expert = {};
  assert(graph.Materialize(shared_expert_index, nullptr, 0, &shared_expert));
  const auto* shared_output =
      FindOperand(shared_expert, OPENNPUX_NPU_OPERAND_OUTPUT);
  assert(shared_output != nullptr);
  const auto* shared_output_data = reinterpret_cast<const float*>(
      graph.arena().Translate(shared_output->address,
                              shared_output->byte_size));
  assert(shared_output_data != nullptr);
  for (size_t index = 0; index < shared_output->byte_size / sizeof(float);
       ++index) {
    assert(std::isfinite(shared_output_data[index]));
  }
  std::printf("functional_graph_add_elements=%zu\n", count);
  std::puts("functional_graph_gptq_projection=PASS");
  std::puts("functional_graph_routed_expert=PASS");
  std::puts("functional_graph_direct_expert=PASS");
  std::puts("functional_graph_gptq_router=PASS");
  std::puts("functional_graph_auto_dispatch=PASS");
  std::puts("functional_graph_program_failure_location=PASS");
  std::puts("functional_graph_token_io=PASS");
  std::puts("functional_graph_autoregressive_reconfigure=PASS");
  std::puts("functional_graph_linear_attention_projection=PASS");
  std::puts("functional_graph_linear_attention_recurrent=PASS");
  std::puts("functional_graph_linear_attention_gate_norm=PASS");
  std::puts("functional_graph_float_shared_expert=PASS");
  std::puts("gem5_host_functional_graph=PASS");
  std::free(submission);
  opennpux_npu_executable_unload(&executable);
  return 0;
}
