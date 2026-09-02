#include "hw_sim/gem5_bridge/gem5_host_functional_graph.h"
#include "hw_sim/gem5_bridge/gem5_host_xgraph_executor.h"
#include "hw_sim/gem5_bridge/gem5_host_weight_provider.h"
#include "hw_sim/gem5_bridge/gem5_transformer_kernels.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
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
         graph.stats().xgraph_opcodes[OPENNPUX_NPU_OP_ADD] == 1 &&
         graph.stats().xgraph_fallback_requests == 0);
  std::printf("functional_graph_xopennpux_add=PASS\n");

  assert(setenv("OPENNPUX_HOST_FUNCTIONAL_EXECUTION",
                "xopennpux-primitives", 1) == 0);
  assert(setenv("OPENNPUX_HOST_XGRAPH_OPCODE_MASK", "0", 1) == 0);
  assert(setenv("OPENNPUX_HOST_XGRAPH_REQUIRE_FULL", "1", 1) == 0);
  assert(!graph.Execute(&request));
  assert(unsetenv("OPENNPUX_HOST_XGRAPH_REQUIRE_FULL") == 0);
  assert(unsetenv("OPENNPUX_HOST_XGRAPH_OPCODE_MASK") == 0);
  assert(unsetenv("OPENNPUX_HOST_FUNCTIONAL_EXECUTION") == 0);
  assert(graph.stats().completed_commands == 1 &&
         graph.stats().xgraph_fallback_requests == 1 &&
         graph.stats().xgraph_fallback_opcodes[OPENNPUX_NPU_OP_ADD] == 1);
  std::puts("functional_graph_xopennpux_require_full=PASS");

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

  const uint32_t embed_indices[] = {2, 0};
  std::vector<float> embed_weights = {
      1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  std::memcpy(lhs_data, embed_indices, sizeof(embed_indices));
  std::fill(output_data, output_data + 4, 0.0f);
  opennpux_npu_functional_request embed = {};
  embed.magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  embed.version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  embed.struct_size = sizeof(embed);
  embed.opcode = OPENNPUX_NPU_OP_EMBED;
  embed.rows = 2;
  embed.features = 2;
  embed.operand_count = 3;
  embed.operands[0] = {OPENNPUX_NPU_OPERAND_INPUT_INDICES, lhs->address,
                       sizeof(embed_indices), 0};
  embed.operands[1] = {OPENNPUX_NPU_OPERAND_OUTPUT, output->address,
                       UINT32_C(16), 0};
  embed.operands[2] = {
      OPENNPUX_NPU_OPERAND_WEIGHT, UINT32_C(0x60500000),
      static_cast<uint32_t>(embed_weights.size() * sizeof(float)), 0};
  const Gem5FunctionalMemoryRegion embed_region = {
      UINT32_C(0x60500000),
      reinterpret_cast<uint8_t*>(embed_weights.data()),
      embed_weights.size() * sizeof(float)};
  opennpux_npu_operator_parameters embed_parameters = {};
  embed_parameters.magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  embed_parameters.version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  embed_parameters.struct_size = sizeof(embed_parameters);
  embed_parameters.opcode = OPENNPUX_NPU_OP_EMBED;
  embed_parameters.input_features = 3;
  Gem5HostXGraphExecutionStats embed_stats = {};
  assert(ExecuteGem5HostXGraphRequest(
             embed, embed_parameters, &embed_region, 1, &graph.arena(),
             &embed_stats) == Gem5HostXGraphExecutionOutcome::kExecuted);
  assert(output_data[0] == 5.0f && output_data[1] == 6.0f &&
         output_data[2] == 1.0f && output_data[3] == 2.0f);
  assert(embed_stats.commands == 1 && embed_stats.operations == 4 &&
         embed_stats.modeled_cycles == 4);
  std::puts("functional_graph_xopennpux_external_embed=PASS");

  const float topk_input_values[] = {1.0f, 9.0f, 3.0f,
                                     8.0f, 2.0f, 7.0f};
  std::copy(topk_input_values, topk_input_values + 6, lhs_data);
  auto* topk_indices = reinterpret_cast<uint32_t*>(rhs_data);
  topk_indices[0] = UINT32_MAX;
  topk_indices[1] = UINT32_MAX;
  opennpux_npu_functional_request indices_topk = {};
  indices_topk.magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  indices_topk.version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  indices_topk.struct_size = sizeof(indices_topk);
  indices_topk.opcode = OPENNPUX_NPU_OP_TOPK;
  indices_topk.rows = 2;
  indices_topk.features = 3;
  indices_topk.top_k = 1;
  indices_topk.operand_count = 2;
  indices_topk.operands[0] = {OPENNPUX_NPU_OPERAND_INPUT, lhs->address,
                              UINT32_C(24), 0};
  indices_topk.operands[1] = {OPENNPUX_NPU_OPERAND_OUTPUT_INDICES,
                              rhs->address, UINT32_C(8), 0};
  opennpux_npu_operator_parameters topk_parameters = {};
  topk_parameters.magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  topk_parameters.version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  topk_parameters.struct_size = sizeof(topk_parameters);
  topk_parameters.opcode = OPENNPUX_NPU_OP_TOPK;
  const Gem5FunctionalMemoryRegion topk_region = {
      graph.arena().base(), graph.arena().data(), graph.arena().size()};
  Gem5HostXGraphExecutionStats topk_stats = {};
  assert(ExecuteGem5HostXGraphRequest(
             indices_topk, topk_parameters, &topk_region, 1, &graph.arena(),
             &topk_stats) == Gem5HostXGraphExecutionOutcome::kExecuted);
  assert(topk_indices[0] == 1 && topk_indices[1] == 0);
  assert(topk_stats.commands == 1 && topk_stats.operations == 6 &&
         topk_stats.modeled_cycles == 6);
  std::puts("functional_graph_xopennpux_indices_only_topk=PASS");

  constexpr uint32_t router_rows = 2;
  constexpr uint32_t router_input_features = 2048;
  constexpr uint32_t router_experts = 256;
  constexpr uint32_t router_top_k = 8;
  std::vector<float> large_router_input(router_rows * router_input_features);
  std::vector<float> large_router_weights(
      router_input_features * router_experts);
  uint32_t random_state = UINT32_C(0x4f50454e);
  for (float& value : large_router_input) {
    random_state = random_state * UINT32_C(1664525) + UINT32_C(1013904223);
    value = static_cast<float>(static_cast<int32_t>(random_state >> 16) -
                               INT32_C(32768)) /
            32768.0f;
  }
  // Router weights use the same [output_features, input_features] layout as
  // RunGem5MatMulF32 and the generic functional-request ABI.
  for (uint32_t expert = 0; expert < router_experts; ++expert) {
    for (uint32_t feature = 0; feature < router_input_features; ++feature) {
      random_state = random_state * UINT32_C(1664525) + UINT32_C(1013904223);
      large_router_weights[expert * router_input_features + feature] =
          static_cast<float>(static_cast<int32_t>(random_state >> 16) -
                             INT32_C(32768)) /
          65536.0f;
    }
  }
  std::vector<float> reference_logits(router_rows * router_experts);
  std::vector<float> reference_weights(router_rows * router_top_k);
  std::vector<uint32_t> reference_ids(router_rows * router_top_k);
  Gem5TransformerKernelStats reference_matmul_stats = {};
  assert(RunGem5MatMulF32(
      large_router_input.data(), large_router_weights.data(), router_rows,
      router_input_features, router_experts, reference_logits.data(),
      &reference_matmul_stats));
  for (uint32_t row = 0; row < router_rows; ++row) {
    Gem5TransformerKernelStats reference_topk_stats = {};
    float* row_weights = reference_weights.data() + row * router_top_k;
    assert(RunGem5TopKF32(
        reference_logits.data() + row * router_experts, router_experts,
        router_top_k, row_weights,
        reference_ids.data() + row * router_top_k, &reference_topk_stats));
    const float maximum = row_weights[0];
    double sum = 0.0;
    for (uint32_t route = 0; route < router_top_k; ++route) {
      row_weights[route] = std::exp(row_weights[route] - maximum);
      sum += row_weights[route];
    }
    for (uint32_t route = 0; route < router_top_k; ++route) {
      row_weights[route] /= static_cast<float>(sum);
    }
  }
  std::fill(output_data, output_data + router_rows * router_top_k, 0.0f);
  auto* large_router_ids = reinterpret_cast<uint32_t*>(rhs_data);
  std::fill(large_router_ids,
            large_router_ids + router_rows * router_top_k, UINT32_MAX);
  opennpux_npu_functional_request large_router = {};
  large_router.magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  large_router.version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  large_router.struct_size = sizeof(large_router);
  large_router.opcode = OPENNPUX_NPU_OP_ROUTER;
  large_router.rows = router_rows;
  large_router.features = router_top_k;
  large_router.top_k = router_top_k;
  large_router.operand_count = 4;
  large_router.operands[0] = {
      OPENNPUX_NPU_OPERAND_INPUT, UINT32_C(0x60500000),
      router_rows * router_input_features * sizeof(float), 0};
  large_router.operands[1] = {
      OPENNPUX_NPU_OPERAND_OUTPUT, output->address,
      router_rows * router_top_k * sizeof(float), 0};
  large_router.operands[2] = {
      OPENNPUX_NPU_OPERAND_OUTPUT_INDICES, rhs->address,
      router_rows * router_top_k * sizeof(uint32_t), 0};
  large_router.operands[3] = {
      OPENNPUX_NPU_OPERAND_WEIGHT, UINT32_C(0x60600000),
      static_cast<uint32_t>(large_router_weights.size() * sizeof(float)), 0};
  const Gem5FunctionalMemoryRegion large_router_regions[] = {
      {UINT32_C(0x60500000),
       reinterpret_cast<uint8_t*>(large_router_input.data()),
       large_router_input.size() * sizeof(float)},
      {UINT32_C(0x60600000),
       reinterpret_cast<uint8_t*>(large_router_weights.data()),
       large_router_weights.size() * sizeof(float)}};
  opennpux_npu_operator_parameters large_router_parameters = {};
  large_router_parameters.magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  large_router_parameters.version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  large_router_parameters.struct_size = sizeof(large_router_parameters);
  large_router_parameters.opcode = OPENNPUX_NPU_OP_ROUTER;
  large_router_parameters.input_features = router_input_features;
  large_router_parameters.output_features = router_experts;
  Gem5HostXGraphExecutionStats large_router_stats = {};
  assert(ExecuteGem5HostXGraphRequest(
             large_router, large_router_parameters, large_router_regions, 2,
             &graph.arena(), &large_router_stats) ==
         Gem5HostXGraphExecutionOutcome::kExecuted);
  assert(large_router_stats.commands > 5);
  for (uint32_t row = 0; row < router_rows; ++row) {
    float sum = 0.0f;
    for (uint32_t route = 0; route < router_top_k; ++route) {
      const uint32_t index = row * router_top_k + route;
      assert(large_router_ids[index] == reference_ids[index]);
      assert(std::fabs(output_data[index] - reference_weights[index]) <
             1.0e-5f);
      sum += output_data[index];
    }
    assert(std::fabs(sum - 1.0f) < 1.0e-5f);
  }
  std::puts("functional_graph_xopennpux_tiled_float_router=PASS");

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

  Gem5HostFunctionalGraph shadow_graph;
  assert(shadow_graph.LoadTensorPlan(argv[2]));
  assert(shadow_graph.Configure(submission, submission_size,
                                UINT32_C(0x24000000)));
  auto* shadow_lhs = reinterpret_cast<float*>(
      shadow_graph.arena().Translate(lhs->address, lhs->byte_size));
  auto* shadow_rhs = reinterpret_cast<float*>(
      shadow_graph.arena().Translate(rhs->address, rhs->byte_size));
  auto* shadow_output = reinterpret_cast<float*>(
      shadow_graph.arena().Translate(output->address, output->byte_size));
  assert(shadow_lhs != nullptr && shadow_rhs != nullptr &&
         shadow_output != nullptr);
  std::copy(recurrent_qkv, recurrent_qkv + 12, shadow_lhs);
  std::fill(shadow_lhs + 12, shadow_lhs + 16, 0.5f);
  std::fill(shadow_rhs, shadow_rhs + 4, 0.0f);
  std::fill(shadow_output, shadow_output + 4, 0.0f);
  const std::vector<uint8_t> shadow_initial(
      shadow_graph.arena().data(),
      shadow_graph.arena().data() + shadow_graph.arena().size());
  constexpr uint32_t kRecurrentParameterAddress = UINT32_C(0x62000200);
  const Gem5FunctionalMemoryRegion shadow_regions[] = {
      recurrent_weight_regions[0], recurrent_weight_regions[1],
      {kRecurrentParameterAddress,
       reinterpret_cast<uint8_t*>(&recurrent_parameters),
       sizeof(recurrent_parameters)}};
  opennpux_npu_functional_request shadow_recurrent = recurrent;
  shadow_recurrent.parameter_address = kRecurrentParameterAddress;
  shadow_recurrent.parameter_size = sizeof(recurrent_parameters);
  assert(setenv("OPENNPUX_HOST_FUNCTIONAL_EXECUTION",
                "xopennpux-primitives", 1) == 0);
  assert(unsetenv("OPENNPUX_HOST_XGRAPH_SHADOW_COMPARE") == 0);
  assert(shadow_graph.Execute(&shadow_recurrent, shadow_regions, 3));
  const std::vector<uint8_t> expected_xgraph_arena(
      shadow_graph.arena().data(),
      shadow_graph.arena().data() + shadow_graph.arena().size());

  std::memcpy(shadow_graph.arena().data(), shadow_initial.data(),
              shadow_initial.size());
  shadow_recurrent = recurrent;
  shadow_recurrent.parameter_address = kRecurrentParameterAddress;
  shadow_recurrent.parameter_size = sizeof(recurrent_parameters);
  assert(setenv("OPENNPUX_HOST_XGRAPH_SHADOW_COMPARE", "all", 1) == 0);
  assert(shadow_graph.Execute(&shadow_recurrent, shadow_regions, 3));
  assert(std::equal(expected_xgraph_arena.begin(),
                    expected_xgraph_arena.end(),
                    shadow_graph.arena().data()));
  assert(unsetenv("OPENNPUX_HOST_XGRAPH_SHADOW_COMPARE") == 0);
  assert(unsetenv("OPENNPUX_HOST_FUNCTIONAL_EXECUTION") == 0);
  std::puts("functional_graph_xopennpux_shadow_state=PASS");

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

  constexpr uint32_t dma_rows = 2;
  constexpr uint32_t dma_kv_heads = 1;
  constexpr uint32_t dma_head_dim = 2;
  constexpr uint32_t dma_kv_length = 4;
  const float dma_key[] = {1.0f, 2.0f, 3.0f, 4.0f};
  const float dma_value[] = {5.0f, 6.0f, 7.0f, 8.0f};
  std::copy(dma_key, dma_key + 4, lhs_data);
  std::copy(dma_value, dma_value + 4, lhs_data + 4);
  std::fill(output_data, output_data + 16, 0.0f);
  opennpux_npu_functional_request dma = {};
  dma.magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  dma.version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  dma.struct_size = sizeof(dma);
  dma.opcode = OPENNPUX_NPU_OP_DMA;
  dma.rows = dma_rows;
  dma.features = dma_kv_heads * dma_head_dim;
  dma.kv_heads = dma_kv_heads;
  dma.head_dim = dma_head_dim;
  dma.kv_length = dma_kv_length;
  dma.operand_count = 3;
  dma.operands[0] = {OPENNPUX_NPU_OPERAND_INPUT, lhs->address,
                     UINT32_C(16), 0};
  dma.operands[1] = {OPENNPUX_NPU_OPERAND_SECONDARY,
                     lhs->address + UINT32_C(16), UINT32_C(16), 0};
  dma.operands[2] = {OPENNPUX_NPU_OPERAND_OUTPUT, output->address,
                     UINT32_C(64), 0};
  opennpux_npu_operator_parameters dma_parameters = {};
  dma_parameters.magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  dma_parameters.version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  dma_parameters.struct_size = sizeof(dma_parameters);
  dma_parameters.opcode = OPENNPUX_NPU_OP_DMA;
  Gem5HostXGraphExecutionStats dma_stats = {};
  assert(ExecuteGem5HostXGraphRequest(
             dma, dma_parameters, &arena_region, 1, &graph.arena(),
             &dma_stats) == Gem5HostXGraphExecutionOutcome::kExecuted);
  for (uint32_t index = 0; index < 4; ++index) {
    assert(output_data[4 + index] == dma_key[index]);
    assert(output_data[12 + index] == dma_value[index]);
  }
  assert(dma_stats.commands == 2 && dma_stats.operations == 8 &&
         dma_stats.modeled_cycles == 8);
  std::puts("functional_graph_xopennpux_kv_dma=PASS");

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
  const auto matmul_stats_before = graph.stats();
  assert(setenv("OPENNPUX_HOST_FUNCTIONAL_EXECUTION",
                "xopennpux-primitives", 1) == 0);
  assert(setenv("OPENNPUX_HOST_XGRAPH_GPTQ_MATMUL_SCOPE", "all", 1) == 0);
  assert(graph.ExecuteGptqQkv(matmul_index, &weights));
  assert(unsetenv("OPENNPUX_HOST_XGRAPH_GPTQ_MATMUL_SCOPE") == 0);
  assert(unsetenv("OPENNPUX_HOST_FUNCTIONAL_EXECUTION") == 0);
  assert(graph.stats().xgraph_requests ==
             matmul_stats_before.xgraph_requests + 1 &&
         graph.stats().xgraph_commands > matmul_stats_before.xgraph_commands &&
         graph.stats().xgraph_operations >
             matmul_stats_before.xgraph_operations &&
         graph.stats().xgraph_fallback_requests ==
             matmul_stats_before.xgraph_fallback_requests &&
         graph.stats().xgraph_opcodes[OPENNPUX_NPU_OP_MATMUL] ==
             matmul_stats_before.xgraph_opcodes[OPENNPUX_NPU_OP_MATMUL] + 1);
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
  const auto router_stats_before = graph.stats();
  assert(setenv("OPENNPUX_HOST_FUNCTIONAL_EXECUTION",
                "xopennpux-primitives", 1) == 0);
  assert(graph.ExecuteGptqRouter(router_index, &weights));
  assert(unsetenv("OPENNPUX_HOST_FUNCTIONAL_EXECUTION") == 0);
  assert(graph.stats().xgraph_requests ==
             router_stats_before.xgraph_requests + 1 &&
         graph.stats().xgraph_commands > router_stats_before.xgraph_commands &&
         graph.stats().xgraph_operations >
             router_stats_before.xgraph_operations &&
         graph.stats().xgraph_fallback_requests ==
             router_stats_before.xgraph_fallback_requests &&
         graph.stats().xgraph_opcodes[OPENNPUX_NPU_OP_ROUTER] ==
             router_stats_before.xgraph_opcodes[OPENNPUX_NPU_OP_ROUTER] + 1);
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
  opennpux_npu_functional_request rope_request = {};
  assert(graph.Materialize(rope_index, nullptr, 0, &rope_request));
  const auto rope_stats_before = graph.stats();
  assert(setenv("OPENNPUX_HOST_FUNCTIONAL_EXECUTION",
                "xopennpux-primitives", 1) == 0);
  assert(setenv("OPENNPUX_HOST_XGRAPH_SHADOW_COMPARE", "all", 1) == 0);
  assert(graph.ExecuteCommand(rope_index, &weights));
  assert(unsetenv("OPENNPUX_HOST_XGRAPH_SHADOW_COMPARE") == 0);
  assert(unsetenv("OPENNPUX_HOST_FUNCTIONAL_EXECUTION") == 0);
  assert(graph.stats().xgraph_requests ==
             rope_stats_before.xgraph_requests + 1 &&
         graph.stats().xgraph_commands ==
             rope_stats_before.xgraph_commands +
                 static_cast<uint64_t>(rope_request.rows) *
                     (rope_request.heads + rope_request.kv_heads) * 2 &&
         graph.stats().xgraph_fallback_requests ==
             rope_stats_before.xgraph_fallback_requests);
  std::puts("functional_graph_xopennpux_multihead_rope=PASS");
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
  opennpux_npu_functional_request linear_gate_norm = {};
  assert(graph.Materialize(linear_gate_norm_index, nullptr, 0,
                           &linear_gate_norm));
  const auto* linear_gate_norm_output =
      FindOperand(linear_gate_norm, OPENNPUX_NPU_OPERAND_OUTPUT);
  assert(linear_gate_norm_output != nullptr);
  auto* linear_gate_norm_data = reinterpret_cast<float*>(
      graph.arena().Translate(linear_gate_norm_output->address,
                              linear_gate_norm_output->byte_size));
  assert(linear_gate_norm_data != nullptr);
  const size_t linear_gate_norm_count =
      linear_gate_norm_output->byte_size / sizeof(float);
  const std::vector<float> linear_gate_norm_reference(
      linear_gate_norm_data, linear_gate_norm_data + linear_gate_norm_count);
  const uint64_t gated_commands_before = graph.stats().xgraph_commands;
  assert(setenv("OPENNPUX_HOST_FUNCTIONAL_EXECUTION",
                "xopennpux-primitives", 1) == 0);
  assert(setenv("OPENNPUX_HOST_XGRAPH_NORMALIZE_SCOPE", "gated", 1) == 0);
  assert(graph.ExecuteCommand(linear_gate_norm_index, &weights));
  assert(unsetenv("OPENNPUX_HOST_XGRAPH_NORMALIZE_SCOPE") == 0);
  assert(unsetenv("OPENNPUX_HOST_FUNCTIONAL_EXECUTION") == 0);
  assert(graph.stats().xgraph_commands > gated_commands_before + 2);
  for (size_t index = 0; index < linear_gate_norm_count; ++index) {
    assert(linear_gate_norm_data[index] == linear_gate_norm_reference[index]);
  }
  opennpux_npu_functional_request large_gated_norm = {};
  large_gated_norm.magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  large_gated_norm.version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  large_gated_norm.struct_size = sizeof(large_gated_norm);
  large_gated_norm.opcode = OPENNPUX_NPU_OP_NORMALIZE;
  large_gated_norm.rows = 18;
  large_gated_norm.features = 4096;
  large_gated_norm.epsilon = 1.0e-6f;
  large_gated_norm.operand_count = 5;
  large_gated_norm.operands[0] = {
      OPENNPUX_NPU_OPERAND_INPUT, UINT32_C(0x60000000), 18 * 4096 * 4, 0};
  large_gated_norm.operands[1] = {
      OPENNPUX_NPU_OPERAND_SECONDARY, UINT32_C(0x60100000), 18 * 2048 * 4, 0};
  large_gated_norm.operands[2] = {
      OPENNPUX_NPU_OPERAND_OUTPUT, UINT32_C(0x60200000), 18 * 4096 * 4, 0};
  large_gated_norm.operands[3] = {
      OPENNPUX_NPU_OPERAND_LINEAR_GATE_WEIGHT, UINT32_C(0x61000000),
      2048 * 4096 * 4, 0};
  large_gated_norm.operands[4] = {
      OPENNPUX_NPU_OPERAND_LINEAR_NORM_WEIGHT, UINT32_C(0x63000000),
      128 * 4, 0};
  opennpux_npu_operator_parameters large_gated_parameters = {};
  large_gated_parameters.magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  large_gated_parameters.version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  large_gated_parameters.struct_size = sizeof(large_gated_parameters);
  large_gated_parameters.opcode = OPENNPUX_NPU_OP_NORMALIZE;
  large_gated_parameters.flags =
      OPENNPUX_NPU_PARAMETER_GATED_DELTA_NET |
      OPENNPUX_NPU_PARAMETER_BFLOAT16_INTERMEDIATE;
  large_gated_parameters.input_features = 2048;
  large_gated_parameters.output_features = 4096;
  std::vector<opennpux_xgraph_command> large_gated_commands(
      OPENNPUX_XGRAPH_MAX_COMMANDS);
  uint32_t large_gated_command_count = 0;
  assert(opennpux_npu_xgraph_lower_gated_normalize(
             &large_gated_norm, &large_gated_parameters,
             UINT32_C(0x60000000), UINT32_C(0x08000000),
             UINT32_C(0x64000000), UINT32_C(0x00100000), 0,
             large_gated_commands.data(), large_gated_commands.size(),
             &large_gated_command_count) == 0);
  assert(large_gated_command_count > 18 * 32 + 3);
  for (uint32_t index = 0; index < large_gated_command_count; ++index) {
    const auto& command = large_gated_commands[index];
    if (command.opcode == OPENNPUX_XGRAPH_OP_TMMA) {
      assert(command.dim0 <= 1023 && command.dim1 <= 1023 &&
             command.dim2 <= 1023);
    }
  }
  assert(large_gated_commands[large_gated_command_count - 2].opcode ==
         OPENNPUX_XGRAPH_OP_TSILU);
  assert(large_gated_commands[large_gated_command_count - 1].opcode ==
         OPENNPUX_XGRAPH_OP_TMUL);
  constexpr uint32_t kTiledInputFeatures = 1024;
  constexpr uint32_t kTiledOutputFeatures = 4;
  std::vector<float> tiled_projection_input(kTiledInputFeatures);
  std::vector<float> tiled_gate_weight(
      kTiledInputFeatures * kTiledOutputFeatures);
  std::vector<float> tiled_norm_weight = {0.75f, 1.25f};
  for (uint32_t index = 0; index < kTiledInputFeatures; ++index) {
    tiled_projection_input[index] =
        static_cast<float>(static_cast<int32_t>(index % 17) - 8) / 32.0f;
  }
  for (uint32_t index = 0; index < tiled_gate_weight.size(); ++index) {
    tiled_gate_weight[index] =
        static_cast<float>(static_cast<int32_t>(index % 13) - 6) / 64.0f;
  }
  const float tiled_norm_input[] = {0.75f, -1.25f, 2.0f, -0.5f};
  auto* tiled_norm_data = reinterpret_cast<float*>(
      graph.arena().Translate(lhs->address,
                              kTiledOutputFeatures * sizeof(float)));
  auto* tiled_output_data = reinterpret_cast<float*>(
      graph.arena().Translate(output->address,
                              kTiledOutputFeatures * sizeof(float)));
  assert(tiled_norm_data != nullptr && tiled_output_data != nullptr);
  std::copy(std::begin(tiled_norm_input), std::end(tiled_norm_input),
            tiled_norm_data);
  std::fill(tiled_output_data,
            tiled_output_data + kTiledOutputFeatures, 0.0f);
  opennpux_npu_functional_request tiled_gated_norm = {};
  tiled_gated_norm.magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  tiled_gated_norm.version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  tiled_gated_norm.struct_size = sizeof(tiled_gated_norm);
  tiled_gated_norm.opcode = OPENNPUX_NPU_OP_NORMALIZE;
  tiled_gated_norm.rows = 1;
  tiled_gated_norm.features = kTiledOutputFeatures;
  tiled_gated_norm.epsilon = 1.0e-6f;
  tiled_gated_norm.operand_count = 5;
  tiled_gated_norm.operands[0] = {
      OPENNPUX_NPU_OPERAND_INPUT, lhs->address,
      kTiledOutputFeatures * sizeof(float), 0};
  tiled_gated_norm.operands[1] = {
      OPENNPUX_NPU_OPERAND_SECONDARY, UINT32_C(0x60000000),
      kTiledInputFeatures * sizeof(float), 0};
  tiled_gated_norm.operands[2] = {
      OPENNPUX_NPU_OPERAND_OUTPUT, output->address,
      kTiledOutputFeatures * sizeof(float), 0};
  tiled_gated_norm.operands[3] = {
      OPENNPUX_NPU_OPERAND_LINEAR_GATE_WEIGHT, UINT32_C(0x60010000),
      kTiledInputFeatures * kTiledOutputFeatures * sizeof(float), 0};
  tiled_gated_norm.operands[4] = {
      OPENNPUX_NPU_OPERAND_LINEAR_NORM_WEIGHT, UINT32_C(0x60020000),
      static_cast<uint32_t>(tiled_norm_weight.size() * sizeof(float)), 0};
  opennpux_npu_operator_parameters tiled_gated_parameters = {};
  tiled_gated_parameters.magic = OPENNPUX_NPU_OPERATOR_PARAMETERS_MAGIC;
  tiled_gated_parameters.version = OPENNPUX_NPU_OPERATOR_PARAMETERS_VERSION;
  tiled_gated_parameters.struct_size = sizeof(tiled_gated_parameters);
  tiled_gated_parameters.opcode = OPENNPUX_NPU_OP_NORMALIZE;
  tiled_gated_parameters.flags =
      OPENNPUX_NPU_PARAMETER_NORM_WEIGHT_OFFSET |
      OPENNPUX_NPU_PARAMETER_GATED_DELTA_NET |
      OPENNPUX_NPU_PARAMETER_BFLOAT16_INTERMEDIATE;
  tiled_gated_parameters.input_features = kTiledInputFeatures;
  tiled_gated_parameters.output_features = kTiledOutputFeatures;
  tiled_gated_parameters.intermediate_features = tiled_norm_weight.size();
  const Gem5FunctionalMemoryRegion tiled_regions[] = {
      {UINT32_C(0x60000000),
       reinterpret_cast<uint8_t*>(tiled_projection_input.data()),
       tiled_projection_input.size() * sizeof(float)},
      {UINT32_C(0x60010000),
       reinterpret_cast<uint8_t*>(tiled_gate_weight.data()),
       tiled_gate_weight.size() * sizeof(float)},
      {UINT32_C(0x60020000),
       reinterpret_cast<uint8_t*>(tiled_norm_weight.data()),
       tiled_norm_weight.size() * sizeof(float)},
  };
  Gem5HostXGraphExecutionStats tiled_gated_stats = {};
  opennpux_npu_functional_request tiled_projection = {};
  tiled_projection.magic = OPENNPUX_NPU_FUNCTIONAL_MAGIC;
  tiled_projection.version = OPENNPUX_NPU_FUNCTIONAL_VERSION;
  tiled_projection.struct_size = sizeof(tiled_projection);
  tiled_projection.opcode = OPENNPUX_NPU_OP_MATMUL;
  tiled_projection.rows = 1;
  tiled_projection.features = kTiledOutputFeatures;
  tiled_projection.operand_count = 3;
  tiled_projection.operands[0] = {
      OPENNPUX_NPU_OPERAND_INPUT, UINT32_C(0x60000000),
      kTiledInputFeatures * sizeof(float), 0};
  tiled_projection.operands[1] = {
      OPENNPUX_NPU_OPERAND_OUTPUT, output->address,
      kTiledOutputFeatures * sizeof(float), 0};
  tiled_projection.operands[2] = {
      OPENNPUX_NPU_OPERAND_WEIGHT, UINT32_C(0x60010000),
      kTiledInputFeatures * kTiledOutputFeatures * sizeof(float), 0};
  opennpux_npu_operator_parameters tiled_projection_parameters =
      tiled_gated_parameters;
  tiled_projection_parameters.opcode = OPENNPUX_NPU_OP_MATMUL;
  tiled_projection_parameters.flags = 0;
  Gem5HostXGraphExecutionStats tiled_projection_xgraph_stats = {};
  assert(ExecuteGem5HostXGraphRequest(
             tiled_projection, tiled_projection_parameters, tiled_regions, 2,
             &graph.arena(), &tiled_projection_xgraph_stats) ==
         Gem5HostXGraphExecutionOutcome::kExecuted);
  std::vector<float> tiled_gate(kTiledOutputFeatures);
  Gem5TransformerKernelStats tiled_projection_stats = {};
  assert(RunGem5MatMulF32(
      tiled_projection_input.data(), tiled_gate_weight.data(), 1,
      kTiledInputFeatures, kTiledOutputFeatures, tiled_gate.data(),
      &tiled_projection_stats));
  for (uint32_t index = 0; index < kTiledOutputFeatures; ++index) {
    assert(tiled_output_data[index] == tiled_gate[index]);
  }
  std::fill(tiled_output_data,
            tiled_output_data + kTiledOutputFeatures, 0.0f);
  assert(ExecuteGem5HostXGraphRequest(
             tiled_gated_norm, tiled_gated_parameters, tiled_regions,
             std::size(tiled_regions), &graph.arena(), &tiled_gated_stats) ==
         Gem5HostXGraphExecutionOutcome::kExecuted);
  std::vector<float> tiled_expected(kTiledOutputFeatures);
  Gem5TransformerKernelStats tiled_norm_stats = {};
  assert(RunGem5GatedRmsNormF32(
      tiled_norm_input, tiled_gate.data(), tiled_norm_weight.data(), 1, 2, 2,
      tiled_gated_norm.epsilon, tiled_expected.data(), &tiled_norm_stats,
      true));
  assert(tiled_gated_stats.commands == 6);
  for (uint32_t index = 0; index < kTiledOutputFeatures; ++index) {
    assert(tiled_output_data[index] == tiled_expected[index]);
  }
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
  std::puts("functional_graph_xopennpux_gated_norm=PASS");
  std::puts("functional_graph_xopennpux_large_gated_norm_lowering=PASS");
  std::puts("functional_graph_xopennpux_tiled_gated_norm=PASS");
  std::puts("functional_graph_float_shared_expert=PASS");
  std::puts("gem5_host_functional_graph=PASS");
  std::free(submission);
  opennpux_npu_executable_unload(&executable);
  return 0;
}
