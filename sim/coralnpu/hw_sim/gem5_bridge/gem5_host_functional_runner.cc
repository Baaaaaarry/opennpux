#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "hw_sim/gem5_bridge/gem5_host_functional_graph.h"
#include "hw_sim/gem5_bridge/gem5_host_weight_provider.h"
#include "opennpux/npu_executable.h"

namespace {

bool ParseTokens(const char* text, std::vector<uint32_t>* tokens) {
  if (text == nullptr || text[0] == '\0' || tokens == nullptr) {
    return false;
  }
  const char* cursor = text;
  while (*cursor != '\0') {
    while (std::isspace(static_cast<unsigned char>(*cursor))) {
      ++cursor;
    }
    if (*cursor == '\0') {
      break;
    }
    char* end = nullptr;
    errno = 0;
    const unsigned long value = std::strtoul(cursor, &end, 10);
    if (errno != 0 || end == cursor || value > UINT32_MAX) {
      return false;
    }
    while (std::isspace(static_cast<unsigned char>(*end))) {
      ++end;
    }
    if (*end != '\0' && *end != ',') {
      return false;
    }
    tokens->push_back(static_cast<uint32_t>(value));
    if (*end == ',') {
      cursor = end + 1;
      if (*cursor == '\0') {
        return false;
      }
    } else {
      cursor = end;
    }
  }
  return !tokens->empty();
}

bool ParseCount(const char* text, uint32_t* count) {
  if (text == nullptr || count == nullptr) {
    return false;
  }
  while (std::isspace(static_cast<unsigned char>(*text))) {
    ++text;
  }
  char* end = nullptr;
  errno = 0;
  const unsigned long value = std::strtoul(text, &end, 10);
  while (std::isspace(static_cast<unsigned char>(*end))) {
    ++end;
  }
  if (errno != 0 || end == text || *end != '\0' || value == 0 || value > 32) {
    return false;
  }
  *count = static_cast<uint32_t>(value);
  return true;
}

bool TraceEnabled() {
  const char* value = std::getenv("OPENNPUX_HOST_FUNCTIONAL_TRACE");
  if (value == nullptr || value[0] == '\0') {
    value = std::getenv("CORAL_HOST_FUNCTIONAL_TRACE");
  }
  return value != nullptr && value[0] != '\0' &&
         std::strcmp(value, "0") != 0;
}

uint32_t TraceStep() {
  const char* value = std::getenv("OPENNPUX_HOST_FUNCTIONAL_TRACE_STEP");
  uint32_t step = 0;
  return value != nullptr && ParseCount(value, &step) ? step : 0;
}

bool LogitsTraceEnabled() {
  const char* value =
      std::getenv("OPENNPUX_HOST_FUNCTIONAL_LOGITS_TRACE");
  return TraceEnabled() ||
         (value != nullptr && value[0] != '\0' &&
          std::strcmp(value, "0") != 0);
}

bool ProgressEnabled() {
  const char* value = std::getenv("OPENNPUX_HOST_FUNCTIONAL_PROGRESS");
  return TraceEnabled() ||
         (value != nullptr && value[0] != '\0' &&
          std::strcmp(value, "0") != 0);
}

bool TeacherForcingEnabled() {
  const char* value =
      std::getenv("OPENNPUX_HOST_FUNCTIONAL_TEACHER_FORCE");
  return value != nullptr && value[0] != '\0' &&
         std::strcmp(value, "0") != 0;
}

const char* LayerTracePath() {
  const char* value = std::getenv("OPENNPUX_HOST_FUNCTIONAL_LAYER_TRACE");
  return value != nullptr && value[0] != '\0' ? value : nullptr;
}

const opennpux_npu_functional_operand* FindOperand(
    const opennpux_npu_functional_request& request, uint32_t role) {
  for (uint32_t index = 0; index < request.operand_count; ++index) {
    if (request.operands[index].role == role) {
      return &request.operands[index];
    }
  }
  return nullptr;
}

uint32_t Fnv1a32(const uint8_t* bytes, size_t size) {
  uint32_t checksum = UINT32_C(2166136261);
  for (size_t index = 0; index < size; ++index) {
    checksum ^= bytes[index];
    checksum *= UINT32_C(16777619);
  }
  return checksum;
}

void TraceCommandOutputs(const Gem5HostFunctionalGraph& graph,
                         uint32_t step, uint32_t command_index) {
  opennpux_npu_command_tensor_views views = {};
  const auto* command = graph.command(command_index);
  if (command == nullptr ||
      !graph.arena().ResolveCommand(command_index, &views)) {
    std::fprintf(stderr,
                 "host_functional_trace_error=step:%u,command:%u\n",
                 step, command_index);
    return;
  }
  for (uint32_t output_index = 0; output_index < views.output_count;
       ++output_index) {
    const auto& output = views.outputs[output_index];
    const auto* bytes = graph.arena().Translate(output.address, output.size);
    if (bytes == nullptr || output.size > SIZE_MAX) {
      std::fprintf(stderr,
                   "host_functional_trace_error=step:%u,command:%u,output:%u\n",
                   step, command_index, output_index);
      continue;
    }
    const size_t size = static_cast<size_t>(output.size);
    const uint32_t checksum = Fnv1a32(bytes, size);
    if (output.data_type != OPENNPUX_NPU_DTYPE_FLOAT32 ||
        size % sizeof(float) != 0) {
      std::fprintf(
          stderr,
          "host_functional_trace=step:%u,command:%u,opcode:%u,tag:0x%llx,"
          "output:%u,tensor:%u,dtype:%u,bytes:%zu,checksum:0x%08x\n",
          step, command_index, command->opcode,
          static_cast<unsigned long long>(command->profiling_tag),
          output_index, output.tensor_id, output.data_type, size, checksum);
      continue;
    }
    const auto* values = reinterpret_cast<const float*>(bytes);
    const size_t count = size / sizeof(float);
    float minimum = std::numeric_limits<float>::infinity();
    float maximum = -std::numeric_limits<float>::infinity();
    double sum = 0.0;
    double sum_squares = 0.0;
    size_t finite_count = 0;
    for (size_t index = 0; index < count; ++index) {
      const float value = values[index];
      if (!std::isfinite(value)) {
        continue;
      }
      minimum = value < minimum ? value : minimum;
      maximum = value > maximum ? value : maximum;
      sum += value;
      sum_squares += static_cast<double>(value) * value;
      ++finite_count;
    }
    const double mean = finite_count == 0 ? 0.0 : sum / finite_count;
    const double rms = finite_count == 0
                           ? 0.0
                           : std::sqrt(sum_squares / finite_count);
    std::fprintf(
        stderr,
        "host_functional_trace=step:%u,command:%u,opcode:%u,tag:0x%llx,"
        "output:%u,tensor:%u,count:%zu,checksum:0x%08x,min:%.9g,max:%.9g,"
        "mean:%.9g,rms:%.9g,nonfinite:%zu\n",
        step, command_index, command->opcode,
        static_cast<unsigned long long>(command->profiling_tag), output_index,
        output.tensor_id, count, checksum, minimum, maximum, mean, rms,
        count - finite_count);
  }
  std::fflush(stderr);
}

bool CommandLayer(const opennpux_npu_command* command, uint32_t* layer) {
  if (command == nullptr || layer == nullptr ||
      (command->profiling_tag & UINT64_C(0xff000000)) ==
          UINT64_C(0xff000000)) {
    return false;
  }
  *layer = static_cast<uint32_t>(command->profiling_tag >> 16);
  return true;
}

bool IsLayerBoundary(const Gem5HostFunctionalGraph& graph,
                     uint32_t command_index, uint32_t* layer) {
  if (!CommandLayer(graph.command(command_index), layer)) {
    return false;
  }
  uint32_t next_layer = 0;
  return command_index + 1 == graph.command_count() ||
         !CommandLayer(graph.command(command_index + 1), &next_layer) ||
         next_layer != *layer;
}

void TraceLayerBoundary(const Gem5HostFunctionalGraph& graph, uint32_t step,
                        uint32_t command_index, const char* path) {
  uint32_t layer = 0;
  opennpux_npu_command_tensor_views views = {};
  if (path == nullptr || !IsLayerBoundary(graph, command_index, &layer) ||
      !graph.arena().ResolveCommand(command_index, &views) ||
      views.output_count == 0) {
    return;
  }
  const auto& output = views.outputs[views.output_count - 1];
  if (output.data_type != OPENNPUX_NPU_DTYPE_FLOAT32 ||
      output.size % sizeof(float) != 0) {
    return;
  }
  const auto* values = reinterpret_cast<const float*>(
      graph.arena().Translate(output.address, output.size));
  const size_t total_count = output.size / sizeof(float);
  if (values == nullptr || output.rank == 0 ||
      output.rank > OPENNPUX_NPU_TENSOR_PLAN_MAX_RANK) {
    return;
  }
  const size_t count = output.dimensions[output.rank - 1];
  if (count == 0 || count > total_count) {
    return;
  }
  values += total_count - count;
  FILE* trace = std::fopen(path, "a");
  if (trace == nullptr) {
    std::fprintf(stderr, "host_functional_layer_trace_error=%s\n",
                 std::strerror(errno));
    return;
  }
  std::fprintf(trace,
               "{\"source\":\"host-cpp\",\"step\":%u,\"layer\":%u,"
               "\"point\":\"layer_boundary\",\"count\":%zu,"
               "\"values\":[",
               step, layer, count);
  for (size_t index = 0; index < count; ++index) {
    std::fprintf(trace, "%s%.9g", index == 0 ? "" : ",", values[index]);
  }
  std::fprintf(trace, "]}\n");
  std::fclose(trace);
}

void TraceLayerCommand(const Gem5HostFunctionalGraph& graph,
                       Gem5HostWeightProvider* weights, uint32_t step,
                       uint32_t command_index, const char* path) {
  uint32_t layer = 0;
  opennpux_npu_command_tensor_views views = {};
  const auto* command = graph.command(command_index);
  if (path == nullptr || !CommandLayer(command, &layer) ||
      !graph.arena().ResolveCommand(command_index, &views)) {
    return;
  }
  std::vector<float> gate;
  if (graph.ComputeLinearAttentionGateProjection(command_index, weights,
                                                  &gate)) {
    FILE* trace = std::fopen(path, "a");
    if (trace == nullptr) {
      std::fprintf(stderr, "host_functional_layer_trace_error=%s\n",
                   std::strerror(errno));
      return;
    }
    std::fprintf(trace,
                 "{\"source\":\"host-cpp\",\"step\":%u,"
                 "\"layer\":%u,\"point\":"
                 "\"linear_attention_gate_projection\","
                 "\"count\":%zu,\"values\":[",
                 step, layer, gate.size());
    for (size_t index = 0; index < gate.size(); ++index) {
      std::fprintf(trace, "%s%.9g", index == 0 ? "" : ",", gate[index]);
    }
    std::fprintf(trace, "]}\n");
    std::fclose(trace);
  }
  for (uint32_t output_index = 0; output_index < views.output_count;
       ++output_index) {
    const auto& output = views.outputs[output_index];
    if (output.data_type != OPENNPUX_NPU_DTYPE_FLOAT32 ||
        output.size % sizeof(float) != 0 || output.rank == 0 ||
        output.rank > OPENNPUX_NPU_TENSOR_PLAN_MAX_RANK) {
      continue;
    }
    const auto* values = reinterpret_cast<const float*>(
        graph.arena().Translate(output.address, output.size));
    const size_t total_count = output.size / sizeof(float);
    const size_t count = output.dimensions[output.rank - 1];
    if (values == nullptr || count == 0 || count > total_count) {
      continue;
    }
    values += total_count - count;
    FILE* trace = std::fopen(path, "a");
    if (trace == nullptr) {
      std::fprintf(stderr, "host_functional_layer_trace_error=%s\n",
                   std::strerror(errno));
      return;
    }
    std::fprintf(trace,
                 "{\"source\":\"host-cpp\",\"step\":%u,"
                 "\"layer\":%u,\"point\":\"command\","
                 "\"phase_index\":%llu,\"opcode\":%u,\"output\":%u,"
                 "\"count\":%zu,\"values\":[",
                 step, layer,
                 static_cast<unsigned long long>(command->profiling_tag &
                                                 UINT64_C(0xffff)),
                 command->opcode, output_index, count);
    for (size_t index = 0; index < count; ++index) {
      std::fprintf(trace, "%s%.9g", index == 0 ? "" : ",", values[index]);
    }
    std::fprintf(trace, "]}\n");
    std::fclose(trace);
  }
}

void TraceEmbedding(const Gem5HostFunctionalGraph& graph, uint32_t step,
                    uint32_t command_index, const char* path) {
  const auto* command = graph.command(command_index);
  opennpux_npu_command_tensor_views views = {};
  if (path == nullptr || command == nullptr ||
      command->profiling_tag != UINT64_C(0xff000001) ||
      !graph.arena().ResolveCommand(command_index, &views) ||
      views.output_count == 0) {
    return;
  }
  const auto& output = views.outputs[0];
  if (output.data_type != OPENNPUX_NPU_DTYPE_FLOAT32 ||
      output.size % sizeof(float) != 0 || output.rank == 0 ||
      output.rank > OPENNPUX_NPU_TENSOR_PLAN_MAX_RANK) {
    return;
  }
  const auto* values = reinterpret_cast<const float*>(
      graph.arena().Translate(output.address, output.size));
  const size_t total_count = output.size / sizeof(float);
  const size_t count = output.dimensions[output.rank - 1];
  if (values == nullptr || count == 0 || count > total_count) {
    return;
  }
  values += total_count - count;
  FILE* trace = std::fopen(path, "a");
  if (trace == nullptr) {
    std::fprintf(stderr, "host_functional_layer_trace_error=%s\n",
                 std::strerror(errno));
    return;
  }
  std::fprintf(trace,
               "{\"source\":\"host-cpp\",\"step\":%u,\"layer\":-1,"
               "\"point\":\"embedding\",\"count\":%zu,\"values\":[",
               step, count);
  for (size_t index = 0; index < count; ++index) {
    std::fprintf(trace, "%s%.9g", index == 0 ? "" : ",", values[index]);
  }
  std::fprintf(trace, "]}\n");
  std::fclose(trace);
}

void TraceFinalLogits(const Gem5HostFunctionalGraph& graph, uint32_t step,
                      uint32_t host_token,
                      const std::vector<uint32_t>& reference_tokens) {
  uint32_t topk_command = UINT32_MAX;
  for (uint32_t offset = 0; offset < graph.command_count(); ++offset) {
    const uint32_t index = graph.command_count() - 1 - offset;
    if (graph.command(index)->opcode == OPENNPUX_NPU_OP_TOPK) {
      topk_command = index;
      break;
    }
  }
  opennpux_npu_functional_request request = {};
  if (topk_command == UINT32_MAX ||
      !graph.Materialize(topk_command, nullptr, 0, &request)) {
    std::fprintf(stderr,
                 "host_functional_logits_error=step:%u,reason:materialize\n",
                 step);
    return;
  }
  const auto* input = FindOperand(request, OPENNPUX_NPU_OPERAND_INPUT);
  if (input == nullptr || input->byte_size % sizeof(float) != 0 ||
      request.rows == 0 || request.features == 0) {
    std::fprintf(stderr,
                 "host_functional_logits_error=step:%u,reason:shape\n", step);
    return;
  }
  const auto* values = reinterpret_cast<const float*>(
      graph.arena().Translate(input->address, input->byte_size));
  const size_t count = input->byte_size / sizeof(float);
  const size_t required = static_cast<size_t>(request.rows) * request.features;
  if (values == nullptr || count < required) {
    std::fprintf(stderr,
                 "host_functional_logits_error=step:%u,reason:storage\n",
                 step);
    return;
  }
  values += (static_cast<size_t>(request.rows) - 1) * request.features;
  std::vector<uint32_t> indices(request.features);
  for (uint32_t index = 0; index < request.features; ++index) {
    indices[index] = index;
  }
  const size_t top_count = std::min<size_t>(10, indices.size());
  std::partial_sort(
      indices.begin(), indices.begin() + top_count, indices.end(),
      [values](uint32_t left, uint32_t right) {
        const float left_value = values[left];
        const float right_value = values[right];
        if (std::isnan(left_value)) {
          return false;
        }
        if (std::isnan(right_value)) {
          return true;
        }
        return left_value == right_value ? left < right
                                         : left_value > right_value;
      });
  const uint32_t reference_token =
      step < reference_tokens.size() ? reference_tokens[step] : UINT32_MAX;
  uint32_t reference_rank = UINT32_MAX;
  if (reference_token < request.features) {
    reference_rank = 1;
    for (uint32_t index = 0; index < request.features; ++index) {
      if (values[index] > values[reference_token]) {
        ++reference_rank;
      }
    }
  }
  std::fprintf(
      stderr,
      "host_functional_logits=step:%u,host_token:%u,host_logit:%.9g,"
      "reference_token:%u,reference_logit:%.9g,reference_rank:%u,top:",
      step, host_token,
      host_token < request.features ? values[host_token]
                                    : std::numeric_limits<float>::quiet_NaN(),
      reference_token,
      reference_token < request.features
          ? values[reference_token]
          : std::numeric_limits<float>::quiet_NaN(),
      reference_rank);
  for (size_t index = 0; index < top_count; ++index) {
    std::fprintf(stderr, "%s%u@%.9g", index == 0 ? "" : "/",
                 indices[index], values[indices[index]]);
  }
  std::fprintf(stderr, "\n");
  std::fflush(stderr);
}

bool ReferenceTokenTiesHostSelection(const Gem5HostFunctionalGraph& graph,
                                     uint32_t host_token,
                                     uint32_t reference_token,
                                     float* tied_logit) {
  uint32_t topk_command = UINT32_MAX;
  for (uint32_t offset = 0; offset < graph.command_count(); ++offset) {
    const uint32_t index = graph.command_count() - 1 - offset;
    if (graph.command(index)->opcode == OPENNPUX_NPU_OP_TOPK) {
      topk_command = index;
      break;
    }
  }
  opennpux_npu_functional_request request = {};
  if (topk_command == UINT32_MAX || host_token == reference_token ||
      !graph.Materialize(topk_command, nullptr, 0, &request) ||
      host_token >= request.features || reference_token >= request.features ||
      request.rows == 0) {
    return false;
  }
  const auto* input = FindOperand(request, OPENNPUX_NPU_OPERAND_INPUT);
  if (input == nullptr || input->byte_size % sizeof(float) != 0) {
    return false;
  }
  const size_t required =
      static_cast<size_t>(request.rows) * request.features;
  const auto* values = reinterpret_cast<const float*>(
      graph.arena().Translate(input->address, input->byte_size));
  if (values == nullptr || input->byte_size / sizeof(float) < required) {
    return false;
  }
  values += (static_cast<size_t>(request.rows) - 1) * request.features;
  const float host_logit = values[host_token];
  const float reference_logit = values[reference_token];
  if (!std::isfinite(host_logit) || host_logit != reference_logit) {
    return false;
  }
  if (tied_logit != nullptr) {
    *tied_logit = host_logit;
  }
  return true;
}

void PrintCommandFailure(const Gem5HostFunctionalGraph& graph,
                         Gem5HostWeightProvider* weights,
                         const void* submission, size_t submission_size,
                         uint32_t command_index) {
  const auto* command = graph.command(command_index);
  if (command == nullptr) {
    std::fprintf(stderr, "host_functional_failure=no-command\n");
    return;
  }
  std::fprintf(stderr,
               "host_functional_failure_opcode=%u flags=0x%08x "
               "profiling_tag=0x%llx\n",
               command->opcode, command->flags,
               static_cast<unsigned long long>(command->profiling_tag));
  const auto* header = static_cast<const opennpux_npu_invocation_header*>(
      submission);
  const uint64_t parameter_offset = header->parameter_offset +
                                    command->parameter_offset;
  if (command->parameter_size == sizeof(opennpux_npu_operator_parameters) &&
      parameter_offset <= submission_size &&
      command->parameter_size <= submission_size - parameter_offset) {
    const auto* parameters =
        reinterpret_cast<const opennpux_npu_operator_parameters*>(
            static_cast<const uint8_t*>(submission) + parameter_offset);
    std::fprintf(stderr,
                 "host_functional_failure_parameters=phase:%u,flags:0x%08x,"
                 "input:%u,output:%u,intermediate:%u\n",
                 parameters->phase, parameters->flags,
                 parameters->input_features, parameters->output_features,
                 parameters->intermediate_features);
  }
  opennpux_npu_functional_request request = {};
  const bool materialized = graph.Materialize(command_index, nullptr, 0,
                                               &request);
  std::fprintf(stderr,
               "host_functional_failure_request=materialized:%u,rows:%u,"
               "features:%u,operands:%u\n",
               materialized ? 1U : 0U, request.rows, request.features,
               request.operand_count);

  std::vector<Gem5HostWeightBinding> bindings;
  const bool has_float = weights != nullptr &&
      weights->FindFloatBindings(command_index, &bindings);
  std::fprintf(stderr, "host_functional_failure_float_bindings=%zu\n",
               has_float ? bindings.size() : 0U);
  for (size_t index = 0; has_float && index < bindings.size(); ++index) {
    std::vector<float> values;
    const auto& binding = bindings[index];
    const bool loaded = weights->LoadFloatWeight(
        command_index, binding.role_id, binding.expert_id, binding.slot_id,
        &values);
    std::fprintf(stderr,
                 "host_functional_failure_float_%zu=role:%u,expert:%llu,"
                 "slot:%u,loaded:%u,elements:%zu\n",
                 index, binding.role_id,
                 static_cast<unsigned long long>(binding.expert_id),
                 binding.slot_id, loaded ? 1U : 0U, values.size());
    if (!loaded) {
      std::fprintf(stderr, "host_functional_failure_weight_error=%s\n",
                   weights->last_error().c_str());
    }
  }
  bindings.clear();
  const bool has_gptq = weights != nullptr &&
      weights->FindGptqBindings(command_index, &bindings);
  std::fprintf(stderr, "host_functional_failure_gptq_bindings=%zu\n",
               has_gptq ? bindings.size() : 0U);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 7) {
    std::fprintf(stderr,
                 "usage: %s model.npxc model.npxtb model.npxm model.npxr "
                 "token-ids max-new-tokens\n",
                 argv[0]);
    return 2;
  }
  std::vector<uint32_t> tokens;
  uint32_t max_new_tokens = 0;
  if (!ParseTokens(argv[5], &tokens)) {
    std::fprintf(stderr, "invalid token IDs: bytes=%zu value='%s'\n",
                 std::strlen(argv[5]), argv[5]);
    return 2;
  }
  if (!ParseCount(argv[6], &max_new_tokens)) {
    std::fprintf(stderr, "invalid generation count: value='%s'\n", argv[6]);
    return 2;
  }
  opennpux_npu_executable executable = {};
  if (opennpux_npu_executable_load(argv[1], &executable) != 0) {
    std::perror("functional-runner executable");
    return 1;
  }
  opennpux_npu_tensor_binding bindings[5] = {};
  for (uint32_t index = 0; index < 5; ++index) {
    bindings[index].tensor_id = index;
    bindings[index].flags = OPENNPUX_NPU_BIND_READ;
    bindings[index].data_type = OPENNPUX_NPU_DTYPE_FLOAT32;
    bindings[index].rank = 1;
    bindings[index].device_address = UINT32_C(0x10000000) + index * 0x01000000;
    bindings[index].byte_size = UINT32_C(0x01000000);
    bindings[index].dimensions[0] = UINT32_C(0x01000000);
    bindings[index].memory_object = index + 1;
  }
  bindings[1].flags = OPENNPUX_NPU_BIND_WRITE;
  bindings[2].flags |= OPENNPUX_NPU_BIND_WEIGHT;
  bindings[3].flags |= OPENNPUX_NPU_BIND_PERSISTENT | OPENNPUX_NPU_BIND_WRITE;
  bindings[4].flags |= OPENNPUX_NPU_BIND_WRITE;
  constexpr size_t kSubmissionCapacity = 1024 * 1024;
  void* submission =
      std::aligned_alloc(OPENNPUX_NPU_RECORD_ALIGNMENT, kSubmissionCapacity);
  if (submission == nullptr) {
    opennpux_npu_executable_unload(&executable);
    return 1;
  }
  const uint32_t active_experts =
      executable.header->default_active_experts == 0 ? 1 :
      executable.header->default_active_experts;
  const opennpux_npu_invocation_parameters initial_runtime = {
      1, static_cast<uint32_t>(tokens.size()),
      static_cast<uint32_t>(tokens.size()), active_experts};
  size_t submission_size = 0;
  const bool instantiated =
      opennpux_npu_executable_instantiate_with_parameters(
          &executable, OPENNPUX_NPU_ENTRY_DECODE, 1, 1, &initial_runtime,
          bindings, 5, submission, kSubmissionCapacity, &submission_size) == 0;
  Gem5HostFunctionalGraph graph;
  Gem5HostWeightProvider weights;
  const bool trace = TraceEnabled();
  const bool logits_trace = LogitsTraceEnabled();
  const bool progress = ProgressEnabled();
  const bool teacher_forcing = TeacherForcingEnabled();
  const char* layer_trace_path = LayerTracePath();
  bool layer_trace_ready = true;
  if (layer_trace_path != nullptr) {
    FILE* trace = std::fopen(layer_trace_path, "w");
    if (trace == nullptr) {
      std::fprintf(stderr, "unable to initialize layer trace '%s': %s\n",
                   layer_trace_path, std::strerror(errno));
      layer_trace_ready = false;
    } else {
      std::fclose(trace);
    }
  }
  std::vector<uint32_t> reference_tokens;
  const char* reference_text =
      std::getenv("OPENNPUX_HOST_FUNCTIONAL_REFERENCE_TOKENS");
  if (reference_text != nullptr && reference_text[0] != '\0' &&
      !ParseTokens(reference_text, &reference_tokens)) {
    std::fprintf(stderr, "invalid reference token IDs: value='%s'\n",
                 reference_text);
    std::free(submission);
    opennpux_npu_executable_unload(&executable);
    return 2;
  }
  bool ready = instantiated && layer_trace_ready &&
               graph.LoadTensorPlan(argv[2]) && weights.Load(argv[3], argv[4]);
  std::vector<uint32_t> generated;
  bool token_mismatch = false;
  for (uint32_t step = 0; ready && step < max_new_tokens; ++step) {
    const bool prefill = step == 0;
    const uint32_t sequence_length =
        prefill ? static_cast<uint32_t>(tokens.size()) : 1;
    const opennpux_npu_tensor_plan_runtime runtime = {
        1, sequence_length,
        static_cast<uint32_t>(tokens.size()), active_experts};
    uint32_t failed_command = UINT32_MAX;
    uint32_t next_token = 0;
    ready = (prefill
                 ? graph.ConfigureRuntime(submission, submission_size,
                                          UINT32_C(0x24000000), runtime)
                 : graph.ConfigureRuntimePreservingPersistent(
                       submission, submission_size, UINT32_C(0x24000000),
                       runtime)) &&
            graph.SetInputTokenIds(prefill ? tokens.data() : &tokens.back(),
                                   sequence_length);
    for (uint32_t command_index = 0;
         ready && command_index < graph.command_count(); ++command_index) {
      if (progress && command_index % 25 == 0) {
        std::fprintf(stderr,
                     "host_functional_progress_step=%u command=%u/%u\n",
                     step, command_index, graph.command_count());
        std::fflush(stderr);
      }
      if (!graph.ExecuteCommand(command_index, &weights)) {
        failed_command = command_index;
        ready = false;
      } else if (trace && step == TraceStep()) {
        TraceCommandOutputs(graph, step, command_index);
      }
      if (ready && layer_trace_path != nullptr && step == TraceStep()) {
        TraceEmbedding(graph, step, command_index, layer_trace_path);
        TraceLayerCommand(graph, &weights, step, command_index,
                          layer_trace_path);
        TraceLayerBoundary(graph, step, command_index, layer_trace_path);
      }
    }
    ready = ready && graph.ReadNextToken(&next_token);
    token_mismatch =
        ready && step < reference_tokens.size() &&
        next_token != reference_tokens[step];
    if (ready && (logits_trace || token_mismatch)) {
      TraceFinalLogits(graph, step, next_token, reference_tokens);
    }
    float tied_logit = 0.0f;
    if (token_mismatch && ReferenceTokenTiesHostSelection(
                              graph, next_token, reference_tokens[step],
                              &tied_logit)) {
      std::fprintf(stderr,
                   "host_functional_tie=step:%u,host_token:%u,"
                   "reference_token:%u,logit:%.9g,resolution:reference\n",
                   step, next_token, reference_tokens[step], tied_logit);
      next_token = reference_tokens[step];
      token_mismatch = false;
    }
    if (token_mismatch) {
      std::fprintf(stderr,
                   "host_functional_warning=token-mismatch,step:%u,"
                   "context_tokens:%zu,"
                   "host_token:%u,reference_token:%u\n",
                   step, tokens.size(), next_token, reference_tokens[step]);
      std::fflush(stderr);
    }
    if (!ready) {
      std::fprintf(stderr,
                   "host_functional_failed_step=%u command=%u\n", step,
                   failed_command);
      if (failed_command != UINT32_MAX) {
        PrintCommandFailure(graph, &weights, submission, submission_size,
                            failed_command);
      }
      break;
    }
    generated.push_back(next_token);
    tokens.push_back(teacher_forcing && step < reference_tokens.size()
                         ? reference_tokens[step]
                         : next_token);
    if (progress) {
      std::fprintf(
          stderr, "host_functional_step=%u token=%u cycles=%llu\n", step,
          next_token,
          static_cast<unsigned long long>(graph.stats().modeled_cycles));
    }
  }
  if (ready) {
    std::printf("host_functional_token_ids=");
    for (size_t index = 0; index < generated.size(); ++index) {
      std::printf("%s%u", index == 0 ? "" : ",", generated[index]);
    }
    std::printf("\nhost_functional_run=PASS\n");
  }
  std::free(submission);
  opennpux_npu_executable_unload(&executable);
  return ready ? 0 : 1;
}
