#include "hw_sim/gem5_bridge/gem5_host_functional_graph.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "hw_sim/gem5_bridge/gem5_host_weight_provider.h"
#include "hw_sim/gem5_bridge/gem5_host_routed_expert.h"

namespace {

bool DecodeRuntime(const void* submission, size_t submission_size,
                   opennpux_npu_tensor_plan_runtime* runtime) {
  if (runtime == nullptr ||
      opennpux_npu_submission_validate(submission, submission_size) != 0) {
    return false;
  }
  const auto* header =
      static_cast<const opennpux_npu_invocation_header*>(submission);
  if (header->command_count == 0) {
    return false;
  }
  const auto* commands = reinterpret_cast<const opennpux_npu_command*>(
      static_cast<const uint8_t*>(submission) + header->command_offset);
  const uint64_t shape = commands[0].runtime_shape;
  runtime->batch_size = shape & OPENNPUX_NPU_RUNTIME_FIELD_MASK;
  runtime->sequence_length =
      (shape >> OPENNPUX_NPU_RUNTIME_SEQUENCE_SHIFT) &
      OPENNPUX_NPU_RUNTIME_FIELD_MASK;
  runtime->kv_length = (shape >> OPENNPUX_NPU_RUNTIME_KV_SHIFT) &
                       OPENNPUX_NPU_RUNTIME_FIELD_MASK;
  runtime->active_experts =
      (shape >> OPENNPUX_NPU_RUNTIME_EXPERT_SHIFT) &
      OPENNPUX_NPU_RUNTIME_FIELD_MASK;
  return runtime->batch_size != 0 && runtime->sequence_length != 0;
}

bool RegionsOverlap(uint32_t first_base, size_t first_size,
                    uint32_t second_base, size_t second_size) {
  const uint64_t first_end = static_cast<uint64_t>(first_base) + first_size;
  const uint64_t second_end = static_cast<uint64_t>(second_base) + second_size;
  return first_base < second_end && second_base < first_end;
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

bool UseBfloat16Boundaries() {
  const char* precision =
      std::getenv("OPENNPUX_HOST_FUNCTIONAL_PRECISION");
  return precision != nullptr && std::strcmp(precision, "bf16") == 0;
}

bool IsFloatingOutputRole(uint32_t role) {
  return role == OPENNPUX_NPU_OPERAND_OUTPUT ||
         role == OPENNPUX_NPU_OPERAND_OUTPUT_SECONDARY ||
         role == OPENNPUX_NPU_OPERAND_OUTPUT_TERTIARY ||
         role == OPENNPUX_NPU_OPERAND_OUTPUT_QUATERNARY;
}

uint8_t* TranslateRegion(const std::vector<Gem5FunctionalMemoryRegion>& regions,
                         uint32_t address, uint32_t size) {
  for (const auto& region : regions) {
    if (address < region.base) {
      continue;
    }
    const uint64_t offset = static_cast<uint64_t>(address) - region.base;
    if (offset <= region.size && size <= region.size - offset) {
      return region.data + offset;
    }
  }
  return nullptr;
}

void RoundOutputsToBfloat16(
    const opennpux_npu_functional_request& request,
    const std::vector<Gem5FunctionalMemoryRegion>& regions) {
  for (uint32_t operand_index = 0; operand_index < request.operand_count;
       ++operand_index) {
    const auto& operand = request.operands[operand_index];
    if (!IsFloatingOutputRole(operand.role) ||
        operand.byte_size % sizeof(float) != 0) {
      continue;
    }
    auto* values = reinterpret_cast<float*>(
        TranslateRegion(regions, operand.address, operand.byte_size));
    if (values == nullptr) {
      continue;
    }
    const size_t count = operand.byte_size / sizeof(float);
    for (size_t index = 0; index < count; ++index) {
      uint32_t bits = 0;
      std::memcpy(&bits, values + index, sizeof(bits));
      const uint32_t exponent = bits & UINT32_C(0x7f800000);
      if (exponent != UINT32_C(0x7f800000)) {
        bits += UINT32_C(0x00007fff) + ((bits >> 16) & 1U);
        bits &= UINT32_C(0xffff0000);
        std::memcpy(values + index, &bits, sizeof(bits));
      }
    }
  }
}

}  // namespace

bool Gem5HostFunctionalGraph::LoadTensorPlan(
    const std::string& tensor_plan_path) {
  ResetInvocation();
  return arena_.Load(tensor_plan_path);
}

bool Gem5HostFunctionalGraph::Configure(const void* submission,
                                         size_t submission_size,
                                         uint32_t submission_base,
                                         uint32_t arena_base) {
  opennpux_npu_tensor_plan_runtime runtime = {};
  if (!DecodeRuntime(submission, submission_size, &runtime)) {
    return false;
  }
  return ConfigureRuntime(submission, submission_size, submission_base,
                          runtime, arena_base);
}

bool Gem5HostFunctionalGraph::ConfigureRuntime(
    const void* submission, size_t submission_size, uint32_t submission_base,
    const opennpux_npu_tensor_plan_runtime& runtime, uint32_t arena_base) {
  ResetInvocation();
  if (!arena_.loaded() || submission == nullptr || submission_size == 0 ||
      submission_size > UINT32_MAX || runtime.batch_size == 0 ||
      runtime.sequence_length == 0 ||
      static_cast<uint64_t>(submission_base) + submission_size >
          (UINT64_C(1) << 32)) {
    return false;
  }
  try {
    const auto* bytes = static_cast<const uint8_t*>(submission);
    submission_.assign(bytes, bytes + submission_size);
  } catch (...) {
    ResetInvocation();
    return false;
  }
  if (!arena_.Configure(runtime, arena_base) ||
      RegionsOverlap(submission_base, submission_.size(), arena_.base(),
                     arena_.size()) ||
      opennpux_npu_functional_program_init(
          &program_, submission_.data(), submission_.size(), submission_base,
          &arena_.plan(), &arena_.memory()) != 0) {
    ResetInvocation();
    return false;
  }
  program_.runtime = runtime;
  submission_base_ = submission_base;
  configured_ = true;
  return true;
}

bool Gem5HostFunctionalGraph::Materialize(
    uint32_t command_index,
    const opennpux_npu_functional_operand* extra_operands,
    uint32_t extra_operand_count,
    opennpux_npu_functional_request* request) const {
  return configured_ && request != nullptr &&
         opennpux_npu_functional_program_materialize(
             &program_, command_index, extra_operands, extra_operand_count,
             request) == 0;
}

bool Gem5HostFunctionalGraph::Execute(
    opennpux_npu_functional_request* request,
    const Gem5FunctionalMemoryRegion* extra_regions,
    size_t extra_region_count) {
  if (!configured_ || request == nullptr ||
      (extra_region_count != 0 && extra_regions == nullptr)) {
    return false;
  }
  std::vector<Gem5FunctionalMemoryRegion> regions;
  try {
    regions.reserve(2 + extra_region_count);
    regions.push_back(
        {submission_base_, submission_.data(), submission_.size()});
    regions.push_back({arena_.base(), arena_.data(), arena_.size()});
    if (extra_region_count != 0) {
      regions.insert(regions.end(), extra_regions,
                     extra_regions + extra_region_count);
    }
  } catch (...) {
    return false;
  }
  if (!ExecuteGem5FunctionalRequestInRegions(request, regions.data(),
                                              regions.size())) {
    return false;
  }
  if (UseBfloat16Boundaries()) {
    RoundOutputsToBfloat16(*request, regions);
  }
  ++stats_.completed_commands;
  stats_.operations += request->operation_count;
  stats_.modeled_cycles += request->modeled_cycles;
  stats_.bytes_read += request->bytes_read;
  stats_.bytes_written += request->bytes_written;
  return true;
}

bool Gem5HostFunctionalGraph::ExecuteGptqProjection(
    uint32_t command_index, Gem5HostWeightProvider* weights,
    uint32_t role_id, uint64_t expert_id, uint32_t slot_id) {
  if (!configured_ || weights == nullptr) {
    return false;
  }
  Gem5HostGptqWeights loaded = {};
  if (!weights->LoadProjection(command_index, role_id, expert_id, slot_id,
                               &loaded)) {
    return false;
  }
  const Gem5HostConstBuffer components[] = {
      loaded.qweight, loaded.qzeros, loaded.scales, loaded.g_idx};
  const uint32_t roles[] = {
      OPENNPUX_NPU_OPERAND_QWEIGHT, OPENNPUX_NPU_OPERAND_QZEROS,
      OPENNPUX_NPU_OPERAND_SCALES, OPENNPUX_NPU_OPERAND_G_IDX};
  opennpux_npu_functional_operand operands[4] = {};
  Gem5FunctionalMemoryRegion regions[4] = {};
  uint32_t count = 0;
  uint64_t address = UINT32_C(0x50000000);
  for (size_t index = 0; index < 4; ++index) {
    if (components[index].data == nullptr || components[index].size == 0) {
      continue;
    }
    address = (address + 63) & ~UINT64_C(63);
    if (components[index].size > UINT32_MAX ||
        address + components[index].size > (UINT64_C(1) << 32)) {
      return false;
    }
    operands[count] = {roles[index], static_cast<uint32_t>(address),
                       static_cast<uint32_t>(components[index].size), 0};
    regions[count] = {
        static_cast<uint32_t>(address),
        const_cast<uint8_t*>(
            static_cast<const uint8_t*>(components[index].data)),
        components[index].size};
    address += components[index].size;
    ++count;
  }
  opennpux_npu_functional_request request = {};
  return count >= 3 && Materialize(command_index, operands, count, &request) &&
         Execute(&request, regions, count);
}

bool Gem5HostFunctionalGraph::ExecuteGptqQkv(
    uint32_t command_index, Gem5HostWeightProvider* weights) {
  if (!configured_ || weights == nullptr) {
    return false;
  }
  const uint32_t semantic_roles[] = {
      OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_Q_PROJ,
      OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_K_PROJ,
      OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_V_PROJ};
  const uint32_t slots[] = {OPENNPUX_NPU_WEIGHT_SLOT_Q_PROJ,
                            OPENNPUX_NPU_WEIGHT_SLOT_K_PROJ,
                            OPENNPUX_NPU_WEIGHT_SLOT_V_PROJ};
  const uint32_t operand_roles[][4] = {
      {OPENNPUX_NPU_OPERAND_Q_QWEIGHT,
       OPENNPUX_NPU_OPERAND_Q_QZEROS,
       OPENNPUX_NPU_OPERAND_Q_SCALES, OPENNPUX_NPU_OPERAND_Q_G_IDX},
      {OPENNPUX_NPU_OPERAND_K_QWEIGHT,
       OPENNPUX_NPU_OPERAND_K_QZEROS,
       OPENNPUX_NPU_OPERAND_K_SCALES, OPENNPUX_NPU_OPERAND_K_G_IDX},
      {OPENNPUX_NPU_OPERAND_V_QWEIGHT,
       OPENNPUX_NPU_OPERAND_V_QZEROS,
       OPENNPUX_NPU_OPERAND_V_SCALES, OPENNPUX_NPU_OPERAND_V_G_IDX}};
  Gem5HostGptqWeights loaded[3] = {};
  opennpux_npu_functional_operand operands[14] = {};
  Gem5FunctionalMemoryRegion regions[14] = {};
  uint32_t count = 0;
  uint64_t address = UINT32_C(0x50000000);
  for (uint32_t projection = 0; projection < 3; ++projection) {
    if (!weights->LoadProjection(
            command_index, semantic_roles[projection],
            OPENNPUX_NPU_WEIGHT_EXPERT_NONE, slots[projection],
            &loaded[projection], projection)) {
      return false;
    }
    const Gem5HostConstBuffer components[] = {
        loaded[projection].qweight, loaded[projection].qzeros,
        loaded[projection].scales, loaded[projection].g_idx};
    for (uint32_t component = 0; component < 4; ++component) {
      if (components[component].data == nullptr ||
          components[component].size == 0) {
        continue;
      }
      address = (address + 63) & ~UINT64_C(63);
      if (components[component].size > UINT32_MAX ||
          address + components[component].size > (UINT64_C(1) << 32)) {
        return false;
      }
      operands[count] = {operand_roles[projection][component],
                         static_cast<uint32_t>(address),
                         static_cast<uint32_t>(components[component].size), 0};
      regions[count] = {
          static_cast<uint32_t>(address),
          const_cast<uint8_t*>(static_cast<const uint8_t*>(
              components[component].data)),
          components[component].size};
      address += components[component].size;
      ++count;
    }
  }
  std::vector<Gem5HostWeightBinding> floating;
  std::vector<float> norm_weights[2];
  const uint32_t norm_semantic_roles[] = {
      OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_Q_NORM,
      OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_K_NORM};
  const uint32_t norm_operand_roles[] = {
      OPENNPUX_NPU_OPERAND_ATTENTION_Q_NORM_WEIGHT,
      OPENNPUX_NPU_OPERAND_ATTENTION_K_NORM_WEIGHT};
  if (weights->FindFloatBindings(command_index, &floating)) {
    for (size_t index = 0; index < 2; ++index) {
      const auto binding = std::find_if(
          floating.begin(), floating.end(), [&](const auto& candidate) {
            return candidate.role_id == norm_semantic_roles[index] &&
                   candidate.expert_id == OPENNPUX_NPU_WEIGHT_EXPERT_NONE;
          });
      if (binding == floating.end() ||
          !weights->LoadFloatWeight(command_index, binding->role_id,
                                    binding->expert_id, binding->slot_id,
                                    &norm_weights[index]) ||
          norm_weights[index].empty()) {
        return false;
      }
      address = (address + 63) & ~UINT64_C(63);
      const size_t bytes = norm_weights[index].size() * sizeof(float);
      if (bytes > UINT32_MAX ||
          address + bytes > (UINT64_C(1) << 32)) {
        return false;
      }
      operands[count] = {norm_operand_roles[index],
                         static_cast<uint32_t>(address),
                         static_cast<uint32_t>(bytes), 0};
      regions[count] = {
          static_cast<uint32_t>(address),
          reinterpret_cast<uint8_t*>(norm_weights[index].data()), bytes};
      address += bytes;
      ++count;
    }
  }
  opennpux_npu_functional_request request = {};
  return count >= 9 && Materialize(command_index, operands, count, &request) &&
         Execute(&request, regions, count);
}

bool Gem5HostFunctionalGraph::ExecuteGptqRouter(
    uint32_t command_index, Gem5HostWeightProvider* weights) {
  const auto* selected = command(command_index);
  if (selected == nullptr || selected->opcode != OPENNPUX_NPU_OP_ROUTER ||
      weights == nullptr) {
    return false;
  }
  if (ExecuteGptqProjection(command_index, weights,
                            OPENNPUX_NPU_WEIGHT_ROLE_ROUTER,
                            OPENNPUX_NPU_WEIGHT_EXPERT_NONE,
                            OPENNPUX_NPU_WEIGHT_SLOT_DEFAULT)) {
    return true;
  }
  const Gem5HostWeightBinding binding = {
      OPENNPUX_NPU_WEIGHT_ROLE_ROUTER, OPENNPUX_NPU_WEIGHT_EXPERT_NONE,
      OPENNPUX_NPU_WEIGHT_SLOT_DEFAULT};
  return ExecuteFloatWeight(command_index, weights, binding);
}

bool Gem5HostFunctionalGraph::ExecuteRoutedExpert(
    uint32_t command_index, Gem5HostWeightProvider* weights) {
  opennpux_npu_functional_request request = {};
  if (!configured_ || weights == nullptr ||
      !Materialize(command_index, nullptr, 0, &request) ||
      request.opcode != OPENNPUX_NPU_OP_EXPERT ||
      !weights->ConfigureRoutedExpert(command_index)) {
    return false;
  }
  const auto* input = FindOperand(request, OPENNPUX_NPU_OPERAND_INPUT);
  const auto* ids = FindOperand(request, OPENNPUX_NPU_OPERAND_SECONDARY);
  const auto* route_weights =
      FindOperand(request, OPENNPUX_NPU_OPERAND_INPUT_TERTIARY);
  const auto* output = FindOperand(request, OPENNPUX_NPU_OPERAND_OUTPUT);
  if (input == nullptr || ids == nullptr || route_weights == nullptr ||
      output == nullptr || request.rows == 0 ||
      arena_.runtime().active_experts == 0 ||
      request.parameter_address < submission_base_ ||
      request.parameter_size > submission_.size() ||
      request.parameter_address - submission_base_ >
          submission_.size() - request.parameter_size) {
    return false;
  }
  auto* input_data = reinterpret_cast<const float*>(
      arena_.Translate(input->address, input->byte_size));
  auto* id_data = reinterpret_cast<const uint32_t*>(
      arena_.Translate(ids->address, ids->byte_size));
  auto* route_data = reinterpret_cast<const float*>(
      arena_.Translate(route_weights->address, route_weights->byte_size));
  auto* output_data = reinterpret_cast<float*>(
      arena_.Translate(output->address, output->byte_size));
  const uint64_t route_count =
      static_cast<uint64_t>(request.rows) * arena_.runtime().active_experts;
  if (input_data == nullptr || id_data == nullptr || route_data == nullptr ||
      output_data == nullptr || route_count > ids->byte_size / sizeof(uint32_t) ||
      route_count > route_weights->byte_size / sizeof(float)) {
    return false;
  }
  const void* parameters = submission_.data() +
      (request.parameter_address - submission_base_);
  Gem5HostRoutedExpertStats routed_stats = {};
  if (!RunGem5HostRoutedExpert(
          parameters, request.rows, input_data, input->byte_size, id_data,
          route_data, arena_.runtime().active_experts, output_data,
          output->byte_size, weights, &routed_stats)) {
    return false;
  }
  ++stats_.completed_commands;
  stats_.operations += routed_stats.operations;
  stats_.modeled_cycles += routed_stats.modeled_cycles;
  stats_.bytes_read += routed_stats.bytes_read;
  stats_.bytes_written += routed_stats.bytes_written;
  return true;
}

bool Gem5HostFunctionalGraph::ExecuteGptqExpert(
    uint32_t command_index, Gem5HostWeightProvider* weights,
    const Gem5HostWeightBinding& binding) {
  opennpux_npu_functional_request initial = {};
  if (!configured_ || weights == nullptr ||
      !Materialize(command_index, nullptr, 0, &initial) ||
      initial.opcode != OPENNPUX_NPU_OP_EXPERT ||
      initial.parameter_address < submission_base_ ||
      initial.parameter_size != sizeof(opennpux_npu_operator_parameters) ||
      initial.parameter_address - submission_base_ >
          submission_.size() - initial.parameter_size) {
    return false;
  }
  const auto* parameters =
      reinterpret_cast<const opennpux_npu_operator_parameters*>(
          submission_.data() + initial.parameter_address - submission_base_);
  const uint64_t intermediate_elements =
      static_cast<uint64_t>(initial.rows) * parameters->intermediate_features;
  if (parameters->intermediate_features == 0 ||
      intermediate_elements > SIZE_MAX / sizeof(float) ||
      intermediate_elements > UINT32_MAX / sizeof(float)) {
    return false;
  }
  const uint32_t slots[] = {OPENNPUX_NPU_WEIGHT_SLOT_GATE_PROJ,
                            OPENNPUX_NPU_WEIGHT_SLOT_UP_PROJ,
                            OPENNPUX_NPU_WEIGHT_SLOT_DOWN_PROJ};
  const uint32_t operand_roles[][4] = {
      {OPENNPUX_NPU_OPERAND_GATE_QWEIGHT,
       OPENNPUX_NPU_OPERAND_GATE_QZEROS,
       OPENNPUX_NPU_OPERAND_GATE_SCALES,
       OPENNPUX_NPU_OPERAND_GATE_G_IDX},
      {OPENNPUX_NPU_OPERAND_UP_QWEIGHT,
       OPENNPUX_NPU_OPERAND_UP_QZEROS,
       OPENNPUX_NPU_OPERAND_UP_SCALES,
       OPENNPUX_NPU_OPERAND_UP_G_IDX},
      {OPENNPUX_NPU_OPERAND_DOWN_QWEIGHT,
       OPENNPUX_NPU_OPERAND_DOWN_QZEROS,
       OPENNPUX_NPU_OPERAND_DOWN_SCALES,
       OPENNPUX_NPU_OPERAND_DOWN_G_IDX}};
  Gem5HostGptqWeights loaded[3] = {};
  opennpux_npu_functional_operand operands[15] = {};
  Gem5FunctionalMemoryRegion regions[15] = {};
  uint32_t count = 0;
  uint64_t address = UINT32_C(0x50000000);
  for (uint32_t projection = 0; projection < 3; ++projection) {
    if (!weights->LoadProjection(command_index, binding.role_id,
                                 binding.expert_id, slots[projection],
                                 &loaded[projection], projection)) {
      return false;
    }
    const Gem5HostConstBuffer components[] = {
        loaded[projection].qweight, loaded[projection].qzeros,
        loaded[projection].scales, loaded[projection].g_idx};
    for (uint32_t component = 0; component < 4; ++component) {
      if (components[component].data == nullptr ||
          components[component].size == 0) {
        continue;
      }
      address = (address + 63) & ~UINT64_C(63);
      if (components[component].size > UINT32_MAX ||
          address + components[component].size > (UINT64_C(1) << 32)) {
        return false;
      }
      operands[count] = {operand_roles[projection][component],
                         static_cast<uint32_t>(address),
                         static_cast<uint32_t>(components[component].size), 0};
      regions[count] = {
          static_cast<uint32_t>(address),
          const_cast<uint8_t*>(static_cast<const uint8_t*>(
              components[component].data)),
          components[component].size};
      address += components[component].size;
      ++count;
    }
  }
  std::vector<float> intermediates[3];
  const uint32_t temporary_roles[] = {OPENNPUX_NPU_OPERAND_GATE_OUTPUT,
                                      OPENNPUX_NPU_OPERAND_UP_OUTPUT,
                                      OPENNPUX_NPU_OPERAND_ACTIVATED};
  for (uint32_t temporary = 0; temporary < 3; ++temporary) {
    try {
      intermediates[temporary].resize(
          static_cast<size_t>(intermediate_elements));
    } catch (...) {
      return false;
    }
    address = (address + 63) & ~UINT64_C(63);
    const size_t size = intermediates[temporary].size() * sizeof(float);
    if (address + size > (UINT64_C(1) << 32)) {
      return false;
    }
    operands[count] = {temporary_roles[temporary],
                       static_cast<uint32_t>(address),
                       static_cast<uint32_t>(size), 0};
    regions[count] = {static_cast<uint32_t>(address),
                      reinterpret_cast<uint8_t*>(
                          intermediates[temporary].data()),
                      size};
    address += size;
    ++count;
  }
  opennpux_npu_functional_request request = {};
  return count >= 12 && Materialize(command_index, operands, count, &request) &&
         Execute(&request, regions, count);
}

bool Gem5HostFunctionalGraph::ExecuteFloatWeight(
    uint32_t command_index, Gem5HostWeightProvider* weights,
    const Gem5HostWeightBinding& binding) {
  if (!configured_ || weights == nullptr) {
    return false;
  }
  std::vector<float> weight;
  if (!weights->LoadFloatWeight(command_index, binding.role_id,
                                binding.expert_id, binding.slot_id, &weight) ||
      weight.empty() || weight.size() > UINT32_MAX / sizeof(float)) {
    return false;
  }
  opennpux_npu_functional_request initial = {};
  if (!Materialize(command_index, nullptr, 0, &initial) ||
      initial.parameter_address < submission_base_ ||
      initial.parameter_size != sizeof(opennpux_npu_operator_parameters) ||
      initial.parameter_address - submission_base_ >
          submission_.size() - initial.parameter_size) {
    return false;
  }
  const auto* parameters =
      reinterpret_cast<const opennpux_npu_operator_parameters*>(
          submission_.data() + initial.parameter_address - submission_base_);
  uint64_t required = 0;
  if (initial.opcode == OPENNPUX_NPU_OP_EMBED) {
    required = static_cast<uint64_t>(parameters->input_features) *
               initial.features;
  } else if (initial.opcode == OPENNPUX_NPU_OP_NORMALIZE) {
    required = initial.features;
  } else if (initial.opcode == OPENNPUX_NPU_OP_MATMUL ||
             initial.opcode == OPENNPUX_NPU_OP_ROUTER) {
    if (parameters->output_features == 0 ||
        weight.size() % parameters->output_features != 0) {
      return false;
    }
    required = weight.size();
  } else if (initial.opcode == OPENNPUX_NPU_OP_CAUSAL_CONVOLUTION) {
    required = static_cast<uint64_t>(parameters->intermediate_features) *
               initial.features;
  }
  if (required == 0 || required != weight.size()) {
    return false;
  }
  constexpr uint32_t kWeightAddress = UINT32_C(0x50000000);
  const opennpux_npu_functional_operand operand = {
      OPENNPUX_NPU_OPERAND_WEIGHT, kWeightAddress,
      static_cast<uint32_t>(weight.size() * sizeof(float)), 0};
  Gem5FunctionalMemoryRegion region = {
      kWeightAddress, reinterpret_cast<uint8_t*>(weight.data()),
      weight.size() * sizeof(float)};
  opennpux_npu_functional_request request = {};
  return Materialize(command_index, &operand, 1, &request) &&
         Execute(&request, &region, 1);
}

bool Gem5HostFunctionalGraph::ExecuteLinearAttentionProjection(
    uint32_t command_index, Gem5HostWeightProvider* weights,
    const std::vector<Gem5HostWeightBinding>& bindings) {
  if (!configured_ || weights == nullptr || bindings.size() != 3) {
    return false;
  }
  const uint32_t semantic_roles[] = {
      OPENNPUX_NPU_WEIGHT_ROLE_LINEAR_QKV,
      OPENNPUX_NPU_WEIGHT_ROLE_LINEAR_ALPHA,
      OPENNPUX_NPU_WEIGHT_ROLE_LINEAR_BETA};
  const uint32_t operand_roles[] = {
      OPENNPUX_NPU_OPERAND_LINEAR_QKV_WEIGHT,
      OPENNPUX_NPU_OPERAND_LINEAR_ALPHA_WEIGHT,
      OPENNPUX_NPU_OPERAND_LINEAR_BETA_WEIGHT};
  std::vector<float> loaded[3];
  opennpux_npu_functional_operand operands[3] = {};
  Gem5FunctionalMemoryRegion regions[3] = {};
  uint64_t address = UINT32_C(0x50000000);
  for (size_t index = 0; index < 3; ++index) {
    const auto binding = std::find_if(
        bindings.begin(), bindings.end(), [&](const auto& candidate) {
          return candidate.role_id == semantic_roles[index];
        });
    if (binding == bindings.end() || !weights->LoadFloatWeight(
            command_index, binding->role_id, binding->expert_id,
            binding->slot_id, &loaded[index]) || loaded[index].empty() ||
        loaded[index].size() > UINT32_MAX / sizeof(float)) {
      return false;
    }
    address = (address + 63) & ~UINT64_C(63);
    const size_t bytes = loaded[index].size() * sizeof(float);
    if (address + bytes > (UINT64_C(1) << 32)) {
      return false;
    }
    operands[index] = {operand_roles[index], static_cast<uint32_t>(address),
                       static_cast<uint32_t>(bytes), 0};
    regions[index] = {static_cast<uint32_t>(address),
                      reinterpret_cast<uint8_t*>(loaded[index].data()), bytes};
    address += bytes;
  }
  opennpux_npu_functional_request request = {};
  return Materialize(command_index, operands, 3, &request) &&
         request.opcode == OPENNPUX_NPU_OP_MATMUL &&
         Execute(&request, regions, 3);
}

bool Gem5HostFunctionalGraph::ExecuteLinearAttentionGateNorm(
    uint32_t command_index, Gem5HostWeightProvider* weights,
    const std::vector<Gem5HostWeightBinding>& bindings) {
  if (!configured_ || weights == nullptr || bindings.size() != 2) {
    return false;
  }
  const uint32_t semantic_roles[] = {
      OPENNPUX_NPU_WEIGHT_ROLE_LINEAR_GATE,
      OPENNPUX_NPU_WEIGHT_ROLE_LINEAR_NORM};
  const uint32_t operand_roles[] = {
      OPENNPUX_NPU_OPERAND_LINEAR_GATE_WEIGHT,
      OPENNPUX_NPU_OPERAND_LINEAR_NORM_WEIGHT};
  std::vector<float> loaded[2];
  opennpux_npu_functional_operand operands[2] = {};
  Gem5FunctionalMemoryRegion regions[2] = {};
  uint64_t address = UINT32_C(0x50000000);
  for (size_t index = 0; index < 2; ++index) {
    const auto binding = std::find_if(
        bindings.begin(), bindings.end(), [&](const auto& candidate) {
          return candidate.role_id == semantic_roles[index];
        });
    if (binding == bindings.end() || !weights->LoadFloatWeight(
            command_index, binding->role_id, binding->expert_id,
            binding->slot_id, &loaded[index]) || loaded[index].empty() ||
        loaded[index].size() > UINT32_MAX / sizeof(float)) {
      return false;
    }
    address = (address + 63) & ~UINT64_C(63);
    const size_t bytes = loaded[index].size() * sizeof(float);
    if (address + bytes > (UINT64_C(1) << 32)) {
      return false;
    }
    operands[index] = {operand_roles[index], static_cast<uint32_t>(address),
                       static_cast<uint32_t>(bytes), 0};
    regions[index] = {static_cast<uint32_t>(address),
                      reinterpret_cast<uint8_t*>(loaded[index].data()), bytes};
    address += bytes;
  }
  opennpux_npu_functional_request request = {};
  return Materialize(command_index, operands, 2, &request) &&
         request.opcode == OPENNPUX_NPU_OP_NORMALIZE &&
         Execute(&request, regions, 2);
}

bool Gem5HostFunctionalGraph::ExecuteLinearAttentionRecurrent(
    uint32_t command_index, Gem5HostWeightProvider* weights,
    const std::vector<Gem5HostWeightBinding>& bindings) {
  if (!configured_ || weights == nullptr || bindings.size() != 2) {
    return false;
  }
  const uint32_t slots[] = {
      OPENNPUX_NPU_WEIGHT_SLOT_A_LOG,
      OPENNPUX_NPU_WEIGHT_SLOT_DT_BIAS};
  const uint32_t operand_roles[] = {
      OPENNPUX_NPU_OPERAND_LINEAR_A_LOG_WEIGHT,
      OPENNPUX_NPU_OPERAND_LINEAR_DT_BIAS_WEIGHT};
  std::vector<float> loaded[2];
  opennpux_npu_functional_operand operands[2] = {};
  Gem5FunctionalMemoryRegion regions[2] = {};
  uint64_t address = UINT32_C(0x50000000);
  for (size_t index = 0; index < 2; ++index) {
    const auto binding = std::find_if(
        bindings.begin(), bindings.end(), [&](const auto& candidate) {
          return candidate.role_id == OPENNPUX_NPU_WEIGHT_ROLE_LINEAR_DECAY &&
                 candidate.expert_id == OPENNPUX_NPU_WEIGHT_EXPERT_NONE &&
                 candidate.slot_id == slots[index];
        });
    if (binding == bindings.end() ||
        !weights->LoadFloatWeight(command_index, binding->role_id,
                                  binding->expert_id, binding->slot_id,
                                  &loaded[index]) ||
        loaded[index].empty() ||
        loaded[index].size() > UINT32_MAX / sizeof(float)) {
      return false;
    }
    address = (address + 63) & ~UINT64_C(63);
    const size_t bytes = loaded[index].size() * sizeof(float);
    if (address + bytes > (UINT64_C(1) << 32)) {
      return false;
    }
    operands[index] = {operand_roles[index], static_cast<uint32_t>(address),
                       static_cast<uint32_t>(bytes), 0};
    regions[index] = {static_cast<uint32_t>(address),
                      reinterpret_cast<uint8_t*>(loaded[index].data()), bytes};
    address += bytes;
  }
  opennpux_npu_functional_request request = {};
  return Materialize(command_index, operands, 2, &request) &&
         request.opcode == OPENNPUX_NPU_OP_RECURRENT_UPDATE &&
         Execute(&request, regions, 2);
}

bool Gem5HostFunctionalGraph::ExecuteFloatSharedExpert(
    uint32_t command_index, Gem5HostWeightProvider* weights,
    const std::vector<Gem5HostWeightBinding>& bindings) {
  if (!configured_ || weights == nullptr || bindings.size() != 4) {
    return false;
  }
  const uint32_t slots[] = {OPENNPUX_NPU_WEIGHT_SLOT_GATE_PROJ,
                            OPENNPUX_NPU_WEIGHT_SLOT_UP_PROJ,
                            OPENNPUX_NPU_WEIGHT_SLOT_DOWN_PROJ,
                            OPENNPUX_NPU_WEIGHT_SLOT_DEFAULT};
  const uint32_t operand_roles[] = {
      OPENNPUX_NPU_OPERAND_SHARED_GATE_WEIGHT,
      OPENNPUX_NPU_OPERAND_SHARED_UP_WEIGHT,
      OPENNPUX_NPU_OPERAND_SHARED_DOWN_WEIGHT,
      OPENNPUX_NPU_OPERAND_SHARED_ROUTER_WEIGHT};
  std::vector<float> loaded[4];
  opennpux_npu_functional_operand operands[4] = {};
  Gem5FunctionalMemoryRegion regions[4] = {};
  uint64_t address = UINT32_C(0x50000000);
  for (size_t index = 0; index < 4; ++index) {
    const auto binding = std::find_if(
        bindings.begin(), bindings.end(), [&](const auto& candidate) {
          return candidate.role_id ==
                     OPENNPUX_NPU_WEIGHT_ROLE_SHARED_EXPERT &&
                 candidate.expert_id == OPENNPUX_NPU_WEIGHT_EXPERT_NONE &&
                 candidate.slot_id == slots[index];
        });
    if (binding == bindings.end() ||
        !weights->LoadFloatWeight(command_index, binding->role_id,
                                  binding->expert_id, binding->slot_id,
                                  &loaded[index]) ||
        loaded[index].empty() ||
        loaded[index].size() > UINT32_MAX / sizeof(float)) {
      return false;
    }
    address = (address + 63) & ~UINT64_C(63);
    const size_t bytes = loaded[index].size() * sizeof(float);
    if (address + bytes > (UINT64_C(1) << 32)) {
      return false;
    }
    operands[index] = {operand_roles[index], static_cast<uint32_t>(address),
                       static_cast<uint32_t>(bytes), 0};
    regions[index] = {static_cast<uint32_t>(address),
                      reinterpret_cast<uint8_t*>(loaded[index].data()), bytes};
    address += bytes;
  }
  opennpux_npu_functional_request request = {};
  return Materialize(command_index, operands, 4, &request) &&
         request.opcode == OPENNPUX_NPU_OP_EXPERT &&
         Execute(&request, regions, 4);
}

bool Gem5HostFunctionalGraph::ExecuteFloatQkv(
    uint32_t command_index, Gem5HostWeightProvider* weights,
    const std::vector<Gem5HostWeightBinding>& bindings) {
  if (!configured_ || weights == nullptr || bindings.size() != 5) {
    return false;
  }
  const uint32_t semantic_roles[] = {
      OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_Q_PROJ,
      OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_K_PROJ,
      OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_V_PROJ,
      OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_Q_NORM,
      OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_K_NORM};
  const uint32_t operand_roles[] = {
      OPENNPUX_NPU_OPERAND_ATTENTION_Q_WEIGHT,
      OPENNPUX_NPU_OPERAND_ATTENTION_K_WEIGHT,
      OPENNPUX_NPU_OPERAND_ATTENTION_V_WEIGHT,
      OPENNPUX_NPU_OPERAND_ATTENTION_Q_NORM_WEIGHT,
      OPENNPUX_NPU_OPERAND_ATTENTION_K_NORM_WEIGHT};
  std::vector<float> loaded[5];
  opennpux_npu_functional_operand operands[5] = {};
  Gem5FunctionalMemoryRegion regions[5] = {};
  uint64_t address = UINT32_C(0x50000000);
  for (size_t index = 0; index < 5; ++index) {
    const auto binding = std::find_if(
        bindings.begin(), bindings.end(), [&](const auto& candidate) {
          return candidate.role_id == semantic_roles[index] &&
                 candidate.expert_id == OPENNPUX_NPU_WEIGHT_EXPERT_NONE;
        });
    if (binding == bindings.end() ||
        !weights->LoadFloatWeight(command_index, binding->role_id,
                                  binding->expert_id, binding->slot_id,
                                  &loaded[index]) ||
        loaded[index].empty() ||
        loaded[index].size() > UINT32_MAX / sizeof(float)) {
      return false;
    }
    address = (address + 63) & ~UINT64_C(63);
    const size_t bytes = loaded[index].size() * sizeof(float);
    if (address + bytes > (UINT64_C(1) << 32)) {
      return false;
    }
    operands[index] = {operand_roles[index], static_cast<uint32_t>(address),
                       static_cast<uint32_t>(bytes), 0};
    regions[index] = {static_cast<uint32_t>(address),
                      reinterpret_cast<uint8_t*>(loaded[index].data()), bytes};
    address += bytes;
  }
  opennpux_npu_functional_request request = {};
  return Materialize(command_index, operands, 5, &request) &&
         request.opcode == OPENNPUX_NPU_OP_MATMUL &&
         Execute(&request, regions, 5);
}

bool Gem5HostFunctionalGraph::ExecutePositioned(uint32_t command_index) {
  opennpux_npu_functional_request initial = {};
  if (!Materialize(command_index, nullptr, 0, &initial) || initial.rows == 0 ||
      initial.rows > UINT32_MAX / sizeof(uint32_t)) {
    return false;
  }
  std::vector<uint32_t> positions(initial.rows);
  for (uint32_t row = 0; row < initial.rows; ++row) {
    positions[row] = row;
  }
  constexpr uint32_t kPositionAddress = UINT32_C(0x50000000);
  const opennpux_npu_functional_operand operand = {
      OPENNPUX_NPU_OPERAND_POSITIONS, kPositionAddress,
      static_cast<uint32_t>(positions.size() * sizeof(uint32_t)), 0};
  Gem5FunctionalMemoryRegion region = {
      kPositionAddress, reinterpret_cast<uint8_t*>(positions.data()),
      positions.size() * sizeof(uint32_t)};
  opennpux_npu_functional_request request = {};
  return Materialize(command_index, &operand, 1, &request) &&
         Execute(&request, &region, 1);
}

bool Gem5HostFunctionalGraph::ExecuteCommand(
    uint32_t command_index, Gem5HostWeightProvider* weights) {
  const auto* selected = command(command_index);
  if (selected == nullptr || weights == nullptr || !weights->loaded()) {
    return false;
  }
  if (selected->opcode == OPENNPUX_NPU_OP_ROUTER) {
    return ExecuteGptqRouter(command_index, weights);
  }
  if (selected->opcode == OPENNPUX_NPU_OP_ROPE) {
    return ExecutePositioned(command_index);
  }
  std::vector<Gem5HostWeightBinding> gptq;
  const bool has_gptq = weights->FindGptqBindings(command_index, &gptq);
  if (selected->opcode == OPENNPUX_NPU_OP_MATMUL && has_gptq) {
    const auto has_role = [&](uint32_t role) {
      return std::find_if(gptq.begin(), gptq.end(),
                          [&](const auto& binding) {
                            return binding.role_id == role;
                          }) != gptq.end();
    };
    if (has_role(OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_Q_PROJ) &&
        has_role(OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_K_PROJ) &&
        has_role(OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_V_PROJ)) {
      return ExecuteGptqQkv(command_index, weights);
    }
    if (gptq.size() == 1) {
      return ExecuteGptqProjection(command_index, weights, gptq[0].role_id,
                                   gptq[0].expert_id, gptq[0].slot_id);
    }
    return false;
  }
  if (selected->opcode == OPENNPUX_NPU_OP_EXPERT && has_gptq) {
    const auto routed = std::find_if(
        gptq.begin(), gptq.end(), [](const auto& binding) {
          return binding.role_id == OPENNPUX_NPU_WEIGHT_ROLE_ROUTED_EXPERT &&
                 binding.expert_id != OPENNPUX_NPU_WEIGHT_EXPERT_NONE;
        });
    if (routed != gptq.end()) {
      return ExecuteRoutedExpert(command_index, weights);
    }
    const auto direct = std::find_if(
        gptq.begin(), gptq.end(), [](const auto& binding) {
          return binding.slot_id == OPENNPUX_NPU_WEIGHT_SLOT_GATE_PROJ;
        });
    return direct != gptq.end() &&
           ExecuteGptqExpert(command_index, weights, *direct);
  }
  std::vector<Gem5HostWeightBinding> floating;
  if (weights->FindFloatBindings(command_index, &floating)) {
    const auto has_role = [&](uint32_t role) {
      return std::any_of(floating.begin(), floating.end(),
                         [&](const auto& binding) {
                           return binding.role_id == role;
                         });
    };
    if (selected->opcode == OPENNPUX_NPU_OP_MATMUL &&
        floating.size() == 5 &&
        has_role(OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_Q_PROJ) &&
        has_role(OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_K_PROJ) &&
        has_role(OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_V_PROJ) &&
        has_role(OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_Q_NORM) &&
        has_role(OPENNPUX_NPU_WEIGHT_ROLE_ATTENTION_K_NORM)) {
      return ExecuteFloatQkv(command_index, weights, floating);
    }
    if (floating.size() == 3 &&
        has_role(OPENNPUX_NPU_WEIGHT_ROLE_LINEAR_QKV) &&
        has_role(OPENNPUX_NPU_WEIGHT_ROLE_LINEAR_ALPHA) &&
        has_role(OPENNPUX_NPU_WEIGHT_ROLE_LINEAR_BETA)) {
      return ExecuteLinearAttentionProjection(command_index, weights,
                                              floating);
    }
    if (floating.size() == 2 &&
        has_role(OPENNPUX_NPU_WEIGHT_ROLE_LINEAR_GATE) &&
        has_role(OPENNPUX_NPU_WEIGHT_ROLE_LINEAR_NORM)) {
      return ExecuteLinearAttentionGateNorm(command_index, weights, floating);
    }
    if (selected->opcode == OPENNPUX_NPU_OP_RECURRENT_UPDATE &&
        floating.size() == 2 &&
        std::all_of(floating.begin(), floating.end(), [](const auto& binding) {
          return binding.role_id ==
                     OPENNPUX_NPU_WEIGHT_ROLE_LINEAR_DECAY &&
                 binding.expert_id == OPENNPUX_NPU_WEIGHT_EXPERT_NONE;
        })) {
      return ExecuteLinearAttentionRecurrent(command_index, weights,
                                             floating);
    }
    if (selected->opcode == OPENNPUX_NPU_OP_EXPERT &&
        floating.size() == 4 &&
        std::all_of(floating.begin(), floating.end(), [](const auto& binding) {
          return binding.role_id ==
                     OPENNPUX_NPU_WEIGHT_ROLE_SHARED_EXPERT &&
                 binding.expert_id == OPENNPUX_NPU_WEIGHT_EXPERT_NONE;
        })) {
      return ExecuteFloatSharedExpert(command_index, weights, floating);
    }
    return floating.size() == 1 &&
           ExecuteFloatWeight(command_index, weights, floating[0]);
  }
  if (selected->opcode == OPENNPUX_NPU_OP_EMBED ||
      selected->opcode == OPENNPUX_NPU_OP_MATMUL ||
      selected->opcode == OPENNPUX_NPU_OP_NORMALIZE ||
      selected->opcode == OPENNPUX_NPU_OP_ROUTER ||
      selected->opcode == OPENNPUX_NPU_OP_EXPERT) {
    return false;
  }
  opennpux_npu_functional_request request = {};
  return Materialize(command_index, nullptr, 0, &request) && Execute(&request);
}

bool Gem5HostFunctionalGraph::ExecuteProgram(
    Gem5HostWeightProvider* weights, uint32_t* failed_command) {
  if (failed_command != nullptr) {
    *failed_command = UINT32_MAX;
  }
  if (!configured_ || weights == nullptr || !weights->loaded()) {
    return false;
  }
  for (uint32_t index = 0; index < command_count(); ++index) {
    if (!ExecuteCommand(index, weights)) {
      if (failed_command != nullptr) {
        *failed_command = index;
      }
      return false;
    }
  }
  return true;
}

bool Gem5HostFunctionalGraph::SetInputTokenIds(
    const uint32_t* token_ids, size_t token_count) {
  opennpux_npu_functional_request request = {};
  if (!configured_ || token_ids == nullptr || token_count == 0 ||
      !Materialize(0, nullptr, 0, &request) ||
      request.opcode != OPENNPUX_NPU_OP_EMBED) {
    return false;
  }
  const auto* input = FindOperand(request, OPENNPUX_NPU_OPERAND_INPUT_INDICES);
  if (input == nullptr || token_count != request.rows ||
      input->byte_size < token_count * sizeof(uint32_t)) {
    return false;
  }
  auto* destination = arena_.Translate(input->address, input->byte_size);
  if (destination == nullptr) {
    return false;
  }
  std::memcpy(destination, token_ids, token_count * sizeof(uint32_t));
  return true;
}

bool Gem5HostFunctionalGraph::ReadNextToken(uint32_t* token_id) const {
  if (!configured_ || token_id == nullptr) {
    return false;
  }
  for (uint32_t offset = 0; offset < command_count(); ++offset) {
    const uint32_t index = command_count() - 1 - offset;
    if (command(index)->opcode != OPENNPUX_NPU_OP_TOPK) {
      continue;
    }
    opennpux_npu_functional_request request = {};
    if (!Materialize(index, nullptr, 0, &request) || request.rows == 0 ||
        request.top_k == 0) {
      return false;
    }
    const auto* output =
        FindOperand(request, OPENNPUX_NPU_OPERAND_OUTPUT_INDICES);
    if (output == nullptr) {
      return false;
    }
    const uint64_t output_count = output->byte_size / sizeof(uint32_t);
    if (output_count < request.top_k) {
      return false;
    }
    const uint64_t selected = output_count - request.top_k;
    const auto* values = reinterpret_cast<const uint32_t*>(
        arena_.Translate(output->address, output->byte_size));
    if (values == nullptr) {
      return false;
    }
    *token_id = values[selected];
    return true;
  }
  return false;
}

const opennpux_npu_command* Gem5HostFunctionalGraph::command(
    uint32_t index) const {
  return configured_ && index < program_.header->command_count
             ? &program_.commands[index]
             : nullptr;
}

void Gem5HostFunctionalGraph::ResetInvocation() {
  submission_.clear();
  submission_base_ = 0;
  program_ = {};
  stats_ = {};
  configured_ = false;
  arena_.Reset();
}
