#include "hw_sim/gem5_bridge/gem5_host_functional_graph.h"

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
  ResetInvocation();
  if (!arena_.loaded() || submission == nullptr || submission_size == 0 ||
      submission_size > UINT32_MAX ||
      static_cast<uint64_t>(submission_base) + submission_size >
          (UINT64_C(1) << 32)) {
    return false;
  }
  opennpux_npu_tensor_plan_runtime runtime = {};
  if (!DecodeRuntime(submission, submission_size, &runtime)) {
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
  opennpux_npu_functional_operand operands[12] = {};
  Gem5FunctionalMemoryRegion regions[12] = {};
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
  std::vector<float> weight;
  if (!weights->LoadFloatWeight(command_index,
                                OPENNPUX_NPU_WEIGHT_ROLE_ROUTER,
                                OPENNPUX_NPU_WEIGHT_EXPERT_NONE,
                                OPENNPUX_NPU_WEIGHT_SLOT_DEFAULT, &weight) ||
      weight.empty() || weight.size() > UINT32_MAX / sizeof(float)) {
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
