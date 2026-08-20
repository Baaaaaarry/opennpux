#include "hw_sim/gem5_bridge/gem5_host_functional_graph.h"

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
