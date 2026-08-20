#include "hw_sim/gem5_bridge/gem5_sim_host_functional.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hw_sim/gem5_bridge/gem5_host_functional_graph.h"
#include "hw_sim/gem5_bridge/gem5_host_weight_provider.h"
#include "hw_sim/gem5_bridge/npu_inference_io.h"
#include "opennpux/npu_submission.h"

namespace {

constexpr uint64_t kExtmemBase = UINT64_C(0x20000000);

bool RangeValid(uint64_t address, uint64_t size,
                const std::vector<uint8_t>& extmem) {
  return address >= kExtmemBase && address - kExtmemBase <= extmem.size() &&
         size <= extmem.size() - (address - kExtmemBase);
}

template <typename T>
T* At(std::vector<uint8_t>* memory, uint64_t address) {
  return reinterpret_cast<T*>(memory->data() + (address - kExtmemBase));
}

uint32_t TokenChecksum(const uint32_t* tokens, size_t count) {
  uint32_t value = UINT32_C(2166136261);
  const auto* bytes = reinterpret_cast<const uint8_t*>(tokens);
  for (size_t index = 0; index < count * sizeof(*tokens); ++index) {
    value = (value ^ bytes[index]) * UINT32_C(16777619);
  }
  return value;
}

}  // namespace

struct Gem5SimHostFunctional::Impl {
  Gem5HostFunctionalGraph graph;
  Gem5HostWeightProvider weights;
  bool configured = false;
  bool valid = false;
  bool published = false;

  Impl() {
    const char* model = std::getenv("CORAL_HOST_FUNCTIONAL_MODEL");
    const char* ranges = std::getenv("CORAL_HOST_FUNCTIONAL_RANGES");
    const char* tensor_plan = std::getenv("CORAL_HOST_FUNCTIONAL_TENSOR_PLAN");
    configured = model != nullptr || ranges != nullptr || tensor_plan != nullptr;
    if (!configured) {
      return;
    }
    valid = model != nullptr && model[0] != '\0' && ranges != nullptr &&
            ranges[0] != '\0' && tensor_plan != nullptr &&
            tensor_plan[0] != '\0' && graph.LoadTensorPlan(tensor_plan) &&
            weights.Load(model, ranges);
    std::fprintf(stderr, "Coral host functional backend %s model='%s' plan='%s'\n",
                 valid ? "enabled" : "invalid", model != nullptr ? model : "",
                 tensor_plan != nullptr ? tensor_plan : "");
    std::fflush(stderr);
  }

  int Service(std::vector<uint8_t>* extmem) {
    if (!configured) {
      return 0;
    }
    if (!valid || extmem == nullptr ||
        extmem->size() < sizeof(opennpux_npu_invocation_header)) {
      return -1;
    }
    if (published) {
      return 0;
    }
    const auto* invocation = reinterpret_cast<
        const opennpux_npu_invocation_header*>(extmem->data());
    if (invocation->magic != OPENNPUX_NPU_INVOCATION_MAGIC ||
        invocation->version != OPENNPUX_NPU_INVOCATION_VERSION ||
        (invocation->flags & OPENNPUX_NPU_INVOKE_INFERENCE_IO) == 0) {
      return 0;
    }
    if (invocation->total_size < sizeof(*invocation) ||
        invocation->total_size > extmem->size() ||
        invocation->binding_count < 2 ||
        invocation->command_count == 0 ||
        invocation->binding_offset > invocation->total_size ||
        invocation->binding_count >
            (invocation->total_size - invocation->binding_offset) /
                sizeof(opennpux_npu_tensor_binding) ||
        invocation->command_offset > invocation->total_size ||
        invocation->command_count >
            (invocation->total_size - invocation->command_offset) /
                sizeof(opennpux_npu_command)) {
      return -1;
    }
    const auto* bindings = reinterpret_cast<const opennpux_npu_tensor_binding*>(
        extmem->data() + invocation->binding_offset);
    if (!RangeValid(bindings[0].device_address, bindings[0].byte_size, *extmem) ||
        !RangeValid(bindings[1].device_address, bindings[1].byte_size, *extmem) ||
        bindings[0].byte_size < sizeof(opennpux_npu_inference_io) ||
        bindings[1].byte_size < sizeof(opennpux_npu_inference_io)) {
      return -1;
    }
    const auto* input = At<opennpux_npu_inference_io>(
        extmem, bindings[0].device_address);
    auto* output = At<opennpux_npu_inference_io>(
        extmem, bindings[1].device_address);
    if (output->state != OPENNPUX_NPU_INFERENCE_COMPLETE) {
      return 0;
    }
    const uint64_t token_bytes =
        static_cast<uint64_t>(input->input_token_count) * sizeof(uint32_t);
    if (input->magic != OPENNPUX_NPU_INFERENCE_IO_MAGIC ||
        input->version != OPENNPUX_NPU_INFERENCE_IO_VERSION ||
        input->struct_size != sizeof(*input) || input->input_token_count == 0 ||
        input->max_new_tokens == 0 ||
        input->max_new_tokens > OPENNPUX_NPU_INFERENCE_MAX_RESULT_TOKENS ||
        bindings[0].byte_size <
            OPENNPUX_NPU_INFERENCE_TOKEN_IDS_OFFSET + token_bytes ||
        !RangeValid(bindings[0].device_address +
                        OPENNPUX_NPU_INFERENCE_TOKEN_IDS_OFFSET,
                    token_bytes, *extmem)) {
      return -1;
    }
    const auto* input_tokens = At<uint32_t>(
        extmem, bindings[0].device_address +
                    OPENNPUX_NPU_INFERENCE_TOKEN_IDS_OFFSET);
    std::vector<uint32_t> tokens;
    std::vector<uint32_t> generated;
    try {
      tokens.assign(input_tokens, input_tokens + input->input_token_count);
      tokens.reserve(tokens.size() + input->max_new_tokens);
      generated.reserve(input->max_new_tokens);
    } catch (...) {
      return -1;
    }
    const auto* commands = reinterpret_cast<const opennpux_npu_command*>(
        extmem->data() + invocation->command_offset);
    const uint64_t initial_shape = commands[0].runtime_shape;
    const uint32_t active_experts =
        (initial_shape >> OPENNPUX_NPU_RUNTIME_EXPERT_SHIFT) &
        OPENNPUX_NPU_RUNTIME_FIELD_MASK;
    uint64_t modeled_cycles = 0;
    uint32_t completed_commands = 0;
    for (uint32_t step = 0; step < input->max_new_tokens; ++step) {
      if (tokens.size() > OPENNPUX_NPU_RUNTIME_FIELD_MASK) {
        return -1;
      }
      const opennpux_npu_tensor_plan_runtime runtime = {
          1, static_cast<uint32_t>(tokens.size()),
          static_cast<uint32_t>(tokens.size()), active_experts};
      uint32_t failed_command = UINT32_MAX;
      uint32_t next_token = 0;
      if (!graph.ConfigureRuntime(extmem->data(), invocation->total_size,
                                  static_cast<uint32_t>(kExtmemBase), runtime) ||
          !graph.SetInputTokenIds(tokens.data(), tokens.size()) ||
          !graph.ExecuteProgram(&weights, &failed_command) ||
          !graph.ReadNextToken(&next_token) ||
          next_token >= input->vocabulary_size) {
        std::fprintf(stderr,
                     "Coral host functional execution failed step=%u command=%u\n",
                     step, failed_command);
        std::fflush(stderr);
        return -1;
      }
      modeled_cycles += graph.stats().modeled_cycles;
      completed_commands += graph.stats().completed_commands;
      generated.push_back(next_token);
      tokens.push_back(next_token);
    }
    const uint64_t result_bytes = generated.size() * sizeof(uint32_t);
    if (bindings[1].byte_size <
        OPENNPUX_NPU_INFERENCE_TOKEN_IDS_OFFSET + result_bytes) {
      return -1;
    }
    auto* result_token = At<uint32_t>(
        extmem, bindings[1].device_address +
                    OPENNPUX_NPU_INFERENCE_TOKEN_IDS_OFFSET);
    std::memcpy(result_token, generated.data(), result_bytes);
    output->mode = OPENNPUX_NPU_INFERENCE_MODE_NUMERICAL;
    output->input_token_count = input->input_token_count;
    output->next_token = generated.front();
    output->result_checksum = TokenChecksum(result_token, generated.size());
    output->modeled_cycles = modeled_cycles;
    output->reserved[OPENNPUX_NPU_INFERENCE_TOKEN_IDS_SOURCE_INDEX] =
        OPENNPUX_NPU_INFERENCE_SOURCE_HOST_FUNCTIONAL;
    output->reserved[OPENNPUX_NPU_INFERENCE_TOKEN_COUNT_INDEX] =
        generated.size();
    output->reserved[OPENNPUX_NPU_INFERENCE_STOP_REASON_INDEX] = 0;
    output->reserved[OPENNPUX_NPU_INFERENCE_TOKEN_IDS_OFFSET_INDEX] =
        OPENNPUX_NPU_INFERENCE_TOKEN_IDS_OFFSET;
    output->reserved[OPENNPUX_NPU_INFERENCE_TOKEN_IDS_BYTES_INDEX] =
        result_bytes;
    __atomic_thread_fence(__ATOMIC_RELEASE);
    published = true;
    std::fprintf(stderr,
                 "Coral host functional publish token=%u generated=%zu "
                 "commands=%u cycles=%llu\n",
                 output->next_token, generated.size(), completed_commands,
                 static_cast<unsigned long long>(modeled_cycles));
    std::fflush(stderr);
    return 1;
  }
};

Gem5SimHostFunctional::Gem5SimHostFunctional()
    : impl_(std::make_unique<Impl>()) {}

Gem5SimHostFunctional::~Gem5SimHostFunctional() = default;

void Gem5SimHostFunctional::Reset() {
  if (impl_ != nullptr) {
    impl_->published = false;
  }
}

int Gem5SimHostFunctional::Service(std::vector<uint8_t>* extmem) {
  return impl_ == nullptr ? -1 : impl_->Service(extmem);
}

bool Gem5SimHostFunctional::enabled() const {
  return impl_ != nullptr && impl_->configured;
}
