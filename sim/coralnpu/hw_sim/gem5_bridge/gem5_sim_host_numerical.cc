#include "hw_sim/gem5_bridge/gem5_sim_host_numerical.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "hw_sim/gem5_bridge/npu_inference_io.h"
#include "hw_sim/gem5_bridge/npu_submission.h"

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

}  // namespace

Gem5SimHostNumerical::Gem5SimHostNumerical() {
  const char* path = std::getenv("CORAL_SIM_HOST_INFERENCE_RESULT");
  if (path == nullptr || path[0] == '\0') {
    return;
  }
  configured_ = true;
  FILE* file = std::fopen(path, "rb");
  valid_ = file != nullptr &&
      std::fread(&result_, 1, sizeof(result_), file) == sizeof(result_) &&
      std::fgetc(file) == EOF &&
      result_.magic == OPENNPUX_NPU_NUMERICAL_RESULT_MAGIC &&
      result_.version == OPENNPUX_NPU_NUMERICAL_RESULT_VERSION &&
      result_.struct_size == sizeof(result_) && result_.executable_id != 0 &&
      result_.vocabulary_size != 0 &&
      result_.next_token < result_.vocabulary_size &&
      result_.logits_count == result_.vocabulary_size &&
      result_.generated_token_count != 0 &&
      result_.generated_token_count <=
          OPENNPUX_NPU_NUMERICAL_RESULT_MAX_TOKENS &&
      result_.token_text_size != 0 &&
      result_.token_text_size <= OPENNPUX_NPU_INFERENCE_PROMPT_BYTES;
  if (valid_) {
    for (uint32_t index = 0; index < result_.generated_token_count; ++index) {
      valid_ = valid_ &&
          result_.generated_token_ids[index] < result_.vocabulary_size;
    }
    valid_ = valid_ && result_.generated_token_ids[0] == result_.next_token;
  }
  if (file != nullptr) {
    std::fclose(file);
  }
  if (!valid_) {
    std::fprintf(stderr,
                 "Coral sim-host numerical result is invalid path=%s\n", path);
  } else {
    std::fprintf(stderr,
                 "Coral sim-host numerical loaded executable=0x%016llx "
                 "prompt=0x%08x token=%u generated=%u logits=0x%08x\n",
                 static_cast<unsigned long long>(result_.executable_id),
                 result_.prompt_checksum, result_.next_token,
                 result_.generated_token_count,
                 result_.logits_checksum);
  }
  std::fflush(stderr);
}

void Gem5SimHostNumerical::Reset() {
  published_ = false;
}

int Gem5SimHostNumerical::Publish(std::vector<uint8_t>* extmem) {
  if (!configured_) {
    return 0;
  }
  if (!valid_ || extmem == nullptr ||
      extmem->size() < sizeof(opennpux_npu_invocation_header)) {
    return -1;
  }
  if (published_) {
    return 0;
  }
  const auto* invocation = reinterpret_cast<
      const opennpux_npu_invocation_header*>(extmem->data());
  if (invocation->magic != OPENNPUX_NPU_INVOCATION_MAGIC ||
      invocation->version != OPENNPUX_NPU_INVOCATION_VERSION ||
      (invocation->flags & OPENNPUX_NPU_INVOKE_INFERENCE_IO) == 0 ||
      invocation->binding_count < 2 ||
      invocation->binding_offset > extmem->size() ||
      invocation->binding_count >
          (extmem->size() - invocation->binding_offset) /
              sizeof(opennpux_npu_tensor_binding)) {
    return 0;
  }
  if (invocation->executable_id != result_.executable_id) {
    std::fprintf(stderr,
                 "Coral sim-host numerical executable mismatch "
                 "request=0x%016llx result=0x%016llx\n",
                 static_cast<unsigned long long>(invocation->executable_id),
                 static_cast<unsigned long long>(result_.executable_id));
    return -1;
  }
  const auto* bindings = reinterpret_cast<const opennpux_npu_tensor_binding*>(
      extmem->data() + invocation->binding_offset);
  const uint64_t token_ids_bytes =
      static_cast<uint64_t>(result_.generated_token_count) * sizeof(uint32_t);
  if (!RangeValid(bindings[0].device_address,
                  sizeof(opennpux_npu_inference_io), *extmem) ||
      !RangeValid(bindings[1].device_address,
                  sizeof(opennpux_npu_inference_io), *extmem) ||
      bindings[1].byte_size <
          OPENNPUX_NPU_INFERENCE_TOKEN_IDS_OFFSET + token_ids_bytes ||
      !RangeValid(bindings[1].device_address +
                      OPENNPUX_NPU_INFERENCE_TOKEN_IDS_OFFSET,
                  token_ids_bytes, *extmem)) {
    return -1;
  }
  const auto* input = At<opennpux_npu_inference_io>(
      extmem, bindings[0].device_address);
  auto* output = At<opennpux_npu_inference_io>(
      extmem, bindings[1].device_address);
  const uint64_t expected_commands =
      static_cast<uint64_t>(invocation->command_count) * input->max_new_tokens;
  if (expected_commands > UINT32_MAX) {
    return -1;
  }
  if (output->state != OPENNPUX_NPU_INFERENCE_COMPLETE ||
      output->completed_commands != expected_commands) {
    return 0;
  }
  if (input->magic != OPENNPUX_NPU_INFERENCE_IO_MAGIC ||
      input->version != OPENNPUX_NPU_INFERENCE_IO_VERSION ||
      input->struct_size != sizeof(*input) ||
      input->prompt_checksum != result_.prompt_checksum ||
      input->vocabulary_size != result_.vocabulary_size ||
      input->input_token_count != result_.input_token_count ||
      input->max_new_tokens == 0 ||
      result_.generated_token_count > input->max_new_tokens ||
      output->magic != OPENNPUX_NPU_INFERENCE_IO_MAGIC ||
      output->version != OPENNPUX_NPU_INFERENCE_IO_VERSION ||
      output->struct_size != sizeof(*output)) {
    std::fprintf(stderr,
                 "Coral sim-host numerical prompt/model mismatch "
                 "prompt=0x%08x expected=0x%08x vocab=%u expected_vocab=%u\n",
                 input->prompt_checksum, result_.prompt_checksum,
                 input->vocabulary_size, result_.vocabulary_size);
    return -1;
  }
  output->mode = OPENNPUX_NPU_INFERENCE_MODE_NUMERICAL;
  output->input_token_count = result_.input_token_count;
  output->next_token = result_.next_token;
  output->result_checksum = result_.logits_checksum;
  output->reserved[OPENNPUX_NPU_INFERENCE_TOKEN_IDS_SOURCE_INDEX] =
      OPENNPUX_NPU_INFERENCE_SOURCE_SIM_HOST;
  output->reserved[OPENNPUX_NPU_INFERENCE_TOKEN_TEXT_SIZE_INDEX] =
      result_.token_text_size;
  output->reserved[OPENNPUX_NPU_INFERENCE_TOKEN_COUNT_INDEX] =
      result_.generated_token_count;
  output->reserved[OPENNPUX_NPU_INFERENCE_STOP_REASON_INDEX] =
      result_.stop_reason;
  output->reserved[OPENNPUX_NPU_INFERENCE_TOKEN_IDS_OFFSET_INDEX] =
      OPENNPUX_NPU_INFERENCE_TOKEN_IDS_OFFSET;
  output->reserved[OPENNPUX_NPU_INFERENCE_TOKEN_IDS_BYTES_INDEX] =
      token_ids_bytes;
  auto* token_ids = At<uint32_t>(
      extmem, bindings[1].device_address +
                  OPENNPUX_NPU_INFERENCE_TOKEN_IDS_OFFSET);
  std::memcpy(token_ids, result_.generated_token_ids, token_ids_bytes);
  std::memset(output->prompt, 0, sizeof(output->prompt));
  std::memcpy(output->prompt, result_.token_text, result_.token_text_size);
  __atomic_thread_fence(__ATOMIC_RELEASE);
  published_ = true;
  std::fprintf(stderr,
               "Coral sim-host numerical publish token=%u generated=%u "
               "logits=0x%08x commands=%u\n",
               output->next_token, output->reserved[2], output->result_checksum,
               output->completed_commands);
  std::fflush(stderr);
  return 1;
}
