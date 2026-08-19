#include "hw_sim/gem5_bridge/gem5_sim_host_numerical.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <unistd.h>

#include "hw_sim/gem5_bridge/npu_inference_io.h"
#include "hw_sim/gem5_bridge/npu_submission.h"

namespace {

constexpr uint64_t kBase = UINT64_C(0x20000000);

template <typename T>
T* At(std::vector<uint8_t>* memory, size_t offset) {
  return reinterpret_cast<T*>(memory->data() + offset);
}

}  // namespace

int main() {
  char path[] = "/tmp/opennpux-numerical-result-XXXXXX";
  const int fd = mkstemp(path);
  assert(fd >= 0);
  FILE* file = fdopen(fd, "wb");
  assert(file != nullptr);
  opennpux_npu_numerical_result result = {};
  result.magic = OPENNPUX_NPU_NUMERICAL_RESULT_MAGIC;
  result.version = OPENNPUX_NPU_NUMERICAL_RESULT_VERSION;
  result.struct_size = sizeof(result);
  result.executable_id = UINT64_C(0x1122334455667788);
  result.prompt_checksum = UINT32_C(0xaabbccdd);
  result.next_token = 7;
  result.vocabulary_size = 16;
  result.logits_checksum = UINT32_C(0x12345678);
  result.input_token_count = 4;
  result.token_text_size = 5;
  std::memcpy(result.token_text, " npux", 5);
  result.logits_count = 16;
  result.generated_token_count = 3;
  result.generated_token_ids[0] = 7;
  result.generated_token_ids[1] = 8;
  result.generated_token_ids[2] = 9;
  assert(fwrite(&result, 1, sizeof(result), file) == sizeof(result));
  assert(fclose(file) == 0);
  assert(setenv("CORAL_SIM_HOST_INFERENCE_RESULT", path, 1) == 0);

  std::vector<uint8_t> extmem(0x10000, 0);
  auto* invocation = At<opennpux_npu_invocation_header>(&extmem, 0);
  invocation->magic = OPENNPUX_NPU_INVOCATION_MAGIC;
  invocation->version = OPENNPUX_NPU_INVOCATION_VERSION;
  invocation->header_size = sizeof(*invocation);
  invocation->flags = OPENNPUX_NPU_INVOKE_INFERENCE_IO;
  invocation->executable_id = result.executable_id;
  invocation->binding_offset = 0x100;
  invocation->binding_count = 2;
  invocation->command_count = 524;
  auto* bindings = At<opennpux_npu_tensor_binding>(&extmem, 0x100);
  bindings[0].device_address = kBase + 0x1000;
  bindings[0].byte_size = sizeof(opennpux_npu_inference_io);
  bindings[1].device_address = kBase + 0x2000;
  bindings[1].byte_size = sizeof(opennpux_npu_inference_io);
  auto* input = At<opennpux_npu_inference_io>(&extmem, 0x1000);
  input->magic = OPENNPUX_NPU_INFERENCE_IO_MAGIC;
  input->version = OPENNPUX_NPU_INFERENCE_IO_VERSION;
  input->struct_size = sizeof(*input);
  input->prompt_checksum = result.prompt_checksum;
  input->vocabulary_size = result.vocabulary_size;
  input->max_new_tokens = 3;
  auto* output = At<opennpux_npu_inference_io>(&extmem, 0x2000);
  output->magic = OPENNPUX_NPU_INFERENCE_IO_MAGIC;
  output->version = OPENNPUX_NPU_INFERENCE_IO_VERSION;
  output->struct_size = sizeof(*output);
  output->state = OPENNPUX_NPU_INFERENCE_COMPLETE;
  output->completed_commands = 524;

  Gem5SimHostNumerical numerical;
  assert(numerical.enabled());
  assert(numerical.Publish(&extmem) == 1);
  assert(output->mode == OPENNPUX_NPU_INFERENCE_MODE_NUMERICAL);
  assert(output->next_token == result.next_token);
  assert(output->result_checksum == result.logits_checksum);
  assert(output->input_token_count == result.input_token_count);
  assert(output->reserved[0] == OPENNPUX_NPU_INFERENCE_SOURCE_SIM_HOST);
  assert(output->reserved[1] == result.token_text_size);
  assert(output->reserved[2] == result.generated_token_count);
  assert(output->reserved[3] == result.stop_reason);
  assert(output->reserved[4] == 7);
  assert(output->reserved[5] == 8);
  assert(output->reserved[6] == 9);
  assert(std::memcmp(output->prompt, " npux", 5) == 0);
  assert(numerical.Publish(&extmem) == 0);
  assert(unlink(path) == 0);
  puts("gem5_sim_host_numerical=PASS");
  return 0;
}
