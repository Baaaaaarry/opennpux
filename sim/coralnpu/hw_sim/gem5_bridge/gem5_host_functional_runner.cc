#include <cerrno>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
  bool ready = instantiated && graph.LoadTensorPlan(argv[2]) &&
               weights.Load(argv[3], argv[4]);
  std::vector<uint32_t> generated;
  for (uint32_t step = 0; ready && step < max_new_tokens; ++step) {
    const opennpux_npu_tensor_plan_runtime runtime = {
        1, static_cast<uint32_t>(tokens.size()),
        static_cast<uint32_t>(tokens.size()), active_experts};
    uint32_t failed_command = UINT32_MAX;
    uint32_t next_token = 0;
    ready = graph.ConfigureRuntime(submission, submission_size,
                                   UINT32_C(0x24000000), runtime) &&
            graph.SetInputTokenIds(tokens.data(), tokens.size()) &&
            graph.ExecuteProgram(&weights, &failed_command) &&
            graph.ReadNextToken(&next_token);
    if (!ready) {
      std::fprintf(stderr,
                   "host_functional_failed_step=%u command=%u\n", step,
                   failed_command);
      break;
    }
    generated.push_back(next_token);
    tokens.push_back(next_token);
    std::fprintf(stderr,
                 "host_functional_step=%u token=%u cycles=%llu\n", step,
                 next_token,
                 static_cast<unsigned long long>(graph.stats().modeled_cycles));
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
