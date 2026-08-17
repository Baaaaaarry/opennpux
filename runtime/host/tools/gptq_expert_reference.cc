#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <vector>

#include "hw_sim/gem5_bridge/coral_gptq_expert.h"
#include "hw_sim/gem5_bridge/coral_operator.h"
#include "hw_sim/gem5_bridge/gem5_hybrid_operator.h"

namespace {

constexpr uint32_t kExtmemBase = UINT32_C(0x20000000);

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: gptq-expert-reference <expert.bin>\n";
    return 2;
  }
  std::ifstream source(argv[1], std::ios::binary);
  if (!source) {
    std::cerr << "error: cannot open GPTQ expert image\n";
    return 1;
  }
  std::vector<uint8_t> image{
      std::istreambuf_iterator<char>(source), std::istreambuf_iterator<char>()};
  if (image.size() < sizeof(coral_gptq_expert_request)) {
    std::cerr << "error: cannot read GPTQ expert image\n";
    return 1;
  }

  coral_operator_descriptor descriptor = {};
  descriptor.magic = CORAL_OPERATOR_ABI_MAGIC;
  descriptor.version = CORAL_OPERATOR_ABI_VERSION;
  descriptor.descriptor_size = sizeof(descriptor);
  descriptor.opcode = CORAL_OPERATOR_OP_GPTQ_GATED_MLP;
  descriptor.state = CORAL_OPERATOR_STATE_SUBMITTED;
  descriptor.execution_mode = CORAL_OPERATOR_MODE_HYBRID;
  descriptor.tensor_count = 1;
  descriptor.tensors[0].address = kExtmemBase;
  descriptor.tensors[0].size = sizeof(coral_gptq_expert_request);
  descriptor.tensors[0].rank = 1;
  descriptor.tensors[0].dimensions[0] = sizeof(coral_gptq_expert_request);
  descriptor.tensors[0].element_type = CORAL_OPERATOR_ELEMENT_INT8;
  Gem5HybridOperatorResult result = {};
  if (!DispatchGem5HybridOperator(
          &descriptor, image.data(), kExtmemBase, image.size(), &result)) {
    const auto* request =
        reinterpret_cast<const coral_gptq_expert_request*>(image.data());
    std::cerr << "error: GPTQ expert reference failed state=" << request->state
              << " error=" << request->error << "\n";
    return 1;
  }

  const auto* request =
      reinterpret_cast<const coral_gptq_expert_request*>(image.data());
  std::cout << "gptq_expert_reference_operations=" << request->operations
            << "\n";
  std::cout << "gptq_expert_reference_cycles=" << request->modeled_cycles
            << "\n";
  std::cout << "gptq_expert_reference_checksum=0x" << std::hex
            << std::setfill('0') << std::setw(8) << request->output_checksum
            << "\n";
  std::cout << "gptq_expert_reference=PASS\n";
  return 0;
}
