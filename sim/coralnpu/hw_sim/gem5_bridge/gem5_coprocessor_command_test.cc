#include "hw_sim/gem5_bridge/gem5_coprocessor_command.h"

#include <cassert>
#include <cstdint>

namespace {

coral_operator_descriptor ValidDescriptor(uint32_t opcode) {
  coral_operator_descriptor descriptor = {};
  descriptor.magic = CORAL_OPERATOR_ABI_MAGIC;
  descriptor.version = CORAL_OPERATOR_ABI_VERSION;
  descriptor.descriptor_size = sizeof(descriptor);
  descriptor.opcode = opcode;
  descriptor.state = CORAL_OPERATOR_STATE_SUBMITTED;
  descriptor.execution_mode = CORAL_OPERATOR_MODE_HYBRID;
  return descriptor;
}

void TestTwoLevelDecodeAndDependencies() {
  Gem5CoprocessorCommandAdapter adapter;
  const coral_operator_descriptor descriptor =
      ValidDescriptor(CORAL_OPERATOR_OP_MATMUL_INT8);
  uint32_t tag = 0;
  uint32_t error = UINT32_MAX;
  assert(adapter.Submit(Gem5CommandSource::kCustomInstruction, 0x20400300,
                        descriptor, &tag, &error));
  assert(tag != 0);
  assert(error == CORAL_OPERATOR_ERROR_NONE);
  assert(adapter.PendingCount() == 5);

  const Gem5MicroOpcode expected_opcodes[] = {
      Gem5MicroOpcode::kFetchDescriptor,
      Gem5MicroOpcode::kReadOperands,
      Gem5MicroOpcode::kExecuteOperator,
      Gem5MicroOpcode::kWriteback,
      Gem5MicroOpcode::kComplete,
  };
  const Gem5CommandEngine expected_engines[] = {
      Gem5CommandEngine::kFrontend,
      Gem5CommandEngine::kTdma,
      Gem5CommandEngine::kTensor,
      Gem5CommandEngine::kTdma,
      Gem5CommandEngine::kCompletion,
  };
  for (size_t i = 0; i < 5; ++i) {
    Gem5CoprocessorCommand command = {};
    assert(adapter.IssueNext(&command));
    assert(command.source == Gem5CommandSource::kCustomInstruction);
    assert(command.opcode == expected_opcodes[i]);
    assert(command.engine == expected_engines[i]);
    Gem5CoprocessorCommand blocked = {};
    assert(!adapter.IssueNext(&blocked));
    assert(adapter.Complete(command.command_id, true));
  }
  assert(adapter.SubmissionComplete(tag));
  assert(adapter.PendingCount() == 0);
}

void TestBadDescriptorRejected() {
  Gem5CoprocessorCommandAdapter adapter;
  coral_operator_descriptor descriptor =
      ValidDescriptor(CORAL_OPERATOR_OP_ADD_INT8);
  descriptor.magic = 0;
  uint32_t tag = 0;
  uint32_t error = 0;
  assert(!adapter.Submit(Gem5CommandSource::kMmioDoorbell, 0x20400300,
                         descriptor, &tag, &error));
  assert(error == CORAL_OPERATOR_ERROR_BAD_DESCRIPTOR);
}

void TestFailureStopsDependentCommands() {
  Gem5CoprocessorCommandAdapter adapter;
  const coral_operator_descriptor descriptor =
      ValidDescriptor(CORAL_OPERATOR_OP_SOFTMAX);
  uint32_t tag = 0;
  uint32_t error = 0;
  assert(adapter.Submit(Gem5CommandSource::kMmioDoorbell, 0x20400300,
                        descriptor, &tag, &error));
  Gem5CoprocessorCommand command = {};
  assert(adapter.IssueNext(&command));
  assert(adapter.Complete(command.command_id, false));
  assert(adapter.SubmissionFailed(tag));
  assert(!adapter.IssueNext(&command));
}

void TestEngineCreditsAndMultipleSubmissions() {
  Gem5CoprocessorCommandAdapter adapter;
  const coral_operator_descriptor tensor_descriptor =
      ValidDescriptor(CORAL_OPERATOR_OP_MATMUL_INT8);
  const coral_operator_descriptor vector_descriptor =
      ValidDescriptor(CORAL_OPERATOR_OP_ADD_INT8);
  uint32_t tensor_tag = 0;
  uint32_t vector_tag = 0;
  uint32_t error = 0;
  assert(adapter.Submit(Gem5CommandSource::kCustomInstruction, 0x20400300,
                        tensor_descriptor, &tensor_tag, &error));
  assert(adapter.Submit(Gem5CommandSource::kCustomInstruction, 0x20400600,
                        vector_descriptor, &vector_tag, &error));
  assert(tensor_tag != vector_tag);
  assert(adapter.PendingCount() == 10);

  Gem5CoprocessorCommand tensor_fetch = {};
  assert(adapter.IssueNext(&tensor_fetch));
  assert(tensor_fetch.submission_tag == tensor_tag);
  assert(tensor_fetch.engine == Gem5CommandEngine::kFrontend);
  assert(adapter.EngineBusy(Gem5CommandEngine::kFrontend));
  assert(adapter.InFlightCount() == 1);

  // The second submission's fetch is dependency-ready but must wait for the
  // single frontend credit.
  Gem5CoprocessorCommand blocked = {};
  assert(!adapter.IssueNext(&blocked));
  assert(adapter.Complete(tensor_fetch.command_id, true));

  Gem5CoprocessorCommand tensor_read = {};
  assert(adapter.IssueNext(&tensor_read));
  assert(tensor_read.submission_tag == tensor_tag);
  assert(tensor_read.engine == Gem5CommandEngine::kTdma);

  // A different engine can issue concurrently while TDMA remains occupied.
  Gem5CoprocessorCommand vector_fetch = {};
  assert(adapter.IssueNext(&vector_fetch));
  assert(vector_fetch.submission_tag == vector_tag);
  assert(vector_fetch.engine == Gem5CommandEngine::kFrontend);
  assert(adapter.InFlightCount() == 2);
  assert(adapter.Complete(tensor_read.command_id, true));
  assert(adapter.Complete(vector_fetch.command_id, true));
  assert(adapter.InFlightCount() == 0);
}

}  // namespace

int main() {
  TestTwoLevelDecodeAndDependencies();
  TestBadDescriptorRejected();
  TestFailureStopsDependentCommands();
  TestEngineCreditsAndMultipleSubmissions();
  return 0;
}
