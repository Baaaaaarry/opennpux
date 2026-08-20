#include "hw_sim/gem5_bridge/gem5_coprocessor_command.h"

#include <cassert>
#include <cstdint>
#include <string>

#include "hw_sim/gem5_bridge/npu_submission.h"

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

void TestSourceAndNameObservability() {
  assert(Gem5CommandSourceName(Gem5CommandSource::kMmioDoorbell) ==
         std::string("mmio-doorbell"));
  assert(Gem5CommandSourceName(Gem5CommandSource::kCustomInstruction) ==
         std::string("custom-instruction"));
  assert(Gem5CommandEngineName(Gem5CommandEngine::kTensor) ==
         std::string("tensor"));
  assert(Gem5MicroOpcodeName(Gem5MicroOpcode::kExecuteOperator) ==
         std::string("execute-operator"));

  const coral_operator_descriptor descriptor =
      ValidDescriptor(CORAL_OPERATOR_OP_ADD_INT8);
  const Gem5CommandSource sources[] = {
      Gem5CommandSource::kMmioDoorbell,
      Gem5CommandSource::kCustomInstruction,
  };
  for (Gem5CommandSource source : sources) {
    Gem5CoprocessorCommandAdapter adapter;
    uint32_t tag = 0;
    uint32_t error = 0;
    assert(adapter.Submit(source, 0x20400300, descriptor, &tag, &error));
    Gem5CoprocessorCommand command = {};
    assert(adapter.IssueNext(&command));
    assert(command.submission_tag == tag);
    assert(command.source == source);
    assert(command.opcode == Gem5MicroOpcode::kFetchDescriptor);
  }
}

void TestGptqOpcodeUsesTensorEngine(uint32_t opcode) {
  Gem5CoprocessorCommandAdapter adapter;
  const coral_operator_descriptor descriptor =
      ValidDescriptor(opcode);
  uint32_t tag = 0;
  uint32_t error = UINT32_MAX;
  assert(adapter.Submit(Gem5CommandSource::kCustomInstruction, 0x20400300,
                        descriptor, &tag, &error));

  Gem5CoprocessorCommand command = {};
  for (size_t index = 0; index < 3; ++index) {
    assert(adapter.IssueNext(&command));
    assert(adapter.Complete(command.command_id, true));
  }
  assert(command.opcode == Gem5MicroOpcode::kExecuteOperator);
  assert(command.engine == Gem5CommandEngine::kTensor);
}

void TestGenericOpcodeSecondLevelDecode(uint32_t opcode,
                                        Gem5CommandEngine expected_engine) {
  Gem5CoprocessorCommandAdapter adapter;
  coral_operator_descriptor descriptor =
      ValidDescriptor(CORAL_OPERATOR_OP_GENERIC_COMMAND);
  descriptor.reserved[0] = opcode;
  uint32_t tag = 0;
  uint32_t error = UINT32_MAX;
  assert(adapter.Submit(Gem5CommandSource::kCustomInstruction, 0x20400300,
                        descriptor, &tag, &error));
  Gem5CoprocessorCommand command = {};
  for (size_t index = 0; index < 3; ++index) {
    assert(adapter.IssueNext(&command));
    assert(adapter.Complete(command.command_id, true));
  }
  assert(command.operator_opcode == CORAL_OPERATOR_OP_GENERIC_COMMAND);
  assert(command.generic_opcode == opcode);
  assert(command.opcode == Gem5MicroOpcode::kExecuteOperator);
  assert(command.engine == expected_engine);
}

void TestUnsupportedGenericOpcodeRejected() {
  Gem5CoprocessorCommandAdapter adapter;
  coral_operator_descriptor descriptor =
      ValidDescriptor(CORAL_OPERATOR_OP_GENERIC_COMMAND);
  descriptor.reserved[0] = 0;
  uint32_t tag = 0;
  uint32_t error = 0;
  assert(!adapter.Submit(Gem5CommandSource::kCustomInstruction, 0x20400300,
                         descriptor, &tag, &error));
  assert(error == CORAL_OPERATOR_ERROR_UNSUPPORTED);
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

void TestCycleDrivenExecution() {
  Gem5CoprocessorCommandAdapter adapter;
  adapter.ConfigureLatencyModel(4, 8, 2);
  coral_operator_descriptor descriptor =
      ValidDescriptor(CORAL_OPERATOR_OP_MATMUL_INT8);
  descriptor.operation_count = 16;
  descriptor.bytes_read = 16;
  descriptor.bytes_written = 8;
  uint32_t tag = 0;
  uint32_t error = 0;
  assert(adapter.Submit(Gem5CommandSource::kCustomInstruction, 0x20400300,
                        descriptor, &tag, &error));
  assert(!adapter.SubmissionComplete(tag));

  const uint64_t expected_latencies[] = {1, 2, 6, 1, 1};
  for (uint64_t expected : expected_latencies) {
    adapter.AdvanceCycle();
    Gem5CoprocessorCommand command = {};
    for (uint64_t cycle = 0;
         !adapter.TakeReadyToComplete(&command); ++cycle) {
      assert(cycle < expected);
      adapter.AdvanceCycle();
    }
    assert(command.latency_cycles == expected);
    assert(adapter.Complete(command.command_id, true));
  }
  assert(adapter.SubmissionComplete(tag));
}

void TestDynamicLatencyUpdate() {
  Gem5CoprocessorCommandAdapter adapter;
  adapter.ConfigureLatencyModel(1, 8, 0);
  coral_operator_descriptor descriptor =
      ValidDescriptor(CORAL_OPERATOR_OP_ADD_INT8);
  uint32_t tag = 0;
  uint32_t error = 0;
  assert(adapter.Submit(Gem5CommandSource::kCustomInstruction, 0x20400300,
                        descriptor, &tag, &error));

  const Gem5MicroOpcode expected_opcodes[] = {
      Gem5MicroOpcode::kFetchDescriptor,
      Gem5MicroOpcode::kReadOperands,
      Gem5MicroOpcode::kExecuteOperator,
  };
  for (Gem5MicroOpcode expected_opcode : expected_opcodes) {
    adapter.AdvanceCycle();
    Gem5CoprocessorCommand command = {};
    assert(adapter.TakeReadyToComplete(&command));
    assert(command.opcode == expected_opcode);
    assert(adapter.Complete(command.command_id, true));
  }

  assert(adapter.SetPendingLatency(tag, Gem5MicroOpcode::kWriteback, 4));
  adapter.AdvanceCycle();
  Gem5CoprocessorCommand writeback = {};
  for (uint64_t cycle = 0;
       !adapter.TakeReadyToComplete(&writeback); ++cycle) {
    assert(cycle < 4);
    adapter.AdvanceCycle();
  }
  assert(writeback.opcode == Gem5MicroOpcode::kWriteback);
  assert(writeback.latency_cycles == 4);
  assert(adapter.Complete(writeback.command_id, true));
}

}  // namespace

int main() {
  TestTwoLevelDecodeAndDependencies();
  TestSourceAndNameObservability();
  TestGptqOpcodeUsesTensorEngine(CORAL_OPERATOR_OP_GPTQ_MATMUL_INT4);
  TestGptqOpcodeUsesTensorEngine(CORAL_OPERATOR_OP_GPTQ_GATED_MLP);
  TestGenericOpcodeSecondLevelDecode(OPENNPUX_NPU_OP_MATMUL,
                                     Gem5CommandEngine::kTensor);
  TestGenericOpcodeSecondLevelDecode(OPENNPUX_NPU_OP_EMBED,
                                     Gem5CommandEngine::kVector);
  TestGenericOpcodeSecondLevelDecode(OPENNPUX_NPU_OP_ADD,
                                     Gem5CommandEngine::kVector);
  TestGenericOpcodeSecondLevelDecode(OPENNPUX_NPU_OP_MUL,
                                     Gem5CommandEngine::kVector);
  TestGenericOpcodeSecondLevelDecode(OPENNPUX_NPU_OP_NORMALIZE,
                                     Gem5CommandEngine::kVector);
  TestGenericOpcodeSecondLevelDecode(OPENNPUX_NPU_OP_ROPE,
                                     Gem5CommandEngine::kVector);
  TestGenericOpcodeSecondLevelDecode(OPENNPUX_NPU_OP_ACTIVATION,
                                     Gem5CommandEngine::kSfu);
  TestGenericOpcodeSecondLevelDecode(OPENNPUX_NPU_OP_SOFTMAX,
                                     Gem5CommandEngine::kSfu);
  TestGenericOpcodeSecondLevelDecode(OPENNPUX_NPU_OP_TOPK,
                                     Gem5CommandEngine::kSfu);
  TestUnsupportedGenericOpcodeRejected();
  TestBadDescriptorRejected();
  TestFailureStopsDependentCommands();
  TestEngineCreditsAndMultipleSubmissions();
  TestCycleDrivenExecution();
  TestDynamicLatencyUpdate();
  return 0;
}
