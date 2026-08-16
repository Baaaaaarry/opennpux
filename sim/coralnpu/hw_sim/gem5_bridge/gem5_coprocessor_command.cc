#include "hw_sim/gem5_bridge/gem5_coprocessor_command.h"

#include <algorithm>

namespace {

uint64_t DivCeil(uint64_t value, uint64_t divisor) {
  return value == 0 ? 0 : 1 + (value - 1) / divisor;
}

}  // namespace

const char* Gem5CommandSourceName(Gem5CommandSource source) {
  switch (source) {
    case Gem5CommandSource::kMmioDoorbell:
      return "mmio-doorbell";
    case Gem5CommandSource::kCustomInstruction:
      return "custom-instruction";
  }
  return "unknown-source";
}

const char* Gem5CommandEngineName(Gem5CommandEngine engine) {
  switch (engine) {
    case Gem5CommandEngine::kFrontend:
      return "frontend";
    case Gem5CommandEngine::kTdma:
      return "tdma";
    case Gem5CommandEngine::kTensor:
      return "tensor";
    case Gem5CommandEngine::kVector:
      return "vector";
    case Gem5CommandEngine::kSfu:
      return "sfu";
    case Gem5CommandEngine::kCompletion:
      return "completion";
  }
  return "unknown-engine";
}

const char* Gem5MicroOpcodeName(Gem5MicroOpcode opcode) {
  switch (opcode) {
    case Gem5MicroOpcode::kFetchDescriptor:
      return "fetch-descriptor";
    case Gem5MicroOpcode::kReadOperands:
      return "read-operands";
    case Gem5MicroOpcode::kExecuteOperator:
      return "execute-operator";
    case Gem5MicroOpcode::kWriteback:
      return "writeback";
    case Gem5MicroOpcode::kComplete:
      return "complete";
  }
  return "unknown-micro-op";
}

void Gem5DependencyScoreboard::Reset() { completed_mask_ = 0; }

bool Gem5DependencyScoreboard::Ready(uint64_t dependency_mask) const {
  return (dependency_mask & ~completed_mask_) == 0;
}

void Gem5DependencyScoreboard::Complete(uint32_t command_id) {
  if (command_id < 64) {
    completed_mask_ |= UINT64_C(1) << command_id;
  }
}

bool Gem5DependencyScoreboard::IsComplete(uint32_t command_id) const {
  return command_id < 64 &&
         (completed_mask_ & (UINT64_C(1) << command_id)) != 0;
}

Gem5CoprocessorCommandAdapter::Gem5CoprocessorCommandAdapter() { Reset(); }

void Gem5CoprocessorCommandAdapter::Reset() {
  commands_.fill({});
  submissions_.fill({});
  engine_busy_.fill(false);
  scoreboard_.Reset();
  next_command_id_ = 0;
  next_submission_tag_ = 1;
  command_count_ = 0;
}

size_t Gem5CoprocessorCommandAdapter::EngineIndex(
    Gem5CommandEngine engine) {
  return static_cast<size_t>(engine);
}

void Gem5CoprocessorCommandAdapter::ConfigureLatencyModel(
    uint64_t operations_per_cycle, uint64_t bytes_per_cycle,
    uint64_t fixed_compute_cycles) {
  operations_per_cycle_ = std::max<uint64_t>(operations_per_cycle, 1);
  bytes_per_cycle_ = std::max<uint64_t>(bytes_per_cycle, 1);
  fixed_compute_cycles_ = fixed_compute_cycles;
}

uint64_t Gem5CoprocessorCommandAdapter::CommandLatency(
    const coral_operator_descriptor& descriptor,
    Gem5MicroOpcode opcode) const {
  switch (opcode) {
    case Gem5MicroOpcode::kReadOperands:
      return std::max<uint64_t>(DivCeil(descriptor.bytes_read,
                                        bytes_per_cycle_), 1);
    case Gem5MicroOpcode::kExecuteOperator:
      return std::max<uint64_t>(fixed_compute_cycles_ +
                                    DivCeil(descriptor.operation_count,
                                            operations_per_cycle_),
                                1);
    case Gem5MicroOpcode::kWriteback:
      return std::max<uint64_t>(DivCeil(descriptor.bytes_written,
                                        bytes_per_cycle_), 1);
    default:
      return 1;
  }
}

bool Gem5CoprocessorCommandAdapter::ValidateDescriptor(
    const coral_operator_descriptor& descriptor, uint32_t* error) const {
  if (error == nullptr) {
    return false;
  }
  *error = CORAL_OPERATOR_ERROR_NONE;
  if (descriptor.magic != CORAL_OPERATOR_ABI_MAGIC ||
      descriptor.version != CORAL_OPERATOR_ABI_VERSION ||
      descriptor.descriptor_size != sizeof(descriptor) ||
      descriptor.state != CORAL_OPERATOR_STATE_SUBMITTED ||
      descriptor.tensor_count > CORAL_OPERATOR_MAX_TENSORS) {
    *error = CORAL_OPERATOR_ERROR_BAD_DESCRIPTOR;
    return false;
  }
  if (OperatorEngine(descriptor.opcode) == Gem5CommandEngine::kFrontend) {
    *error = CORAL_OPERATOR_ERROR_UNSUPPORTED;
    return false;
  }
  return true;
}

Gem5CommandEngine Gem5CoprocessorCommandAdapter::OperatorEngine(
    uint32_t opcode) const {
  switch (opcode) {
    case CORAL_OPERATOR_OP_CONV_2D_INT8:
    case CORAL_OPERATOR_OP_DEPTHWISE_CONV_2D_INT8:
    case CORAL_OPERATOR_OP_MATMUL_INT8:
    case CORAL_OPERATOR_OP_FULLY_CONNECTED_INT8:
    case CORAL_OPERATOR_OP_QWEN_TINY_INFER:
    case CORAL_OPERATOR_OP_GPTQ_MATMUL_INT4:
      return Gem5CommandEngine::kTensor;
    case CORAL_OPERATOR_OP_ADD_INT8:
    case CORAL_OPERATOR_OP_LAYER_NORM:
      return Gem5CommandEngine::kVector;
    case CORAL_OPERATOR_OP_SOFTMAX:
      return Gem5CommandEngine::kSfu;
    default:
      return Gem5CommandEngine::kFrontend;
  }
}

bool Gem5CoprocessorCommandAdapter::DecodeOperator(
    Gem5CommandSource source, uint32_t descriptor_address,
    const coral_operator_descriptor& descriptor, uint32_t submission_tag,
    uint32_t* error) {
  constexpr size_t kCommandsPerSubmission = 5;
  if (command_count_ > kCommandCapacity - kCommandsPerSubmission ||
      next_command_id_ > kCommandCapacity - kCommandsPerSubmission) {
    *error = CORAL_OPERATOR_ERROR_EXECUTION;
    return false;
  }

  const Gem5CommandEngine operator_engine = OperatorEngine(descriptor.opcode);
  const std::array<Gem5CommandEngine, kCommandsPerSubmission> engines = {
      Gem5CommandEngine::kFrontend, Gem5CommandEngine::kTdma,
      operator_engine, Gem5CommandEngine::kTdma,
      Gem5CommandEngine::kCompletion};
  const std::array<Gem5MicroOpcode, kCommandsPerSubmission> opcodes = {
      Gem5MicroOpcode::kFetchDescriptor, Gem5MicroOpcode::kReadOperands,
      Gem5MicroOpcode::kExecuteOperator, Gem5MicroOpcode::kWriteback,
      Gem5MicroOpcode::kComplete};

  uint32_t previous_id = 0;
  for (size_t i = 0; i < kCommandsPerSubmission; ++i) {
    const uint32_t command_id = next_command_id_++;
    Gem5CoprocessorCommand& command = commands_[command_count_++];
    command.command_id = command_id;
    command.submission_tag = submission_tag;
    command.descriptor_address = descriptor_address;
    command.operator_opcode = descriptor.opcode;
    command.dependency_mask =
        i == 0 ? 0 : (UINT64_C(1) << previous_id);
    command.latency_cycles = CommandLatency(descriptor, opcodes[i]);
    command.remaining_cycles = 0;
    command.source = source;
    command.engine = engines[i];
    command.opcode = opcodes[i];
    command.state = Gem5CommandState::kPending;
    command.work_started = false;
    previous_id = command_id;
  }
  Submission* submission = FindSubmission(submission_tag);
  submission->completion_command_id = previous_id;
  return true;
}

bool Gem5CoprocessorCommandAdapter::Submit(
    Gem5CommandSource source, uint32_t descriptor_address,
    const coral_operator_descriptor& descriptor, uint32_t* submission_tag,
    uint32_t* error) {
  // This first implementation executes one command graph at a time. Reclaim
  // terminal entries before accepting the next operator so long-running
  // firmware is not limited by the fixed verification queue depth.
  if (command_count_ != 0 && PendingCount() == 0) {
    commands_.fill({});
    submissions_.fill({});
    scoreboard_.Reset();
    next_command_id_ = 0;
    command_count_ = 0;
  }
  if (submission_tag == nullptr || error == nullptr ||
      !ValidateDescriptor(descriptor, error)) {
    return false;
  }
  auto slot = std::find_if(submissions_.begin(), submissions_.end(),
                           [](const Submission& item) { return !item.valid; });
  if (slot == submissions_.end()) {
    *error = CORAL_OPERATOR_ERROR_EXECUTION;
    return false;
  }
  slot->valid = true;
  slot->failed = false;
  slot->tag = next_submission_tag_++;
  if (!DecodeOperator(source, descriptor_address, descriptor, slot->tag,
                      error)) {
    *slot = {};
    return false;
  }
  *submission_tag = slot->tag;
  return true;
}

bool Gem5CoprocessorCommandAdapter::IssueNext(
    Gem5CoprocessorCommand* command) {
  if (command == nullptr) {
    return false;
  }
  for (size_t i = 0; i < command_count_; ++i) {
    Gem5CoprocessorCommand& candidate = commands_[i];
    if (candidate.state == Gem5CommandState::kPending &&
        scoreboard_.Ready(candidate.dependency_mask) &&
        !EngineBusy(candidate.engine)) {
      candidate.state = Gem5CommandState::kIssued;
      candidate.remaining_cycles = candidate.latency_cycles;
      engine_busy_[EngineIndex(candidate.engine)] = true;
      *command = candidate;
      return true;
    }
  }
  return false;
}

void Gem5CoprocessorCommandAdapter::AdvanceCycle() {
  Gem5CoprocessorCommand issued = {};
  while (IssueNext(&issued)) {
  }

  for (size_t i = 0; i < command_count_; ++i) {
    Gem5CoprocessorCommand& command = commands_[i];
    if (command.state != Gem5CommandState::kIssued) {
      continue;
    }
    if (command.remaining_cycles > 0) {
      --command.remaining_cycles;
    }
    if (command.remaining_cycles == 0) {
      command.state = Gem5CommandState::kReadyToComplete;
    }
  }
}

bool Gem5CoprocessorCommandAdapter::TakeReadyToComplete(
    Gem5CoprocessorCommand* command) const {
  if (command == nullptr) {
    return false;
  }
  for (size_t i = 0; i < command_count_; ++i) {
    if (commands_[i].state == Gem5CommandState::kReadyToComplete) {
      *command = commands_[i];
      return true;
    }
  }
  return false;
}

bool Gem5CoprocessorCommandAdapter::Reschedule(
    uint32_t command_id, uint64_t latency_cycles) {
  for (size_t i = 0; i < command_count_; ++i) {
    Gem5CoprocessorCommand& command = commands_[i];
    if (command.command_id != command_id ||
        command.state != Gem5CommandState::kReadyToComplete) {
      continue;
    }
    command.work_started = true;
    command.latency_cycles = std::max<uint64_t>(latency_cycles, 1);
    command.remaining_cycles = command.latency_cycles;
    command.state = Gem5CommandState::kIssued;
    return true;
  }
  return false;
}

bool Gem5CoprocessorCommandAdapter::SetPendingLatency(
    uint32_t submission_tag, Gem5MicroOpcode opcode,
    uint64_t latency_cycles) {
  for (size_t i = 0; i < command_count_; ++i) {
    Gem5CoprocessorCommand& command = commands_[i];
    if (command.submission_tag == submission_tag &&
        command.opcode == opcode &&
        command.state == Gem5CommandState::kPending) {
      command.latency_cycles = std::max<uint64_t>(latency_cycles, 1);
      return true;
    }
  }
  return false;
}

bool Gem5CoprocessorCommandAdapter::FailSubmission(uint32_t submission_tag) {
  Submission* submission = FindSubmission(submission_tag);
  if (submission == nullptr) {
    return false;
  }
  submission->failed = true;
  for (size_t i = 0; i < command_count_; ++i) {
    Gem5CoprocessorCommand& command = commands_[i];
    if (command.submission_tag != submission_tag ||
        command.state == Gem5CommandState::kComplete ||
        command.state == Gem5CommandState::kError) {
      continue;
    }
    if (command.state == Gem5CommandState::kIssued ||
        command.state == Gem5CommandState::kReadyToComplete) {
      engine_busy_[EngineIndex(command.engine)] = false;
    }
    command.state = Gem5CommandState::kError;
  }
  return true;
}

bool Gem5CoprocessorCommandAdapter::Complete(uint32_t command_id,
                                             bool success) {
  for (size_t i = 0; i < command_count_; ++i) {
    Gem5CoprocessorCommand& command = commands_[i];
    if (command.command_id != command_id ||
        (command.state != Gem5CommandState::kReadyToComplete &&
         command.state != Gem5CommandState::kIssued)) {
      continue;
    }
    command.state = success ? Gem5CommandState::kComplete :
                              Gem5CommandState::kError;
    engine_busy_[EngineIndex(command.engine)] = false;
    Submission* submission = FindSubmission(command.submission_tag);
    if (!success && submission != nullptr) {
      submission->failed = true;
      for (size_t j = 0; j < command_count_; ++j) {
        if (commands_[j].submission_tag == command.submission_tag &&
            commands_[j].state == Gem5CommandState::kPending) {
          commands_[j].state = Gem5CommandState::kError;
        }
      }
    }
    if (success) {
      scoreboard_.Complete(command_id);
    }
    return true;
  }
  return false;
}

bool Gem5CoprocessorCommandAdapter::SubmissionComplete(uint32_t tag) const {
  const Submission* submission = FindSubmission(tag);
  return submission != nullptr && !submission->failed &&
         scoreboard_.IsComplete(submission->completion_command_id);
}

bool Gem5CoprocessorCommandAdapter::SubmissionFailed(uint32_t tag) const {
  const Submission* submission = FindSubmission(tag);
  return submission != nullptr && submission->failed;
}

size_t Gem5CoprocessorCommandAdapter::PendingCount() const {
  size_t pending = 0;
  for (size_t i = 0; i < command_count_; ++i) {
    if (commands_[i].state == Gem5CommandState::kPending ||
        commands_[i].state == Gem5CommandState::kIssued ||
        commands_[i].state == Gem5CommandState::kReadyToComplete) {
      ++pending;
    }
  }
  return pending;
}

size_t Gem5CoprocessorCommandAdapter::InFlightCount() const {
  size_t in_flight = 0;
  for (bool busy : engine_busy_) {
    in_flight += busy ? 1 : 0;
  }
  return in_flight;
}

bool Gem5CoprocessorCommandAdapter::EngineBusy(
    Gem5CommandEngine engine) const {
  const size_t index = EngineIndex(engine);
  return index < engine_busy_.size() && engine_busy_[index];
}

Gem5CoprocessorCommandAdapter::Submission*
Gem5CoprocessorCommandAdapter::FindSubmission(uint32_t tag) {
  for (Submission& submission : submissions_) {
    if (submission.valid && submission.tag == tag) {
      return &submission;
    }
  }
  return nullptr;
}

const Gem5CoprocessorCommandAdapter::Submission*
Gem5CoprocessorCommandAdapter::FindSubmission(uint32_t tag) const {
  for (const Submission& submission : submissions_) {
    if (submission.valid && submission.tag == tag) {
      return &submission;
    }
  }
  return nullptr;
}
