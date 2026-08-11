#include "hw_sim/gem5_bridge/gem5_coprocessor_command.h"

#include <algorithm>

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
  scoreboard_.Reset();
  next_command_id_ = 0;
  next_submission_tag_ = 1;
  command_count_ = 0;
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
    command.source = source;
    command.engine = engines[i];
    command.opcode = opcodes[i];
    command.state = Gem5CommandState::kPending;
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
        scoreboard_.Ready(candidate.dependency_mask)) {
      candidate.state = Gem5CommandState::kIssued;
      *command = candidate;
      return true;
    }
  }
  return false;
}

bool Gem5CoprocessorCommandAdapter::Complete(uint32_t command_id,
                                             bool success) {
  for (size_t i = 0; i < command_count_; ++i) {
    Gem5CoprocessorCommand& command = commands_[i];
    if (command.command_id != command_id ||
        command.state != Gem5CommandState::kIssued) {
      continue;
    }
    command.state = success ? Gem5CommandState::kComplete :
                              Gem5CommandState::kError;
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
        commands_[i].state == Gem5CommandState::kIssued) {
      ++pending;
    }
  }
  return pending;
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
