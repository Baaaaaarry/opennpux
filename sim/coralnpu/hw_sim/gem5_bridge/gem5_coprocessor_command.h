#ifndef HW_SIM_GEM5_BRIDGE_GEM5_COPROCESSOR_COMMAND_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_COPROCESSOR_COMMAND_H_

#include <array>
#include <cstddef>
#include <cstdint>

#include "hw_sim/gem5_bridge/coral_operator.h"

enum class Gem5CommandSource : uint8_t {
  kMmioDoorbell = 0,
  kCustomInstruction = 1,
};

enum class Gem5CommandEngine : uint8_t {
  kFrontend = 0,
  kTdma = 1,
  kTensor = 2,
  kVector = 3,
  kSfu = 4,
  kCompletion = 5,
};

enum class Gem5MicroOpcode : uint8_t {
  kFetchDescriptor = 0,
  kReadOperands = 1,
  kExecuteOperator = 2,
  kWriteback = 3,
  kComplete = 4,
};

enum class Gem5CommandState : uint8_t {
  kPending = 0,
  kIssued = 1,
  kComplete = 2,
  kError = 3,
};

struct Gem5CoprocessorCommand {
  uint32_t command_id;
  uint32_t submission_tag;
  uint32_t descriptor_address;
  uint32_t operator_opcode;
  uint64_t dependency_mask;
  Gem5CommandSource source;
  Gem5CommandEngine engine;
  Gem5MicroOpcode opcode;
  Gem5CommandState state;
};

class Gem5DependencyScoreboard {
 public:
  void Reset();
  bool Ready(uint64_t dependency_mask) const;
  void Complete(uint32_t command_id);
  bool IsComplete(uint32_t command_id) const;

 private:
  uint64_t completed_mask_ = 0;
};

class Gem5CoprocessorCommandAdapter {
 public:
  static constexpr size_t kCommandCapacity = 64;
  static constexpr size_t kSubmissionCapacity = 8;
  static constexpr size_t kEngineCount = 6;

  Gem5CoprocessorCommandAdapter();

  void Reset();
  bool Submit(Gem5CommandSource source, uint32_t descriptor_address,
              const coral_operator_descriptor& descriptor,
              uint32_t* submission_tag, uint32_t* error);
  bool IssueNext(Gem5CoprocessorCommand* command);
  bool Complete(uint32_t command_id, bool success);
  bool SubmissionComplete(uint32_t submission_tag) const;
  bool SubmissionFailed(uint32_t submission_tag) const;
  size_t PendingCount() const;
  size_t InFlightCount() const;
  bool EngineBusy(Gem5CommandEngine engine) const;

 private:
  struct Submission {
    uint32_t tag = 0;
    uint32_t completion_command_id = 0;
    bool valid = false;
    bool failed = false;
  };

  bool ValidateDescriptor(const coral_operator_descriptor& descriptor,
                          uint32_t* error) const;
  bool DecodeOperator(Gem5CommandSource source, uint32_t descriptor_address,
                      const coral_operator_descriptor& descriptor,
                      uint32_t submission_tag, uint32_t* error);
  Gem5CommandEngine OperatorEngine(uint32_t opcode) const;
  Submission* FindSubmission(uint32_t tag);
  const Submission* FindSubmission(uint32_t tag) const;
  static size_t EngineIndex(Gem5CommandEngine engine);

  std::array<Gem5CoprocessorCommand, kCommandCapacity> commands_{};
  std::array<Submission, kSubmissionCapacity> submissions_{};
  std::array<bool, kEngineCount> engine_busy_{};
  Gem5DependencyScoreboard scoreboard_;
  uint32_t next_command_id_ = 0;
  uint32_t next_submission_tag_ = 1;
  size_t command_count_ = 0;
};

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_COPROCESSOR_COMMAND_H_
