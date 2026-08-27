#ifndef HW_SIM_GEM5_BRIDGE_GEM5_TMMA_COPROCESSOR_H_
#define HW_SIM_GEM5_BRIDGE_GEM5_TMMA_COPROCESSOR_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "hw_sim/gem5_bridge/xopennpux_isa.h"

enum class Gem5TmmaSubmitResult {
  kAccepted,
  kBackpressure,
  kIllegalInstruction,
  kInvalidCsrState,
};

enum class Gem5TmmaExecutionError {
  kNone,
  kAddress,
  kUnsupportedDataType,
};

struct Gem5TmmaDispatchPacket {
  uint32_t instruction = 0;
  uint32_t pc = 0;
  uint32_t rs1_value = 0;
  uint32_t rs2_value = 0;
  uint32_t rd_value = 0;
  uint32_t hart_id = 0;
  uint32_t sequence_id = 0;
  uint32_t mma_shape = 0;
  uint32_t mma_data_type = 0;
  uint32_t csr_epoch = 0;
};

struct Gem5TmmaCompletion {
  uint32_t sequence_id = 0;
  uint32_t csr_epoch = 0;
  uint32_t pc = 0;
  uint32_t instruction = 0;
  uint32_t hart_id = 0;
  uint32_t faulting_address = 0;
  uint64_t mac_operations = 0;
  uint64_t modeled_cycles = 0;
  Gem5TmmaExecutionError error = Gem5TmmaExecutionError::kNone;
};

class Gem5TmmaCoprocessor {
 public:
  static constexpr size_t kQueueCapacity = 4;

  void Reset();
  bool WriteCsr(uint16_t address, uint32_t value);
  bool ReadCsr(uint16_t address, uint32_t* value) const;
  uint32_t csr_epoch() const { return csr_epoch_; }

  bool ready() const { return queue_size_ < kQueueCapacity; }
  size_t pending_count() const { return queue_size_; }

  Gem5TmmaSubmitResult Classify(const Gem5TmmaDispatchPacket& packet) const;
  Gem5TmmaSubmitResult Submit(const Gem5TmmaDispatchPacket& packet);
  bool ExecuteNext(std::vector<uint8_t>* memory, uint32_t memory_base,
                   Gem5TmmaCompletion* completion);

 private:
  struct QueuedTmma {
    Gem5TmmaDispatchPacket dispatch;
    xopennpux::MmaShape shape;
    xopennpux::MmaDataTypes data_types;
    uint32_t csr_epoch = 0;
  };

  std::array<QueuedTmma, kQueueCapacity> queue_{};
  size_t queue_head_ = 0;
  size_t queue_size_ = 0;
  uint32_t mma_shape_ = 0;
  uint32_t mma_data_type_ = 0;
  uint32_t csr_epoch_ = 0;
};

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_TMMA_COPROCESSOR_H_
