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
  kInvalidData,
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
  uint32_t tensor_shape = 0;
  uint32_t tensor_data_type = 0;
  uint32_t scalar_param0 = 0;
  uint32_t quant_qzeros_address = 0;
  uint32_t quant_scales_address = 0;
  uint32_t quant_g_idx_address = 0;
  uint32_t quant_config = 0;
  uint32_t quant_qweight_stride = 0;
  uint32_t quant_qzeros_stride = 0;
  uint32_t quant_scales_stride = 0;
  uint32_t quant_group_range = 0;
  uint32_t tensor_aux_source_address = 0;
  uint32_t tensor_aux_destination_address = 0;
  uint32_t attention_heads = 0;
  uint32_t attention_head_dim_flags = 0;
  uint32_t attention_kv_length = 0;
  uint32_t recurrent_heads = 0;
  uint32_t recurrent_dims = 0;
  uint32_t recurrent_beta_address = 0;
  uint32_t recurrent_a_log_address = 0;
  uint32_t recurrent_dt_bias_address = 0;
  uint32_t csr_epoch = 0;
};

struct Gem5TmmaCompletion {
  xopennpux::Operation operation = xopennpux::Operation::kInvalid;
  uint32_t sequence_id = 0;
  uint32_t csr_epoch = 0;
  uint32_t pc = 0;
  uint32_t instruction = 0;
  uint32_t hart_id = 0;
  uint32_t faulting_address = 0;
  uint32_t destination_address = 0;
  uint32_t destination_bytes = 0;
  uint32_t destination_checksum = 0;
  std::array<uint32_t, 4> destination_words{};
  uint64_t mac_operations = 0;
  uint64_t element_operations = 0;
  uint64_t modeled_cycles = 0;
  Gem5TmmaExecutionError error = Gem5TmmaExecutionError::kNone;
};

class Gem5XOpenNpuFunctionalCoprocessor {
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
  struct QueuedOperation {
    Gem5TmmaDispatchPacket dispatch;
    xopennpux::Operation operation = xopennpux::Operation::kInvalid;
    xopennpux::MmaShape shape;
    xopennpux::MmaDataTypes data_types;
    xopennpux::TensorShape tensor_shape;
    xopennpux::MmaDataTypes tensor_data_types;
    uint32_t scalar_param0 = 0;
    uint32_t quant_qzeros_address = 0;
    uint32_t quant_scales_address = 0;
    uint32_t quant_g_idx_address = 0;
    uint32_t quant_config = 0;
    uint32_t quant_qweight_stride = 0;
    uint32_t quant_qzeros_stride = 0;
    uint32_t quant_scales_stride = 0;
    uint32_t quant_group_range = 0;
    uint32_t tensor_aux_source_address = 0;
    uint32_t tensor_aux_destination_address = 0;
    uint32_t attention_heads = 0;
    uint32_t attention_head_dim_flags = 0;
    uint32_t attention_kv_length = 0;
    uint32_t recurrent_heads = 0;
    uint32_t recurrent_dims = 0;
    uint32_t recurrent_beta_address = 0;
    uint32_t recurrent_a_log_address = 0;
    uint32_t recurrent_dt_bias_address = 0;
    uint32_t csr_epoch = 0;
  };

  std::array<QueuedOperation, kQueueCapacity> queue_{};
  size_t queue_head_ = 0;
  size_t queue_size_ = 0;
  uint32_t mma_shape_ = 0;
  uint32_t mma_data_type_ = 0;
  uint32_t tensor_shape_ = 0;
  uint32_t tensor_data_type_ = 0;
  uint32_t scalar_param0_ = 0;
  uint32_t quant_qzeros_address_ = 0;
  uint32_t quant_scales_address_ = 0;
  uint32_t quant_g_idx_address_ = 0;
  uint32_t quant_config_ = 0;
  uint32_t quant_qweight_stride_ = 0;
  uint32_t quant_qzeros_stride_ = 0;
  uint32_t quant_scales_stride_ = 0;
  uint32_t quant_group_range_ = 0;
  uint32_t tensor_aux_source_address_ = 0;
  uint32_t tensor_aux_destination_address_ = 0;
  uint32_t attention_heads_ = 0;
  uint32_t attention_head_dim_flags_ = 0;
  uint32_t attention_kv_length_ = 0;
  uint32_t recurrent_heads_ = 0;
  uint32_t recurrent_dims_ = 0;
  uint32_t recurrent_beta_address_ = 0;
  uint32_t recurrent_a_log_address_ = 0;
  uint32_t recurrent_dt_bias_address_ = 0;
  uint32_t csr_epoch_ = 0;
};

// Compatibility alias for existing bridge code and external tests. New code
// should use the architecture-level name rather than binding to TMMA.
using Gem5TmmaCoprocessor = Gem5XOpenNpuFunctionalCoprocessor;

#endif  // HW_SIM_GEM5_BRIDGE_GEM5_TMMA_COPROCESSOR_H_
