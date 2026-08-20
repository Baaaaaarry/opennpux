#include "hw_sim/gem5_bridge/coralnpu_gem5_abi.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

#include "hw_sim/gem5_bridge/gem5_core_mini_axi_wrapper.h"
#include "hw_sim/gem5_bridge/coral_mobilenet.h"
#include "hw_sim/gem5_bridge/coral_operator.h"
#include "hw_sim/gem5_bridge/gem5_coprocessor_command.h"
#include "hw_sim/gem5_bridge/gem5_custom_mac.h"
#include "hw_sim/gem5_bridge/gem5_dma_request_builder.h"
#include "hw_sim/gem5_bridge/gem5_sim_host_pager.h"
#include "hw_sim/gem5_bridge/gem5_sim_host_numerical.h"
#include "hw_sim/gem5_bridge/npu_weight_queue.h"
#ifdef CORAL_GEM5_RVV_HIGHMEM
#include "hw_sim/gem5_bridge/gem5_hybrid_operator.h"
#endif

namespace {

static_assert(sizeof(coral_operator_descriptor) <= 4096 - 0x100,
              "Coral operator descriptor must fit the synchronized page");

constexpr uint8_t kAxiSlvErr = 2;
constexpr uint32_t kFirmwareProgressAddr =
    OPENNPUX_CORAL_MOBILENET_PROGRESS_ADDR;
constexpr uint32_t kExtmemBase = 0x20000000;
constexpr uint32_t kExtmemSize = 8 * 1024 * 1024;
constexpr uint32_t kExternalPollThreshold = 32;
constexpr uint32_t kExternalPendingState = 1;
#ifdef CORAL_GEM5_RVV_HIGHMEM
constexpr uint32_t kShellCsrBase = 0x00030000;
constexpr uint32_t kCsrSize = 0x00001000;
constexpr uint32_t kRtlCsrBase = 0x00200000;
#endif

uint32_t TranslateSlaveAddress(uint32_t addr) {
#ifdef CORAL_GEM5_RVV_HIGHMEM
  if (addr >= kShellCsrBase && addr < kShellCsrBase + kCsrSize) {
    return kRtlCsrBase + (addr - kShellCsrBase);
  }
#endif
  return addr;
}

bool IsCustomWordAccess(const AxiAddr& addr) {
  return addr.addr_bits_len == 0 && addr.addr_bits_size == 2 &&
         addr.addr_bits_burst == 1 && (addr.addr_bits_addr & 3) == 0;
}

bool IsHybridWordAccess(const AxiAddr& addr) {
  return addr.addr_bits_len == 0 && addr.addr_bits_size == 2 &&
         addr.addr_bits_burst == 1 && (addr.addr_bits_addr & 3) == 0 &&
         addr.addr_bits_addr >= CORAL_OPERATOR_MMIO_BASE &&
         addr.addr_bits_addr < CORAL_OPERATOR_MMIO_BASE + 0x100;
}

size_t AccessWidthBucket(uint32_t size) {
  switch (size) {
    case 1:
      return 0;
    case 2:
      return 1;
    case 4:
      return 2;
    case 8:
      return 3;
    case 16:
      return 4;
    default:
      return 5;
  }
}

uint64_t ReadEnvU64(const char* name, uint64_t default_value) {
  const char* text = std::getenv(name);
  if (text == nullptr || text[0] == '\0') {
    return default_value;
  }
  char* end = nullptr;
  const unsigned long long value = std::strtoull(text, &end, 0);
  return end != text && end != nullptr && *end == '\0' && value != 0 ?
      static_cast<uint64_t>(value) : default_value;
}

uint64_t DivCeil(uint64_t value, uint64_t divisor) {
  return value == 0 ? 0 : 1 + (value - 1) / divisor;
}

bool IsHexMaskText(const char* text) {
  return text != nullptr && text[0] == '0' &&
         (text[1] == 'x' || text[1] == 'X');
}

uint32_t OperatorMaskBit(uint32_t opcode) {
  return opcode < 32 ? (UINT32_C(1) << opcode) : 0;
}

const char* OperatorName(uint32_t opcode) {
  switch (opcode) {
    case CORAL_OPERATOR_OP_PARTIAL_MOBILENET:
      return "partial_mobilenet";
    case CORAL_OPERATOR_OP_CONV_2D_INT8:
      return "conv2d_int8";
    case CORAL_OPERATOR_OP_DEPTHWISE_CONV_2D_INT8:
      return "depthwise_conv2d_int8";
    case CORAL_OPERATOR_OP_MATMUL_INT8:
      return "matmul_int8";
    case CORAL_OPERATOR_OP_FULLY_CONNECTED_INT8:
      return "fully_connected_int8";
    case CORAL_OPERATOR_OP_ADD_INT8:
      return "add_int8";
    case CORAL_OPERATOR_OP_SOFTMAX:
      return "softmax";
    case CORAL_OPERATOR_OP_LAYER_NORM:
      return "layer_norm";
    case CORAL_OPERATOR_OP_QWEN_TINY_INFER:
      return "qwen_tiny_infer";
    case CORAL_OPERATOR_OP_GPTQ_MATMUL_INT4:
      return "gptq_matmul_int4";
    case CORAL_OPERATOR_OP_GPTQ_GATED_MLP:
      return "gptq_gated_mlp";
    case CORAL_OPERATOR_OP_GPTQ_PAGED_MATMUL:
      return "gptq_paged_matmul";
    case CORAL_OPERATOR_OP_GENERIC_COMMAND:
      return "generic_command";
    default:
      return "unknown";
  }
}

bool TokenEquals(const char* begin, const char* end, const char* name) {
  while (begin < end && std::isspace(static_cast<unsigned char>(*begin))) {
    ++begin;
  }
  while (end > begin && std::isspace(static_cast<unsigned char>(end[-1]))) {
    --end;
  }
  const size_t length = static_cast<size_t>(end - begin);
  return std::strlen(name) == length && std::strncmp(begin, name, length) == 0;
}

uint32_t ParseSampledRtlMask() {
  const char* text = std::getenv("CORAL_SAMPLED_RTL_OPS");
  if (text == nullptr || text[0] == '\0' || std::strcmp(text, "none") == 0) {
    return 0;
  }
  if (std::strcmp(text, "all") == 0) {
    return OperatorMaskBit(CORAL_OPERATOR_OP_CONV_2D_INT8) |
           OperatorMaskBit(CORAL_OPERATOR_OP_DEPTHWISE_CONV_2D_INT8) |
           OperatorMaskBit(CORAL_OPERATOR_OP_MATMUL_INT8) |
           OperatorMaskBit(CORAL_OPERATOR_OP_FULLY_CONNECTED_INT8) |
           OperatorMaskBit(CORAL_OPERATOR_OP_ADD_INT8) |
           OperatorMaskBit(CORAL_OPERATOR_OP_SOFTMAX) |
           OperatorMaskBit(CORAL_OPERATOR_OP_LAYER_NORM) |
           OperatorMaskBit(CORAL_OPERATOR_OP_QWEN_TINY_INFER);
  }
  if (IsHexMaskText(text)) {
    return static_cast<uint32_t>(std::strtoul(text, nullptr, 0));
  }

  uint32_t mask = 0;
  const char* token_begin = text;
  for (const char* p = text; ; ++p) {
    if (*p != ',' && *p != '\0') {
      continue;
    }
    if (TokenEquals(token_begin, p, "conv")) {
      mask |= OperatorMaskBit(CORAL_OPERATOR_OP_CONV_2D_INT8);
    } else if (TokenEquals(token_begin, p, "depthwise")) {
      mask |= OperatorMaskBit(CORAL_OPERATOR_OP_DEPTHWISE_CONV_2D_INT8);
    } else if (TokenEquals(token_begin, p, "matmul")) {
      mask |= OperatorMaskBit(CORAL_OPERATOR_OP_MATMUL_INT8);
    } else if (TokenEquals(token_begin, p, "fc") ||
               TokenEquals(token_begin, p, "fully_connected")) {
      mask |= OperatorMaskBit(CORAL_OPERATOR_OP_FULLY_CONNECTED_INT8);
    } else if (TokenEquals(token_begin, p, "add")) {
      mask |= OperatorMaskBit(CORAL_OPERATOR_OP_ADD_INT8);
    } else if (TokenEquals(token_begin, p, "softmax")) {
      mask |= OperatorMaskBit(CORAL_OPERATOR_OP_SOFTMAX);
    } else if (TokenEquals(token_begin, p, "layernorm") ||
               TokenEquals(token_begin, p, "layer_norm")) {
      mask |= OperatorMaskBit(CORAL_OPERATOR_OP_LAYER_NORM);
    } else if (TokenEquals(token_begin, p, "qwen") ||
               TokenEquals(token_begin, p, "qwen_tiny")) {
      mask |= OperatorMaskBit(CORAL_OPERATOR_OP_QWEN_TINY_INFER);
    }
    if (*p == '\0') {
      break;
    }
    token_begin = p + 1;
  }
  return mask;
}

}  // namespace

struct HybridOperatorStats {
  uint64_t count = 0;
  uint64_t host_ns = 0;
  uint64_t operations = 0;
  uint64_t modeled_cycles = 0;
  uint64_t bytes = 0;
};

struct OperatorPhaseStats {
  uint64_t count = 0;
  uint64_t rtl_cycles = 0;
  uint64_t wall_ms = 0;
  uint64_t extmem_accesses = 0;
  uint64_t extmem_bytes = 0;
  bool active = false;
  uint64_t begin_cycles = 0;
  uint64_t begin_accesses = 0;
  uint64_t begin_bytes = 0;
  std::chrono::steady_clock::time_point begin_wall_time;
};

// Dumps a transaction the bridge is about to answer with SLVERR. Used to
// identify exactly which AXI shape (burst type, beat size, strobe pattern)
// the request builder rejected — e.g. vector-unit stores that differ from
// the scalar traffic the bridge was written for.
void DumpRejectedAxi(const char* direction, const AxiAddr& addr,
                     const std::vector<AxiWData>* data, const char* reason) {
  std::fprintf(stderr,
               "Coral AXI reject %s addr=0x%08x id=%u len=%u size=%u "
               "burst=%u reason=%s\n",
               direction, addr.addr_bits_addr, addr.addr_bits_id,
               addr.addr_bits_len, addr.addr_bits_size, addr.addr_bits_burst,
               reason != nullptr ? reason : "unknown");
  if (data != nullptr) {
    for (size_t i = 0; i < data->size(); ++i) {
      std::fprintf(stderr, "  beat[%zu] strb=0x%04x last=%u\n", i,
                   (*data)[i].write_data_bits_strb,
                   (*data)[i].write_data_bits_last);
    }
  }
  std::fflush(stderr);
}

struct AsyncOperatorSubmission {
  bool valid = false;
  uint32_t tag = 0;
  coral_operator_descriptor* descriptor = nullptr;
#ifdef CORAL_GEM5_RVV_HIGHMEM
  Gem5HybridOperatorResult result = {};
#endif
  bool kernel_done = false;
  bool kernel_success = false;
  uint32_t final_error = CORAL_OPERATOR_ERROR_NONE;
};

struct coral_gem5_handle {
  VerilatedContext context;
  Gem5CoreMiniAxiWrapper wrapper;
  Gem5CustomMac custom_mac;
  Gem5CoprocessorCommandAdapter command_adapter;
  Gem5SimHostPager sim_host_pager;
  Gem5SimHostNumerical sim_host_numerical;
  std::array<AsyncOperatorSubmission,
             Gem5CoprocessorCommandAdapter::kSubmissionCapacity>
      async_submissions;
  uint32_t firmware_progress;
  uint32_t operator_mode;
  uint32_t sampled_rtl_mask;
  uint32_t hybrid_status;
  bool local_extmem_enabled;
  std::vector<uint8_t> local_extmem;
  uint64_t local_extmem_reads;
  uint64_t local_extmem_writes;
  uint64_t local_extmem_bytes;
  std::array<uint64_t, 6> local_extmem_widths;
  uint32_t external_poll_addr;
  uint32_t external_poll_count;
  bool external_wait;
  uint64_t rtl_cycles;
  uint64_t hybrid_ops_per_cycle;
  uint64_t hybrid_bytes_per_cycle;
  uint64_t hybrid_fixed_cycles;
  uint64_t progress_cycles;
  uint64_t progress_accesses;
  uint64_t progress_bytes;
  std::chrono::steady_clock::time_point progress_wall_time;
  std::array<HybridOperatorStats, 32> hybrid_operator_stats;
  std::array<OperatorPhaseStats, 32> operator_phase_stats;
  coral_gem5_dma_request pending_dma;
  bool dma_pending;
  uint32_t dma_beat_size;
  uint32_t dma_beat_count;
  // Watchdog: rtl_cycles of the last observed AXI master activity. When the
  // core makes no master transaction for watchdog_cycles, the bridge dumps
  // the channel handshake levels to pinpoint the stall.
  uint64_t last_activity_cycles;
  uint64_t watchdog_cycles;

  void ObserveExtmemRead(uint32_t addr, uint32_t size,
                         const uint8_t* data) {
    uint32_t value = 0;
    if (size != sizeof(value) || data == nullptr) {
      external_poll_addr = 0;
      external_poll_count = 0;
      return;
    }
    std::memcpy(&value, data, sizeof(value));
    const uint32_t state_offset = static_cast<uint32_t>(
        offsetof(opennpux_npu_page_fault, state));
    if (value != kExternalPendingState ||
        addr < kExtmemBase + state_offset) {
      external_poll_addr = 0;
      external_poll_count = 0;
      return;
    }
    const size_t fault_offset = addr - kExtmemBase - state_offset;
    if (fault_offset >
        local_extmem.size() - sizeof(opennpux_npu_page_fault)) {
      external_poll_addr = 0;
      external_poll_count = 0;
      return;
    }
    opennpux_npu_page_fault fault = {};
    std::memcpy(&fault, local_extmem.data() + fault_offset, sizeof(fault));
    if (fault.magic != OPENNPUX_NPU_PAGE_FAULT_MAGIC ||
        fault.version != OPENNPUX_NPU_PAGE_FAULT_VERSION ||
        fault.struct_size != sizeof(fault)) {
      external_poll_addr = 0;
      external_poll_count = 0;
      return;
    }
    if (external_poll_addr == addr) {
      ++external_poll_count;
    } else {
      external_poll_addr = addr;
      external_poll_count = 1;
    }
    if (external_poll_count >= kExternalPollThreshold) {
      external_wait = true;
    }
  }

  void NotifyExtmemWrite() {
    external_poll_addr = 0;
    external_poll_count = 0;
    external_wait = false;
  }

  AsyncOperatorSubmission* FindAsyncSubmission(uint32_t tag) {
    for (AsyncOperatorSubmission& submission : async_submissions) {
      if (submission.valid && submission.tag == tag) {
        return &submission;
      }
    }
    return nullptr;
  }

  bool AddAsyncSubmission(uint32_t tag,
                          coral_operator_descriptor* descriptor) {
    for (AsyncOperatorSubmission& submission : async_submissions) {
      if (!submission.valid) {
        submission.valid = true;
        submission.tag = tag;
        submission.descriptor = descriptor;
#ifdef CORAL_GEM5_RVV_HIGHMEM
        submission.result = {};
#endif
        submission.kernel_done = false;
        submission.kernel_success = false;
        submission.final_error = CORAL_OPERATOR_ERROR_NONE;
        return true;
      }
    }
    return false;
  }

  uint64_t OperatorComputeCycles(
      const coral_operator_descriptor& descriptor) const {
    return std::max<uint64_t>(
        hybrid_fixed_cycles +
            DivCeil(descriptor.operation_count, hybrid_ops_per_cycle),
        1);
  }

  uint64_t OperatorWritebackCycles(
      const coral_operator_descriptor& descriptor) const {
    return std::max<uint64_t>(
        DivCeil(descriptor.bytes_written, hybrid_bytes_per_cycle), 1);
  }

  void PublishOperatorCompletion(AsyncOperatorSubmission* submission,
                                 bool success) {
    if (submission == nullptr || submission->descriptor == nullptr) {
      hybrid_status = CORAL_OPERATOR_STATE_ERROR;
      return;
    }
    coral_operator_descriptor* descriptor = submission->descriptor;
    descriptor->state = success ? CORAL_OPERATOR_STATE_COMPLETE :
                                  CORAL_OPERATOR_STATE_ERROR;
    descriptor->error = success ? CORAL_OPERATOR_ERROR_NONE :
                                  submission->final_error;
    hybrid_status = descriptor->state;
    if (success) {
      const uint64_t traffic_bytes =
          descriptor->bytes_read + descriptor->bytes_written;
      descriptor->modeled_cycles =
          hybrid_fixed_cycles +
          DivCeil(descriptor->operation_count, hybrid_ops_per_cycle) +
          DivCeil(traffic_bytes, hybrid_bytes_per_cycle);
      if (static_cast<size_t>(descriptor->opcode) <
          hybrid_operator_stats.size()) {
        HybridOperatorStats& stats =
            hybrid_operator_stats[descriptor->opcode];
        ++stats.count;
        stats.host_ns += descriptor->host_elapsed_ns;
        stats.operations += descriptor->operation_count;
        stats.modeled_cycles += descriptor->modeled_cycles;
        stats.bytes += traffic_bytes;
      }
      std::fprintf(stderr,
                   "Coral hybrid operator complete opcode=%u name=%s "
                   "count=%llu host_ns=%llu operations=%llu "
                   "modeled_cycles=%llu bytes=%llu tag=%u\n",
                   descriptor->opcode, OperatorName(descriptor->opcode),
                   static_cast<size_t>(descriptor->opcode) <
                           hybrid_operator_stats.size()
                       ? static_cast<unsigned long long>(
                             hybrid_operator_stats[descriptor->opcode].count)
                       : 0ULL,
                   static_cast<unsigned long long>(
                       descriptor->host_elapsed_ns),
                   static_cast<unsigned long long>(
                       descriptor->operation_count),
                   static_cast<unsigned long long>(
                       descriptor->modeled_cycles),
                   static_cast<unsigned long long>(traffic_bytes),
                   submission->tag);
    } else {
      if (descriptor->error == CORAL_OPERATOR_ERROR_NONE) {
        descriptor->error = CORAL_OPERATOR_ERROR_EXECUTION;
      }
      std::fprintf(stderr,
                   "Coral hybrid operator failed opcode=%u error=%u "
                   "tag=%u\n",
                   descriptor->opcode, descriptor->error, submission->tag);
    }
    std::fflush(stderr);
    *submission = {};
  }

  void StepCommandPipeline() {
#ifdef CORAL_GEM5_RVV_HIGHMEM
    command_adapter.AdvanceCycle();
    Gem5CoprocessorCommand command = {};
    while (command_adapter.TakeReadyToComplete(&command)) {
      AsyncOperatorSubmission* submission =
          FindAsyncSubmission(command.submission_tag);
      if (submission == nullptr || submission->descriptor == nullptr) {
        command_adapter.Complete(command.command_id, false);
        continue;
      }

      coral_operator_descriptor* descriptor = submission->descriptor;
      bool ok = true;
      if (command.opcode == Gem5MicroOpcode::kExecuteOperator &&
          !command.work_started) {
        if (local_extmem_enabled) {
          ok = DispatchGem5HybridOperator(
              descriptor, local_extmem.data(), kExtmemBase,
              local_extmem.size(), &submission->result);
          submission->kernel_done = true;
          submission->kernel_success = ok;
          submission->final_error = descriptor->error;
          descriptor->state = CORAL_OPERATOR_STATE_RUNNING;
          descriptor->error = CORAL_OPERATOR_ERROR_NONE;
          command_adapter.SetPendingLatency(
              command.submission_tag, Gem5MicroOpcode::kWriteback,
              OperatorWritebackCycles(*descriptor));
          const bool rescheduled = command_adapter.Reschedule(
              command.command_id, OperatorComputeCycles(*descriptor));
          std::fprintf(stderr,
                       "Coral command execute tag=%u id=%u source=%s "
                       "engine=%s micro_op=%s operator_opcode=%u "
                       "generic_opcode=%u kernel=%s "
                       "compute_cycles=%llu writeback_cycles=%llu\n",
                       command.submission_tag, command.command_id,
                       Gem5CommandSourceName(command.source),
                       Gem5CommandEngineName(command.engine),
                       Gem5MicroOpcodeName(command.opcode),
                       command.operator_opcode, command.generic_opcode,
                       ok ? "done" : "failed",
                       static_cast<unsigned long long>(
                           OperatorComputeCycles(*descriptor)),
                       static_cast<unsigned long long>(
                           OperatorWritebackCycles(*descriptor)));
          std::fflush(stderr);
          if (ok && rescheduled) {
            continue;
          }
          ok = false;
          submission->kernel_success = false;
          if (submission->final_error == CORAL_OPERATOR_ERROR_NONE) {
            submission->final_error = CORAL_OPERATOR_ERROR_EXECUTION;
          }
        } else {
          descriptor->state = CORAL_OPERATOR_STATE_RUNNING;
          descriptor->error = CORAL_OPERATOR_ERROR_NONE;
          submission->kernel_done = true;
          submission->kernel_success = false;
          submission->final_error = CORAL_OPERATOR_ERROR_ADDRESS;
          ok = false;
        }
      }

      if (command.opcode == Gem5MicroOpcode::kComplete &&
          !submission->kernel_success) {
        ok = false;
      }
      command_adapter.Complete(command.command_id, ok);
      std::fprintf(stderr,
                   "Coral command complete tag=%u id=%u source=%s "
                   "engine=%s micro_op=%s ok=%u pending=%zu\n",
                   command.submission_tag, command.command_id,
                   Gem5CommandSourceName(command.source),
                   Gem5CommandEngineName(command.engine),
                   Gem5MicroOpcodeName(command.opcode), ok ? 1 : 0,
                   command_adapter.PendingCount());
      std::fflush(stderr);
      if (!ok) {
        PublishOperatorCompletion(submission, false);
        continue;
      }
      if (command.opcode == Gem5MicroOpcode::kComplete &&
          command_adapter.SubmissionComplete(command.submission_tag)) {
        PublishOperatorCompletion(submission, true);
      }
    }
#endif
  }

  void TrackOperatorPhase(uint32_t marker,
                          std::chrono::steady_clock::time_point now,
                          uint64_t accesses) {
    uint32_t opcode = CORAL_OPERATOR_OP_INVALID;
    bool begin = false;
    if (marker == OPENNPUX_CORAL_MOBILENET_PROGRESS_CONV_BEGIN) {
      opcode = CORAL_OPERATOR_OP_CONV_2D_INT8;
      begin = true;
    } else if (marker == OPENNPUX_CORAL_MOBILENET_PROGRESS_CONV_END) {
      opcode = CORAL_OPERATOR_OP_CONV_2D_INT8;
    } else if (marker ==
               OPENNPUX_CORAL_MOBILENET_PROGRESS_DEPTHWISE_BEGIN) {
      opcode = CORAL_OPERATOR_OP_DEPTHWISE_CONV_2D_INT8;
      begin = true;
    } else if (marker == OPENNPUX_CORAL_MOBILENET_PROGRESS_DEPTHWISE_END) {
      opcode = CORAL_OPERATOR_OP_DEPTHWISE_CONV_2D_INT8;
    } else {
      return;
    }

    OperatorPhaseStats& stats = operator_phase_stats[opcode];
    if (begin) {
      stats.active = true;
      stats.begin_cycles = rtl_cycles;
      stats.begin_accesses = accesses;
      stats.begin_bytes = local_extmem_bytes;
      stats.begin_wall_time = now;
      return;
    }
    if (!stats.active || rtl_cycles < stats.begin_cycles ||
        accesses < stats.begin_accesses ||
        local_extmem_bytes < stats.begin_bytes) {
      return;
    }
    stats.active = false;
    ++stats.count;
    stats.rtl_cycles += rtl_cycles - stats.begin_cycles;
    stats.extmem_accesses += accesses - stats.begin_accesses;
    stats.extmem_bytes += local_extmem_bytes - stats.begin_bytes;
    stats.wall_ms += static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - stats.begin_wall_time).count());
  }

  void PrintOperatorSummary() {
    std::fprintf(stderr,
                 "Coral operator summary begin mode=%u sampled_rtl_mask=0x%08x "
                 "rtl_cycles=%llu extmem_accesses=%llu extmem_bytes=%llu\n",
                 operator_mode, sampled_rtl_mask,
                 static_cast<unsigned long long>(rtl_cycles),
                 static_cast<unsigned long long>(
                     local_extmem_reads + local_extmem_writes),
                 static_cast<unsigned long long>(local_extmem_bytes));
    for (size_t opcode = 0; opcode < operator_phase_stats.size(); ++opcode) {
      const OperatorPhaseStats& stats = operator_phase_stats[opcode];
      if (stats.count == 0) {
        continue;
      }
      const uint32_t opcode_value = static_cast<uint32_t>(opcode);
      std::fprintf(stderr,
                   "Coral operator phase summary opcode=%u name=%s "
                   "count=%llu rtl_cycles=%llu wall_ms=%llu "
                   "extmem_accesses=%llu extmem_bytes=%llu\n",
                   opcode_value, OperatorName(opcode_value),
                   static_cast<unsigned long long>(stats.count),
                   static_cast<unsigned long long>(stats.rtl_cycles),
                   static_cast<unsigned long long>(stats.wall_ms),
                   static_cast<unsigned long long>(stats.extmem_accesses),
                   static_cast<unsigned long long>(stats.extmem_bytes));
    }
    for (size_t opcode = 0; opcode < hybrid_operator_stats.size();
         ++opcode) {
      const HybridOperatorStats& stats = hybrid_operator_stats[opcode];
      if (stats.count == 0) {
        continue;
      }
      const uint32_t opcode_value = static_cast<uint32_t>(opcode);
      std::fprintf(stderr,
                   "Coral hybrid operator summary opcode=%u name=%s "
                   "count=%llu host_ns=%llu operations=%llu "
                   "modeled_cycles=%llu bytes=%llu\n",
                   opcode_value, OperatorName(opcode_value),
                   static_cast<unsigned long long>(stats.count),
                   static_cast<unsigned long long>(stats.host_ns),
                   static_cast<unsigned long long>(stats.operations),
                   static_cast<unsigned long long>(stats.modeled_cycles),
                   static_cast<unsigned long long>(stats.bytes));
    }
    std::fprintf(stderr, "Coral operator summary end\n");
  }

  coral_gem5_handle()
      : context(),
        wrapper(&context),
        custom_mac(&context),
        command_adapter(),
        async_submissions(),
        firmware_progress(0),
        operator_mode(0),
        sampled_rtl_mask(ParseSampledRtlMask()),
        hybrid_status(0),
        local_extmem_enabled(false),
        local_extmem(kExtmemSize, 0),
        local_extmem_reads(0),
        local_extmem_writes(0),
        local_extmem_bytes(0),
        local_extmem_widths(),
        external_poll_addr(0),
        external_poll_count(0),
        external_wait(false),
        rtl_cycles(0),
        hybrid_ops_per_cycle(ReadEnvU64("CORAL_HYBRID_OPS_PER_CYCLE", 1)),
        hybrid_bytes_per_cycle(
            ReadEnvU64("CORAL_HYBRID_BYTES_PER_CYCLE", 16)),
        hybrid_fixed_cycles(
            ReadEnvU64("CORAL_HYBRID_FIXED_CYCLES", 0)),
        progress_cycles(0),
        progress_accesses(0),
        progress_bytes(0),
        progress_wall_time(std::chrono::steady_clock::now()),
        hybrid_operator_stats(),
        operator_phase_stats(),
        pending_dma(),
        dma_pending(false),
        dma_beat_size(0),
        dma_beat_count(0),
        last_activity_cycles(0),
        watchdog_cycles(ReadEnvU64("CORAL_AXI_WATCHDOG_CYCLES", 5000000)) {
    std::fprintf(stderr,
                 "Coral hybrid latency model ops_per_cycle=%llu "
                 "bytes_per_cycle=%llu fixed_cycles=%llu\n",
                 static_cast<unsigned long long>(hybrid_ops_per_cycle),
                 static_cast<unsigned long long>(hybrid_bytes_per_cycle),
                 static_cast<unsigned long long>(hybrid_fixed_cycles));
    std::fprintf(stderr, "Coral sampled RTL operator mask=0x%08x\n",
                 sampled_rtl_mask);
    std::fflush(stderr);
    command_adapter.ConfigureLatencyModel(
        hybrid_ops_per_cycle, hybrid_bytes_per_cycle, hybrid_fixed_cycles);
    wrapper.RegisterDeferredReadCallback([this](const AxiAddr& addr) {
      last_activity_cycles = rtl_cycles;
      if (IsHybridWordAccess(addr)) {
        AxiRData response = {};
        response.read_data_bits_id = addr.addr_bits_id;
        response.read_data_bits_last = 1;
        uint32_t value = 0;
        if (addr.addr_bits_addr == CORAL_OPERATOR_MODE_REG) {
          value = operator_mode;
        } else if (addr.addr_bits_addr == CORAL_OPERATOR_STATUS_REG) {
          value = hybrid_status;
        } else if (addr.addr_bits_addr ==
                   CORAL_OPERATOR_SAMPLED_RTL_MASK_REG) {
          value = sampled_rtl_mask;
        } else if (addr.addr_bits_addr ==
                   CORAL_OPERATOR_CAPABILITIES_REG) {
#ifdef CORAL_GEM5_RVV_HIGHMEM
          value = (UINT32_C(1) << CORAL_OPERATOR_OP_CONV_2D_INT8) |
                  (UINT32_C(1) <<
                   CORAL_OPERATOR_OP_DEPTHWISE_CONV_2D_INT8) |
                  (UINT32_C(1) << CORAL_OPERATOR_OP_MATMUL_INT8) |
                  (UINT32_C(1) <<
                   CORAL_OPERATOR_OP_FULLY_CONNECTED_INT8) |
                  (UINT32_C(1) << CORAL_OPERATOR_OP_ADD_INT8) |
                  (UINT32_C(1) << CORAL_OPERATOR_OP_SOFTMAX) |
                  (UINT32_C(1) << CORAL_OPERATOR_OP_LAYER_NORM) |
                  (UINT32_C(1) << CORAL_OPERATOR_OP_QWEN_TINY_INFER) |
                  (UINT32_C(1) << CORAL_OPERATOR_OP_GPTQ_MATMUL_INT4) |
                  (UINT32_C(1) << CORAL_OPERATOR_OP_GPTQ_GATED_MLP) |
                  (UINT32_C(1) << CORAL_OPERATOR_OP_GPTQ_PAGED_MATMUL) |
                  (UINT32_C(1) << CORAL_OPERATOR_OP_GENERIC_COMMAND);
#endif
        }
        auto* destination = reinterpret_cast<uint8_t*>(
            &response.read_data_bits_data[0]);
        std::memcpy(destination + (addr.addr_bits_addr & 0xf), &value,
                    sizeof(value));
        wrapper.QueueReadResponse(response);
        return;
      }
      if (custom_mac.Contains(addr.addr_bits_addr, 4)) {
        AxiRData response = {};
        response.read_data_bits_id = addr.addr_bits_id;
        response.read_data_bits_last = 1;
        if (!IsCustomWordAccess(addr)) {
          response.read_data_bits_resp = kAxiSlvErr;
        } else {
          const uint32_t value =
              addr.addr_bits_addr == kFirmwareProgressAddr
                  ? firmware_progress
                  : custom_mac.Read32(addr.addr_bits_addr);
          auto* destination = reinterpret_cast<uint8_t*>(
              &response.read_data_bits_data[0]);
          std::memcpy(destination + (addr.addr_bits_addr & 0xf), &value,
                      sizeof(value));
        }
        wrapper.QueueReadResponse(response);
        return;
      }
      const char* read_reject_reason = nullptr;
      if (!BuildGem5DmaReadRequest(
              addr, &pending_dma, &dma_beat_size, &dma_beat_count,
              &read_reject_reason)) {
        DumpRejectedAxi("read", addr, nullptr, read_reject_reason);
        AxiRData response = {};
        response.read_data_bits_id = addr.addr_bits_id;
        response.read_data_bits_resp = kAxiSlvErr;
        response.read_data_bits_last = 1;
        wrapper.QueueReadResponse(response);
        return;
      }
      if (local_extmem_enabled && pending_dma.addr >= kExtmemBase &&
          pending_dma.addr - kExtmemBase <=
              local_extmem.size() - pending_dma.size) {
        const auto* source = local_extmem.data() +
                             (pending_dma.addr - kExtmemBase);
        for (uint32_t beat = 0; beat < dma_beat_count; ++beat) {
          AxiRData response = {};
          const uint32_t beat_addr =
              pending_dma.addr + beat * dma_beat_size;
          const uint32_t lane =
              beat_addr & (CORAL_GEM5_AXI_DATA_BYTES - 1);
          auto* destination = reinterpret_cast<uint8_t*>(
              &response.read_data_bits_data[0]);
          std::memcpy(destination + lane,
                      source + beat * dma_beat_size, dma_beat_size);
          response.read_data_bits_id = pending_dma.id;
          response.read_data_bits_last = beat + 1 == dma_beat_count;
          wrapper.QueueReadResponse(response);
        }
        ObserveExtmemRead(pending_dma.addr, pending_dma.size, source);
        ++local_extmem_reads;
        local_extmem_bytes += pending_dma.size;
        ++local_extmem_widths[AccessWidthBucket(pending_dma.size)];
        const uint64_t accesses = local_extmem_reads + local_extmem_writes;
        if (accesses <= 10 || accesses % 100000 == 0) {
          std::fprintf(stderr,
                       "Coral local EXTMEM accesses=%llu reads=%llu "
                       "writes=%llu bytes=%llu last=read@0x%08x/%u\n",
                       static_cast<unsigned long long>(accesses),
                       static_cast<unsigned long long>(local_extmem_reads),
                       static_cast<unsigned long long>(local_extmem_writes),
                       static_cast<unsigned long long>(local_extmem_bytes),
                       pending_dma.addr, pending_dma.size);
          std::fflush(stderr);
        }
        std::memset(&pending_dma, 0, sizeof(pending_dma));
        return;
      }
      dma_pending = true;
    });
    wrapper.RegisterDeferredWriteCallback(
        [this](const AxiAddr& addr, const std::vector<AxiWData>& data) {
          last_activity_cycles = rtl_cycles;
          if (IsHybridWordAccess(addr)) {
            AxiWResp response = {};
            response.write_resp_bits_id = addr.addr_bits_id;
            const uint32_t lane = addr.addr_bits_addr & 0xf;
            if (data.size() != 1 || data[0].write_data_bits_last == 0 ||
                ((data[0].write_data_bits_strb >> lane) & 0xf) != 0xf) {
              response.write_resp_bits_resp = kAxiSlvErr;
            } else {
              uint32_t command = 0;
              const auto* source = reinterpret_cast<const uint8_t*>(
                  &data[0].write_data_bits_data[0]);
              std::memcpy(&command, source + lane, sizeof(command));
              if (addr.addr_bits_addr != CORAL_OPERATOR_DOORBELL_REG ||
                  (operator_mode != CORAL_OPERATOR_MODE_HYBRID &&
                   operator_mode != CORAL_OPERATOR_MODE_SAMPLED) ||
                  command < kExtmemBase ||
                  command - kExtmemBase >
                      local_extmem.size() -
                          sizeof(coral_operator_descriptor)) {
                response.write_resp_bits_resp = kAxiSlvErr;
              } else {
                auto* descriptor = reinterpret_cast<
                    coral_operator_descriptor*>(
                        local_extmem.data() + (command - kExtmemBase));
#ifdef CORAL_GEM5_RVV_HIGHMEM
                uint32_t submission_tag = 0;
                uint32_t command_error = CORAL_OPERATOR_ERROR_NONE;
                const Gem5CommandSource source =
                    (descriptor->flags &
                     CORAL_OPERATOR_FLAG_CUSTOM_INSTRUCTION) != 0
                        ? Gem5CommandSource::kCustomInstruction
                        : Gem5CommandSource::kMmioDoorbell;
                if (!command_adapter.Submit(source, command, *descriptor,
                                            &submission_tag,
                                            &command_error)) {
                  descriptor->state = CORAL_OPERATOR_STATE_ERROR;
                  descriptor->error = command_error;
                  std::fprintf(stderr,
                               "Coral command submission rejected error=%u\n",
                               command_error);
                  std::fflush(stderr);
                  hybrid_status = CORAL_OPERATOR_STATE_ERROR;
                } else if (!AddAsyncSubmission(submission_tag, descriptor)) {
                  descriptor->state = CORAL_OPERATOR_STATE_ERROR;
                  descriptor->error = CORAL_OPERATOR_ERROR_EXECUTION;
                  command_adapter.FailSubmission(submission_tag);
                  hybrid_status = CORAL_OPERATOR_STATE_ERROR;
                  std::fprintf(stderr,
                               "Coral command submission tag=%u has no "
                               "async slot\n",
                               submission_tag);
                  std::fflush(stderr);
                } else {
                  descriptor->state = CORAL_OPERATOR_STATE_RUNNING;
                  descriptor->error = CORAL_OPERATOR_ERROR_NONE;
                  hybrid_status = CORAL_OPERATOR_STATE_RUNNING;
                  std::fprintf(stderr,
                               "Coral command submission tag=%u "
                               "operator_opcode=%u generic_opcode=%u "
                               "source=%s pending=%zu\n",
                               submission_tag, descriptor->opcode,
                               descriptor->opcode ==
                                       CORAL_OPERATOR_OP_GENERIC_COMMAND
                                   ? descriptor->reserved[0]
                                   : 0,
                               Gem5CommandSourceName(source),
                               command_adapter.PendingCount());
                  std::fflush(stderr);
                }
#else
                descriptor->state = CORAL_OPERATOR_STATE_ERROR;
                descriptor->error = CORAL_OPERATOR_ERROR_UNSUPPORTED;
                hybrid_status = CORAL_OPERATOR_STATE_ERROR;
#endif
              }
            }
            wrapper.QueueWriteResponse(response);
            return;
          }
          if (custom_mac.Contains(addr.addr_bits_addr, 4)) {
            AxiWResp response = {};
            response.write_resp_bits_id = addr.addr_bits_id;
            const uint32_t lane = addr.addr_bits_addr & 0xf;
            if (!IsCustomWordAccess(addr) || data.size() != 1 ||
                data[0].write_data_bits_last == 0 ||
                ((data[0].write_data_bits_strb >> lane) & 0xf) != 0xf) {
              response.write_resp_bits_resp = kAxiSlvErr;
            } else {
              uint32_t value = 0;
              const auto* source = reinterpret_cast<const uint8_t*>(
                  &data[0].write_data_bits_data[0]);
              std::memcpy(&value, source + lane, sizeof(value));
              if (addr.addr_bits_addr == kFirmwareProgressAddr) {
                firmware_progress = value;
                std::fprintf(stderr, "Coral firmware progress=0x%08x\n",
                             firmware_progress);
                const uint64_t accesses =
                    local_extmem_reads + local_extmem_writes;
                const auto now = std::chrono::steady_clock::now();
                const auto wall_ms =
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        now - progress_wall_time).count();
                TrackOperatorPhase(firmware_progress, now, accesses);
                std::fprintf(
                    stderr,
                    "Coral phase stats marker=0x%08x cycles=%llu "
                    "delta_cycles=%llu wall_ms=%lld accesses=%llu "
                    "delta_accesses=%llu bytes=%llu delta_bytes=%llu "
                    "widths=1:%llu,2:%llu,4:%llu,8:%llu,16:%llu,other:%llu\n",
                    firmware_progress,
                    static_cast<unsigned long long>(rtl_cycles),
                    static_cast<unsigned long long>(
                        rtl_cycles - progress_cycles),
                    static_cast<long long>(wall_ms),
                    static_cast<unsigned long long>(accesses),
                    static_cast<unsigned long long>(
                        accesses - progress_accesses),
                    static_cast<unsigned long long>(local_extmem_bytes),
                    static_cast<unsigned long long>(
                        local_extmem_bytes - progress_bytes),
                    static_cast<unsigned long long>(local_extmem_widths[0]),
                    static_cast<unsigned long long>(local_extmem_widths[1]),
                    static_cast<unsigned long long>(local_extmem_widths[2]),
                    static_cast<unsigned long long>(local_extmem_widths[3]),
                    static_cast<unsigned long long>(local_extmem_widths[4]),
                    static_cast<unsigned long long>(local_extmem_widths[5]));
                progress_cycles = rtl_cycles;
                progress_accesses = accesses;
                progress_bytes = local_extmem_bytes;
                progress_wall_time = now;
                if (firmware_progress ==
                    OPENNPUX_CORAL_MOBILENET_PROGRESS_INVOKE_END) {
                  PrintOperatorSummary();
                }
                std::fflush(stderr);
              } else {
                custom_mac.Write32(addr.addr_bits_addr, value);
              }
            }
            wrapper.QueueWriteResponse(response);
            return;
          }
          // Byte-enable-aware local EXTMEM absorb: the Coral LSU emits
          // 16-byte line writes with partial strobes (e.g. vse8.v tails),
          // which coral_gem5_dma_request cannot represent (the ABI has no
          // strobe field). Absorb any INCR write that lies fully inside the
          // local EXTMEM window with a per-byte merge; masked writes that
          // cannot be absorbed fall through to the strict gem5 DMA path
          // below and are rejected loudly.
          const uint32_t bytes_per_beat = 1u << addr.addr_bits_size;
          const uint32_t beats =
              static_cast<uint32_t>(addr.addr_bits_len) + 1;
          const uint64_t total_bytes =
              static_cast<uint64_t>(bytes_per_beat) * beats;
          const uint64_t window_end =
              static_cast<uint64_t>(kExtmemBase) + local_extmem.size();
          if (local_extmem_enabled && addr.addr_bits_burst == 1 &&
              addr.addr_bits_size <= 4 && data.size() == beats &&
              addr.addr_bits_addr >= kExtmemBase &&
              static_cast<uint64_t>(addr.addr_bits_addr) + total_bytes - 1 <
                  window_end) {
            // Validate before applying: a single enabled byte outside the
            // window rejects the whole transaction (no partial apply).
            bool enabled_bytes_in_window = true;
            for (uint32_t beat = 0; beat < beats && enabled_bytes_in_window;
                 ++beat) {
              const uint64_t beat_addr =
                  static_cast<uint64_t>(addr.addr_bits_addr) +
                  beat * bytes_per_beat;
              const uint32_t lane =
                  beat_addr & (CORAL_GEM5_AXI_DATA_BYTES - 1);
              for (uint32_t i = 0; i < bytes_per_beat; ++i) {
                if ((data[beat].write_data_bits_strb &
                     (1u << (lane + i))) != 0 &&
                    beat_addr + i >= window_end) {
                  enabled_bytes_in_window = false;
                  break;
                }
              }
            }
            if (!enabled_bytes_in_window) {
              DumpRejectedAxi("write", addr, &data, "extmem-window-overflow");
              AxiWResp response = {};
              response.write_resp_bits_id = addr.addr_bits_id;
              response.write_resp_bits_resp = kAxiSlvErr;
              wrapper.QueueWriteResponse(response);
              return;
            }
            for (uint32_t beat = 0; beat < beats; ++beat) {
              const uint64_t beat_addr =
                  static_cast<uint64_t>(addr.addr_bits_addr) +
                  beat * bytes_per_beat;
              const uint32_t lane =
                  beat_addr & (CORAL_GEM5_AXI_DATA_BYTES - 1);
              const auto* source = reinterpret_cast<const uint8_t*>(
                  &data[beat].write_data_bits_data[0]);
              uint8_t* destination =
                  local_extmem.data() + (beat_addr - kExtmemBase);
              for (uint32_t i = 0; i < bytes_per_beat; ++i) {
                if ((data[beat].write_data_bits_strb &
                     (1u << (lane + i))) != 0) {
                  destination[i] = source[lane + i];
                }
              }
            }
            NotifyExtmemWrite();
            ++local_extmem_writes;
            local_extmem_bytes += total_bytes;
            ++local_extmem_widths[
                AccessWidthBucket(static_cast<uint32_t>(total_bytes))];
            const uint64_t accesses = local_extmem_reads + local_extmem_writes;
            if (accesses <= 10 || accesses % 100000 == 0) {
              std::fprintf(stderr,
                           "Coral local EXTMEM accesses=%llu reads=%llu "
                           "writes=%llu bytes=%llu last=write@0x%08x/%u\n",
                           static_cast<unsigned long long>(accesses),
                           static_cast<unsigned long long>(local_extmem_reads),
                           static_cast<unsigned long long>(local_extmem_writes),
                           static_cast<unsigned long long>(local_extmem_bytes),
                           addr.addr_bits_addr,
                           static_cast<uint32_t>(total_bytes));
              std::fflush(stderr);
            }
            AxiWResp response = {};
            response.write_resp_bits_id = addr.addr_bits_id;
            wrapper.QueueWriteResponse(response);
            return;
          }
          const char* write_reject_reason = nullptr;
          if (!BuildGem5DmaWriteRequest(addr, data, &pending_dma,
                                        &write_reject_reason)) {
            if (write_reject_reason != nullptr &&
                std::strcmp(write_reject_reason, "partial-strb") == 0 &&
                addr.addr_bits_addr >= kExtmemBase) {
              // Masked write into the EXTMEM region that the absorb path
              // above could not take (local EXTMEM disabled or window
              // overflow): the gem5 DMA ABI carries no byte enables.
              write_reject_reason = "extmem-masked-nonlocal";
            }
            DumpRejectedAxi("write", addr, &data, write_reject_reason);
            AxiWResp response = {};
            response.write_resp_bits_id = addr.addr_bits_id;
            response.write_resp_bits_resp = kAxiSlvErr;
            wrapper.QueueWriteResponse(response);
            return;
          }
          if (local_extmem_enabled && pending_dma.addr >= kExtmemBase &&
              pending_dma.addr - kExtmemBase <=
                  local_extmem.size() - pending_dma.size) {
            std::memcpy(local_extmem.data() +
                            (pending_dma.addr - kExtmemBase),
                        pending_dma.data, pending_dma.size);
            NotifyExtmemWrite();
            ++local_extmem_writes;
            local_extmem_bytes += pending_dma.size;
            ++local_extmem_widths[AccessWidthBucket(pending_dma.size)];
            const uint64_t accesses = local_extmem_reads + local_extmem_writes;
            if (accesses <= 10 || accesses % 100000 == 0) {
              std::fprintf(stderr,
                           "Coral local EXTMEM accesses=%llu reads=%llu "
                           "writes=%llu bytes=%llu last=write@0x%08x/%u\n",
                           static_cast<unsigned long long>(accesses),
                           static_cast<unsigned long long>(local_extmem_reads),
                           static_cast<unsigned long long>(local_extmem_writes),
                           static_cast<unsigned long long>(local_extmem_bytes),
                           pending_dma.addr, pending_dma.size);
              std::fflush(stderr);
            }
            AxiWResp response = {};
            response.write_resp_bits_id = pending_dma.id;
            wrapper.QueueWriteResponse(response);
            std::memset(&pending_dma, 0, sizeof(pending_dma));
            return;
          }
          dma_pending = true;
        });
    wrapper.Reset();
  }
};

extern "C" uint32_t
coral_gem5_abi_version(void)
{
  return CORAL_GEM5_ABI_VERSION;
}

extern "C" coral_gem5_handle*
coral_gem5_create(void)
{
  return new (std::nothrow) coral_gem5_handle();
}

extern "C" void
coral_gem5_destroy(coral_gem5_handle* handle)
{
  delete handle;
}

extern "C" int
coral_gem5_reset(coral_gem5_handle* handle)
{
  if (handle == nullptr) {
    return -1;
  }
  handle->wrapper.Reset();
  handle->custom_mac.Reset();
  handle->command_adapter.Reset();
  handle->sim_host_pager.Reset();
  handle->sim_host_numerical.Reset();
  handle->async_submissions.fill({});
  handle->firmware_progress = 0;
  handle->hybrid_status = 0;
  handle->external_poll_addr = 0;
  handle->external_poll_count = 0;
  handle->external_wait = false;
  handle->rtl_cycles = 0;
  handle->progress_cycles = 0;
  handle->progress_accesses = 0;
  handle->progress_bytes = 0;
  handle->progress_wall_time = std::chrono::steady_clock::now();
  return 0;
}

extern "C" int
coral_gem5_mmio_read(
    coral_gem5_handle* handle, uint32_t addr, void* data, size_t size)
{
  if (handle == nullptr || data == nullptr || size == 0) {
    return -1;
  }
  const std::vector<uint8_t> value =
      handle->wrapper.Read(TranslateSlaveAddress(addr), size);
  if (value.size() != size) {
    return -1;
  }
  std::memcpy(data, value.data(), size);
  return 0;
}

extern "C" int
coral_gem5_mmio_write(
    coral_gem5_handle* handle, uint32_t addr, const void* data, size_t size)
{
  if (handle == nullptr || data == nullptr || size == 0) {
    return -1;
  }
  handle->wrapper.Write(
      TranslateSlaveAddress(addr), size, reinterpret_cast<const char*>(data));
  return 0;
}

extern "C" int
coral_gem5_step(coral_gem5_handle* handle, uint32_t cycles)
{
  if (handle == nullptr || cycles == 0) {
    return CORAL_GEM5_STEP_ERROR;
  }
  const int initial_page = handle->sim_host_pager.Service(
      &handle->local_extmem);
  if (initial_page < 0) {
    return CORAL_GEM5_STEP_ERROR;
  }
  if (initial_page > 0) {
    handle->NotifyExtmemWrite();
  }
  const int initial_numerical = handle->sim_host_numerical.Publish(
      &handle->local_extmem);
  if (initial_numerical < 0) {
    return CORAL_GEM5_STEP_ERROR;
  }
  if (handle->dma_pending) {
    return CORAL_GEM5_STEP_DMA_WAIT;
  }
  if (handle->external_wait) {
    return CORAL_GEM5_STEP_EXTERNAL_WAIT;
  }
  for (uint32_t i = 0; i < cycles; ++i) {
    if (handle->wrapper.HasFault()) {
      return CORAL_GEM5_STEP_FAULT;
    }
    if (handle->wrapper.IsHalted()) {
      return CORAL_GEM5_STEP_HALTED;
    }
    ++handle->rtl_cycles;
    handle->wrapper.Step();
    handle->custom_mac.StepIfActive();
    handle->StepCommandPipeline();
    const int serviced_page = handle->sim_host_pager.Service(
        &handle->local_extmem);
    if (serviced_page < 0) {
      return CORAL_GEM5_STEP_ERROR;
    }
    if (serviced_page > 0) {
      handle->NotifyExtmemWrite();
    }
    const int published_numerical = handle->sim_host_numerical.Publish(
        &handle->local_extmem);
    if (published_numerical < 0) {
      return CORAL_GEM5_STEP_ERROR;
    }
    if (handle->external_wait) {
      return CORAL_GEM5_STEP_EXTERNAL_WAIT;
    }
    if (handle->rtl_cycles - handle->last_activity_cycles >=
        handle->watchdog_cycles) {
      // No AXI master activity for a full watchdog window: dump the channel
      // handshake levels so a stalled transaction can be identified.
      handle->last_activity_cycles = handle->rtl_cycles;
      handle->wrapper.DumpMasterChannels(handle->rtl_cycles);
    }
    if (handle->dma_pending) {
      return CORAL_GEM5_STEP_DMA_WAIT;
    }
    if (handle->wrapper.HasFault()) {
      return CORAL_GEM5_STEP_FAULT;
    }
    if (handle->wrapper.IsHalted()) {
      return CORAL_GEM5_STEP_HALTED;
    }
  }
  return handle->wrapper.IsWfi() ? CORAL_GEM5_STEP_WFI :
                                   CORAL_GEM5_STEP_RUNNING;
}

extern "C" uint64_t
coral_gem5_cycle_count(coral_gem5_handle* handle)
{
  return handle == nullptr ? 0 : handle->rtl_cycles;
}

extern "C" int
coral_gem5_dma_request_get(
    coral_gem5_handle* handle, coral_gem5_dma_request* request)
{
  if (handle == nullptr || request == nullptr) {
    return -1;
  }
  if (!handle->dma_pending) {
    return 0;
  }
  *request = handle->pending_dma;
  return 1;
}

extern "C" int
coral_gem5_dma_complete(
    coral_gem5_handle* handle, const void* data, size_t size, int error)
{
  if (handle == nullptr || !handle->dma_pending) {
    return -1;
  }

  const uint8_t response = error ? 2 : 0;
  if (handle->pending_dma.type == CORAL_GEM5_DMA_READ) {
    if (!error &&
        (data == nullptr || size != handle->pending_dma.size)) {
      return -1;
    }
    if (!error) {
      const auto* source = reinterpret_cast<const uint8_t*>(data);
      for (uint32_t beat = 0; beat < handle->dma_beat_count; ++beat) {
        AxiRData axi_response = {};
        const uint32_t beat_addr =
            handle->pending_dma.addr + beat * handle->dma_beat_size;
        const uint32_t lane =
            beat_addr & (CORAL_GEM5_AXI_DATA_BYTES - 1);
        auto* destination = reinterpret_cast<uint8_t*>(
            &axi_response.read_data_bits_data[0]);
        std::memcpy(destination + lane,
                    source + beat * handle->dma_beat_size,
                    handle->dma_beat_size);
        axi_response.read_data_bits_id = handle->pending_dma.id;
        axi_response.read_data_bits_resp = 0;
        axi_response.read_data_bits_last =
            beat + 1 == handle->dma_beat_count;
        handle->wrapper.QueueReadResponse(axi_response);
      }
    } else {
      AxiRData axi_response = {};
      axi_response.read_data_bits_id = handle->pending_dma.id;
      axi_response.read_data_bits_resp = response;
      axi_response.read_data_bits_last = 1;
      handle->wrapper.QueueReadResponse(axi_response);
    }
  } else if (handle->pending_dma.type == CORAL_GEM5_DMA_WRITE) {
    AxiWResp axi_response = {};
    axi_response.write_resp_bits_id = handle->pending_dma.id;
    axi_response.write_resp_bits_resp = response;
    handle->wrapper.QueueWriteResponse(axi_response);
  } else {
    return -1;
  }

  handle->dma_pending = false;
  std::memset(&handle->pending_dma, 0, sizeof(handle->pending_dma));
  return 0;
}

extern "C" int
coral_gem5_extmem_enable(coral_gem5_handle* handle, int enable)
{
  if (handle == nullptr || handle->dma_pending) {
    return -1;
  }
  handle->local_extmem_enabled = enable != 0;
  handle->NotifyExtmemWrite();
  if (handle->local_extmem_enabled) {
    std::fill(handle->local_extmem.begin(), handle->local_extmem.end(), 0);
    handle->local_extmem_reads = 0;
    handle->local_extmem_writes = 0;
    handle->local_extmem_bytes = 0;
    handle->local_extmem_widths.fill(0);
  }
  return 0;
}

extern "C" int
coral_gem5_extmem_read(
    coral_gem5_handle* handle, uint32_t addr, void* data, size_t size)
{
  if (handle == nullptr || data == nullptr || !handle->local_extmem_enabled ||
      size > kExtmemSize || addr < kExtmemBase ||
      addr - kExtmemBase > kExtmemSize - size) {
    return -1;
  }
  std::memcpy(data, handle->local_extmem.data() + (addr - kExtmemBase), size);
  return 0;
}

extern "C" int
coral_gem5_extmem_write(
    coral_gem5_handle* handle, uint32_t addr, const void* data, size_t size)
{
  if (handle == nullptr || data == nullptr || !handle->local_extmem_enabled ||
      size > kExtmemSize || addr < kExtmemBase ||
      addr - kExtmemBase > kExtmemSize - size) {
    return -1;
  }
  std::memcpy(handle->local_extmem.data() + (addr - kExtmemBase), data, size);
  handle->NotifyExtmemWrite();
  return 0;
}

extern "C" int
coral_gem5_operator_mode(coral_gem5_handle* handle, uint32_t mode)
{
  if (handle == nullptr || mode > CORAL_OPERATOR_MODE_SAMPLED) {
    return -1;
  }
  handle->operator_mode = mode;
  return 0;
}
