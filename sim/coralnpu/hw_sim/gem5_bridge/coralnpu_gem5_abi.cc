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
#include "hw_sim/gem5_bridge/gem5_custom_mac.h"
#include "hw_sim/gem5_bridge/gem5_dma_request_builder.h"
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
           OperatorMaskBit(CORAL_OPERATOR_OP_LAYER_NORM);
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

struct coral_gem5_handle {
  VerilatedContext context;
  Gem5CoreMiniAxiWrapper wrapper;
  Gem5CustomMac custom_mac;
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
        dma_beat_count(0) {
    std::fprintf(stderr,
                 "Coral hybrid latency model ops_per_cycle=%llu "
                 "bytes_per_cycle=%llu fixed_cycles=%llu\n",
                 static_cast<unsigned long long>(hybrid_ops_per_cycle),
                 static_cast<unsigned long long>(hybrid_bytes_per_cycle),
                 static_cast<unsigned long long>(hybrid_fixed_cycles));
    std::fprintf(stderr, "Coral sampled RTL operator mask=0x%08x\n",
                 sampled_rtl_mask);
    std::fflush(stderr);
    wrapper.RegisterDeferredReadCallback([this](const AxiAddr& addr) {
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
                  (UINT32_C(1) << CORAL_OPERATOR_OP_LAYER_NORM);
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
      if (!BuildGem5DmaReadRequest(
              addr, &pending_dma, &dma_beat_size, &dma_beat_count)) {
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
                Gem5HybridOperatorResult result = {};
                bool ok = false;
                if (local_extmem_enabled) {
                  ok = DispatchGem5HybridOperator(
                        descriptor, local_extmem.data(), kExtmemBase,
                        local_extmem.size(), &result);
                } else {
                  descriptor->state = CORAL_OPERATOR_STATE_ERROR;
                  descriptor->error = CORAL_OPERATOR_ERROR_ADDRESS;
                }
                hybrid_status = descriptor->state;
                if (ok) {
                  const uint64_t traffic_bytes =
                      descriptor->bytes_read + descriptor->bytes_written;
                  descriptor->modeled_cycles =
                      hybrid_fixed_cycles +
                      DivCeil(descriptor->operation_count,
                              hybrid_ops_per_cycle) +
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
                               "Coral hybrid operator complete opcode=%u "
                               "name=%s count=%llu host_ns=%llu "
                               "operations=%llu modeled_cycles=%llu "
                               "bytes=%llu\n",
                               descriptor->opcode,
                               OperatorName(descriptor->opcode),
                               static_cast<size_t>(descriptor->opcode) <
                                   hybrid_operator_stats.size() ?
                                   static_cast<unsigned long long>(
                                       hybrid_operator_stats[
                                           descriptor->opcode].count) : 0ULL,
                               static_cast<unsigned long long>(
                                   descriptor->host_elapsed_ns),
                               static_cast<unsigned long long>(
                                   descriptor->operation_count),
                               static_cast<unsigned long long>(
                                   descriptor->modeled_cycles),
                               static_cast<unsigned long long>(
                                   traffic_bytes));
                  std::fflush(stderr);
                } else {
                  std::fprintf(stderr,
                               "Coral hybrid operator failed opcode=%u "
                               "error=%u\n",
                               descriptor->opcode, descriptor->error);
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
          if (!BuildGem5DmaWriteRequest(addr, data, &pending_dma)) {
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
  handle->firmware_progress = 0;
  handle->hybrid_status = 0;
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
    return -1;
  }
  if (handle->dma_pending) {
    return 2;
  }
  for (uint32_t i = 0; i < cycles; ++i) {
    if (handle->wrapper.HasFault()) {
      return 4;
    }
    if (handle->wrapper.IsHalted()) {
      return 1;
    }
    ++handle->rtl_cycles;
    handle->wrapper.Step();
    handle->custom_mac.StepIfActive();
    if (handle->dma_pending) {
      return 2;
    }
    if (handle->wrapper.HasFault()) {
      return 4;
    }
    if (handle->wrapper.IsHalted()) {
      return 1;
    }
  }
  return handle->wrapper.IsWfi() ? 3 : 0;
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
