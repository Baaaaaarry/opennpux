#include "hw_sim/gem5_bridge/gem5_sim_host_pager.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "hw_sim/gem5_bridge/npu_submission.h"
#include "hw_sim/gem5_bridge/npu_weight_queue.h"
#include "hw_sim/gem5_bridge/npu_weight_residency.h"

namespace {

constexpr uint64_t kExtmemBase = UINT64_C(0x20000000);

template <typename T>
T* At(std::vector<uint8_t>* memory, uint64_t address) {
  return reinterpret_cast<T*>(memory->data() + (address - kExtmemBase));
}

}  // namespace

Gem5SimHostPager::Gem5SimHostPager() {
  const char* path = std::getenv("CORAL_SIM_HOST_PAGE_BUNDLE");
  if (path == nullptr || path[0] == '\0') {
    return;
  }
  configured_ = true;
  bundle_ = std::fopen(path, "rb");
  if (bundle_ == nullptr ||
      std::fread(&header_, 1, sizeof(header_), bundle_) != sizeof(header_) ||
      header_.magic != OPENNPUX_NPU_PAGE_BUNDLE_MAGIC ||
      header_.version != OPENNPUX_NPU_PAGE_BUNDLE_VERSION ||
      header_.header_size != sizeof(header_) ||
      header_.record_size != sizeof(opennpux_npu_page_bundle_record) ||
      header_.transfer_size == 0 || header_.record_count == 0) {
    Fail("invalid or unreadable page bundle");
    if (bundle_ != nullptr) {
      std::fclose(bundle_);
      bundle_ = nullptr;
    }
    return;
  }
  std::fprintf(stderr,
               "Coral sim-host pager loaded path=%s records=%u "
               "transfer=%u payload=%llu\n",
               path, header_.record_count, header_.transfer_size,
               static_cast<unsigned long long>(header_.payload_bytes));
  std::fflush(stderr);
}

Gem5SimHostPager::~Gem5SimHostPager() {
  if (bundle_ != nullptr) {
    std::fclose(bundle_);
  }
}

void Gem5SimHostPager::Reset() {
  serviced_ = 0;
  consumed_ = 0;
  transferred_ = 0;
  last_command_id_ = 0;
  have_last_command_ = false;
  failed_ = false;
  if (bundle_ != nullptr) {
    if (std::fseek(bundle_, sizeof(header_), SEEK_SET) != 0) {
      Fail("cannot rewind page bundle");
    }
  }
}

int Gem5SimHostPager::RewindBundle() {
  if (bundle_ == nullptr ||
      std::fseek(bundle_, sizeof(header_), SEEK_SET) != 0) {
    return Fail("cannot rewind page bundle for next execution step");
  }
  consumed_ = 0;
  have_last_command_ = false;
  return 0;
}

bool Gem5SimHostPager::RangeValid(
    uint64_t address, uint64_t size,
    const std::vector<uint8_t>& extmem) const {
  return address >= kExtmemBase && address - kExtmemBase <= extmem.size() &&
      size <= extmem.size() - (address - kExtmemBase);
}

int Gem5SimHostPager::Fail(const char* reason) {
  if (!failed_) {
    std::fprintf(stderr, "Coral sim-host pager error: %s serviced=%llu\n",
                 reason, static_cast<unsigned long long>(serviced_));
    std::fflush(stderr);
  }
  failed_ = true;
  return -1;
}

int Gem5SimHostPager::Service(std::vector<uint8_t>* extmem) {
  if (!configured_) {
    return 0;
  }
  if (bundle_ == nullptr || failed_ || extmem == nullptr || extmem->size() <
          sizeof(opennpux_npu_invocation_header)) {
    return Fail("invalid EXTMEM state");
  }
  const auto* invocation = reinterpret_cast<
      const opennpux_npu_invocation_header*>(extmem->data());
  if (invocation->magic != OPENNPUX_NPU_INVOCATION_MAGIC ||
      invocation->version != OPENNPUX_NPU_INVOCATION_VERSION ||
      invocation->header_size != sizeof(*invocation) ||
      invocation->binding_count > OPENNPUX_NPU_MAX_BINDINGS ||
      invocation->binding_offset > extmem->size() ||
      invocation->binding_count >
          (extmem->size() - invocation->binding_offset) /
              sizeof(opennpux_npu_tensor_binding)) {
    return 0;
  }
  const auto* bindings = reinterpret_cast<const opennpux_npu_tensor_binding*>(
      extmem->data() + invocation->binding_offset);
  const opennpux_npu_tensor_binding* queue_binding = nullptr;
  const opennpux_npu_tensor_binding* cache_binding = nullptr;
  const opennpux_npu_tensor_binding* residency_binding = nullptr;
  for (uint32_t index = 0; index < invocation->binding_count; ++index) {
    if ((bindings[index].flags & OPENNPUX_NPU_BIND_PAGE_QUEUE) != 0) {
      queue_binding = &bindings[index];
    }
    if ((bindings[index].flags & OPENNPUX_NPU_BIND_PAGE_CACHE) != 0) {
      cache_binding = &bindings[index];
    }
    if ((bindings[index].flags & OPENNPUX_NPU_BIND_PAGE_RESIDENCY) != 0) {
      residency_binding = &bindings[index];
    }
  }
  if (queue_binding == nullptr || cache_binding == nullptr ||
      residency_binding == nullptr) {
    return 0;
  }
  if (!RangeValid(queue_binding->device_address, queue_binding->byte_size,
                  *extmem) ||
      !RangeValid(cache_binding->device_address, cache_binding->byte_size,
                  *extmem) ||
      !RangeValid(residency_binding->device_address,
                  residency_binding->byte_size, *extmem)) {
    return Fail("paging binding outside EXTMEM");
  }
  auto* queue = At<opennpux_npu_weight_queue_header>(
      extmem, queue_binding->device_address);
  if (queue->magic != OPENNPUX_NPU_WEIGHT_QUEUE_MAGIC ||
      queue->version != OPENNPUX_NPU_WEIGHT_QUEUE_VERSION ||
      queue->header_size != sizeof(*queue) ||
      queue->entry_size != sizeof(opennpux_npu_page_fault) ||
      queue->capacity == 0 || queue->service_index == queue->producer_index) {
    return 0;
  }
  const uint64_t fault_address = queue_binding->device_address +
      sizeof(*queue) + static_cast<uint64_t>(
          queue->service_index % queue->capacity) * sizeof(opennpux_npu_page_fault);
  if (!RangeValid(fault_address, sizeof(opennpux_npu_page_fault), *extmem)) {
    return Fail("page fault outside queue binding");
  }
  auto* fault = At<opennpux_npu_page_fault>(extmem, fault_address);
  if (fault->state != OPENNPUX_NPU_PAGE_FAULT_PENDING) {
    return 0;
  }
  if (fault->magic != OPENNPUX_NPU_PAGE_FAULT_MAGIC ||
      fault->version != OPENNPUX_NPU_PAGE_FAULT_VERSION ||
      fault->struct_size != sizeof(*fault) ||
      fault->page_size != header_.transfer_size) {
    return Fail("page fault does not match bundle ABI");
  }
  if (consumed_ >= header_.record_count ||
      (have_last_command_ && fault->command_id < last_command_id_)) {
    if (RewindBundle() != 0) {
      return -1;
    }
  }
  opennpux_npu_page_bundle_record record = {};
  for (;;) {
    if (consumed_ >= header_.record_count ||
        std::fread(&record, 1, sizeof(record), bundle_) != sizeof(record) ||
        record.page_size != header_.transfer_size) {
      return Fail("page bundle record order mismatch");
    }
    ++consumed_;
    if (record.command_id >= fault->command_id) {
      break;
    }
    if (std::fseek(bundle_, header_.transfer_size, SEEK_CUR) != 0) {
      return Fail("cannot skip unused bundled page");
    }
  }
  if (record.command_id != fault->command_id) {
    return Fail("page bundle is missing requested command");
  }
  const uint32_t cache_slots = static_cast<uint32_t>(
      cache_binding->byte_size / header_.transfer_size);
  if (cache_slots == 0) {
    return Fail("page cache has no slots");
  }
  const uint32_t slot = fault->command_id % cache_slots;
  const uint64_t cache_address = cache_binding->device_address +
      static_cast<uint64_t>(slot) * header_.transfer_size;
  if (!RangeValid(cache_address, header_.transfer_size, *extmem) ||
      std::fread(At<uint8_t>(extmem, cache_address), 1,
                 header_.transfer_size, bundle_) != header_.transfer_size) {
    return Fail("cannot transfer bundled page into EXTMEM");
  }
  fault->shard_index = record.shard_index;
  fault->file_offset = record.file_offset;
  fault->expert_id = record.expert_id;
  fault->role_id = record.role_id;
  fault->component_id = record.component_id;
  fault->range_file_offset = record.range_file_offset;
  fault->range_size = record.range_size;
  fault->cache_slot = slot;
  fault->error_code = 0;
  fault->flags = (record.flags & OPENNPUX_NPU_PAGE_BUNDLE_LAST) != 0 ?
      OPENNPUX_NPU_PAGE_FAULT_LAST : 0;
  auto* residency = At<opennpux_npu_weight_residency_header>(
      extmem, residency_binding->device_address);
  const uint64_t residency_required = sizeof(*residency) +
      static_cast<uint64_t>(residency->capacity) *
          sizeof(opennpux_npu_weight_residency_record);
  if (residency->magic != OPENNPUX_NPU_WEIGHT_RESIDENCY_MAGIC ||
      residency->version != OPENNPUX_NPU_WEIGHT_RESIDENCY_VERSION ||
      residency->header_size != sizeof(*residency) ||
      residency->record_size != sizeof(opennpux_npu_weight_residency_record) ||
      slot >= residency->capacity ||
      residency_required > residency_binding->byte_size) {
    return Fail("cannot publish residency record");
  }
  auto* residency_records = reinterpret_cast<
      opennpux_npu_weight_residency_record*>(residency + 1);
  auto* residency_record = &residency_records[slot];
  const bool was_valid =
      (residency_record->flags & OPENNPUX_NPU_WEIGHT_RESIDENCY_VALID) != 0;
  residency_record->flags = 0;
  residency_record->command_id = fault->command_id;
  residency_record->role_id = fault->role_id;
  residency_record->component_id = fault->component_id;
  residency_record->shard_index = fault->shard_index;
  residency_record->expert_id = fault->expert_id;
  residency_record->range_file_offset = fault->range_file_offset;
  residency_record->range_size = fault->range_size;
  residency_record->page_file_offset = fault->file_offset;
  residency_record->cache_slot = fault->cache_slot;
  residency_record->page_size = fault->page_size;
  __atomic_thread_fence(__ATOMIC_RELEASE);
  residency_record->flags = OPENNPUX_NPU_WEIGHT_RESIDENCY_VALID;
  if (!was_valid) {
    ++residency->valid_records;
  }
  ++residency->generation;
  __atomic_thread_fence(__ATOMIC_RELEASE);
  fault->state = OPENNPUX_NPU_PAGE_FAULT_READY;
  queue->service_index += 1;
  ++serviced_;
  last_command_id_ = fault->command_id;
  have_last_command_ = true;
  transferred_ += header_.transfer_size;
  if (serviced_ <= 10 || serviced_ % 100 == 0) {
    std::fprintf(stderr,
                 "Coral sim-host page complete count=%llu command=%u "
                 "service=%u last=%u\n",
                 static_cast<unsigned long long>(serviced_),
                 record.command_id, queue->service_index,
                 (record.flags & OPENNPUX_NPU_PAGE_BUNDLE_LAST) != 0);
    std::fflush(stderr);
  }
  return 1;
}
