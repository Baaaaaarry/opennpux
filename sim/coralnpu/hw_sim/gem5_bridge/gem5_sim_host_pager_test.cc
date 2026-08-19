#include "hw_sim/gem5_bridge/gem5_sim_host_pager.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <unistd.h>

#include "hw_sim/gem5_bridge/npu_submission.h"
#include "hw_sim/gem5_bridge/npu_weight_queue.h"
#include "hw_sim/gem5_bridge/npu_weight_residency.h"

namespace {

constexpr uint64_t kBase = UINT64_C(0x20000000);

template <typename T>
T* At(std::vector<uint8_t>* memory, size_t offset) {
  return reinterpret_cast<T*>(memory->data() + offset);
}

}  // namespace

int main() {
  char path[] = "/tmp/opennpux-page-bundle-XXXXXX";
  const int fd = mkstemp(path);
  assert(fd >= 0);
  FILE* file = fdopen(fd, "wb");
  assert(file != nullptr);
  opennpux_npu_page_bundle_header bundle = {};
  bundle.magic = OPENNPUX_NPU_PAGE_BUNDLE_MAGIC;
  bundle.version = OPENNPUX_NPU_PAGE_BUNDLE_VERSION;
  bundle.header_size = sizeof(bundle);
  bundle.record_size = sizeof(opennpux_npu_page_bundle_record);
  bundle.transfer_size = 4096;
  bundle.record_count = 2;
  bundle.command_count = 8;
  bundle.payload_bytes = 8192;
  opennpux_npu_page_bundle_record skipped = {};
  skipped.command_id = 6;
  skipped.page_size = 4096;
  opennpux_npu_page_bundle_record record = {};
  record.command_id = 7;
  record.shard_index = 2;
  record.role_id = 3;
  record.component_id = 4;
  record.expert_id = 5;
  record.file_offset = 0x10000;
  record.range_file_offset = 0x10100;
  record.range_size = 0x2000;
  record.page_size = 4096;
  record.flags = OPENNPUX_NPU_PAGE_BUNDLE_LAST;
  std::vector<uint8_t> page(4096, 0xa5);
  std::vector<uint8_t> skipped_page(4096, 0x5a);
  assert(fwrite(&bundle, 1, sizeof(bundle), file) == sizeof(bundle));
  assert(fwrite(&skipped, 1, sizeof(skipped), file) == sizeof(skipped));
  assert(fwrite(skipped_page.data(), 1, skipped_page.size(), file) ==
         skipped_page.size());
  assert(fwrite(&record, 1, sizeof(record), file) == sizeof(record));
  assert(fwrite(page.data(), 1, page.size(), file) == page.size());
  assert(fclose(file) == 0);
  assert(setenv("CORAL_SIM_HOST_PAGE_BUNDLE", path, 1) == 0);

  std::vector<uint8_t> extmem(0x20000, 0);
  auto* invocation = At<opennpux_npu_invocation_header>(&extmem, 0);
  invocation->magic = OPENNPUX_NPU_INVOCATION_MAGIC;
  invocation->version = OPENNPUX_NPU_INVOCATION_VERSION;
  invocation->header_size = sizeof(*invocation);
  invocation->binding_offset = 0x100;
  invocation->binding_count = 3;
  auto* bindings = At<opennpux_npu_tensor_binding>(&extmem, 0x100);
  bindings[0].flags = OPENNPUX_NPU_BIND_PAGE_QUEUE;
  bindings[0].device_address = kBase + 0x1000;
  bindings[0].byte_size = 0x1000;
  bindings[1].flags = OPENNPUX_NPU_BIND_PAGE_RESIDENCY;
  bindings[1].device_address = kBase + 0x2000;
  bindings[1].byte_size = 0x1000;
  bindings[2].flags = OPENNPUX_NPU_BIND_PAGE_CACHE;
  bindings[2].device_address = kBase + 0x4000;
  bindings[2].byte_size = 0x10000;

  auto* queue = At<opennpux_npu_weight_queue_header>(&extmem, 0x1000);
  queue->magic = OPENNPUX_NPU_WEIGHT_QUEUE_MAGIC;
  queue->version = OPENNPUX_NPU_WEIGHT_QUEUE_VERSION;
  queue->header_size = sizeof(*queue);
  queue->entry_size = sizeof(opennpux_npu_page_fault);
  queue->capacity = 4;
  queue->producer_index = 1;
  auto* fault = reinterpret_cast<opennpux_npu_page_fault*>(queue + 1);
  fault->magic = OPENNPUX_NPU_PAGE_FAULT_MAGIC;
  fault->version = OPENNPUX_NPU_PAGE_FAULT_VERSION;
  fault->struct_size = sizeof(*fault);
  fault->state = OPENNPUX_NPU_PAGE_FAULT_PENDING;
  fault->command_id = 7;
  fault->page_size = 4096;
  auto* residency = At<opennpux_npu_weight_residency_header>(&extmem, 0x2000);
  residency->magic = OPENNPUX_NPU_WEIGHT_RESIDENCY_MAGIC;
  residency->version = OPENNPUX_NPU_WEIGHT_RESIDENCY_VERSION;
  residency->header_size = sizeof(*residency);
  residency->record_size = sizeof(opennpux_npu_weight_residency_record);
  residency->capacity = 16;

  Gem5SimHostPager pager;
  assert(pager.enabled());
  assert(pager.Service(&extmem) == 1);
  assert(pager.serviced() == 1);
  assert(queue->service_index == 1);
  assert(fault->state == OPENNPUX_NPU_PAGE_FAULT_READY);
  assert(fault->cache_slot == 7);
  assert(fault->flags == OPENNPUX_NPU_PAGE_FAULT_LAST);
  assert(extmem[0x4000 + 7 * 4096] == 0xa5);
  auto* records = reinterpret_cast<opennpux_npu_weight_residency_record*>(
      residency + 1);
  assert(residency->generation == 1);
  assert(residency->valid_records == 1);
  assert(records[7].command_id == 7);
  assert((records[7].flags & OPENNPUX_NPU_WEIGHT_RESIDENCY_VALID) != 0);
  assert(unlink(path) == 0);
  puts("gem5_sim_host_pager=PASS");
  return 0;
}
