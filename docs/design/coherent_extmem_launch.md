# Coherent EXTMEM Launch Synchronization

The fast-DMA Verilated backend keeps the configured synchronization range in
bridge-local EXTMEM while firmware executes. Before reset is released, gem5
must publish the CPU-prepared request from the shared DMA window into that
local EXTMEM range.

The launch synchronization is a coherent timing DMA read, not a functional
memory read. This is required because the CPU may still own modified cache
lines when userspace rings the MMIO doorbell. A functional request issued
downstream of the CPU caches can otherwise observe stale backing memory even
when userspace correctly orders its shared-memory stores before MMIO.

Launch ordering is:

1. CPU writes the request into the shared DMA window.
2. Runtime executes a release fence, then issues the reset/start MMIO
   sequence.
3. `NPUDevice` keeps the Coral backend in reset and starts a coherent DMA read
   for `fastDmaSyncOffset..fastDmaSyncSize`.
4. The DMA completion callback copies the bytes into bridge-local EXTMEM.
5. `NPUDevice` releases backend reset and schedules RTL execution.
6. Runtime executes an acquire fence before consuming completion data.

The operator execution fast path remains unchanged. Only the bounded launch
copy uses timing DMA. With `NPUDevice` debug enabled, the completion message
prints the first synchronized word; Qwen device inference expects
`first_word=0x4e495751`.
