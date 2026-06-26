# NPU Runtime Assets

NPU-side ELF sources and test inputs belong here as the flow moves beyond
bring-up.

The host-side runtime API is split into:

- `runtime/host/include/opennpux/coral_runtime.h`
- `runtime/host/src/coral_runtime.c`
- `runtime/host/tools/coralctl.c`

`coralctl` is now a command-line frontend over the reusable runtime API. It
replaces the shell-level reset/start/status sequence with a stable interface:

```sh
coralctl info
coralctl run
coralctl dma-test
coralctl mem-info
coralctl mem-clear
coralctl mem-read32 0x0
coralctl mem-write32 0x0 0x2a
```

`dma-test` discovers the reserved shared-memory window from the NPU shell,
writes two operands, starts the Coral DMA smoke firmware, and verifies the
coherent read/modify/write result.

The `mem-*` commands are the shared-buffer management ABI before a kernel
driver is introduced. They discover the DT reserved-memory window via the NPU
shell CSRs and access it through the current runtime backend, with explicit
bounds and alignment checks.

The first Phase-3 increment keeps `/dev/mem` as the backend but moves all MMIO,
shared-window, run, and DMA-smoke logic into `opennpux_coral_*` APIs. The next
increment replaces this backend with `/dev/opennpux-coral` ioctls and mmap
without changing `coralctl` commands or higher-level runtime calls.
