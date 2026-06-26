# NPU Runtime Assets

NPU-side ELF sources and test inputs belong here as the flow moves beyond
bring-up.

The host-side control utility is `runtime/host/tools/coralctl.c`. It replaces
the shell-level reset/start/status sequence with a stable command-line
interface:

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

The `mem-*` commands are the Phase-2 shared-buffer management ABI before a
kernel driver is introduced. They discover the DT reserved-memory window via
the NPU shell CSRs and access it through `/dev/mem`, with explicit bounds and
alignment checks.
