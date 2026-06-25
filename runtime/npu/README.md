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
```

`dma-test` discovers the reserved shared-memory window from the NPU shell,
writes two operands, starts the Coral DMA smoke firmware, and verifies the
coherent read/modify/write result.
