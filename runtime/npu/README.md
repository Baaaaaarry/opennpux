# NPU Runtime Assets

NPU-side ELF sources and test inputs belong here as the flow moves beyond
bring-up.

The host-side control utility is `runtime/host/tools/coralctl.c`. It replaces
the shell-level reset/start/status sequence with a stable command-line
interface:

```sh
coralctl info
coralctl run
```

The next runtime increment will add shared-buffer allocation and command
submission after the Coral AXI master path is connected to gem5 DMA.
