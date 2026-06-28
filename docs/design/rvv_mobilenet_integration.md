# RVV Highmem MobileNet Integration

## Scope

This increment adds a second Coral bridge without replacing the existing
`CoreMiniAxi` bridge:

- `libcoralnpu_gem5_bridge.so`: scalar/float CoreMiniAxi, 8 KiB/32 KiB TCM;
- `libcoralnpu_gem5_rvv_highmem_bridge.so`: official
  RvvCoreMiniHighmemAxi, RVV enabled, 1 MiB/1 MiB TCM.

The highmem bridge executes the official LiteRT Micro MobileNet V1 0.25 dummy
model and optimized Coral convolution/depthwise-convolution kernels. This is a
framework and full-graph integration test, not yet an accuracy benchmark with
production weights.

## Memory Contract

The firmware uses the official Coral linker layout:

| Coral address | Purpose |
| --- | --- |
| `0x00000000` | 1 MiB ITCM |
| `0x00100000` | 1 MiB DTCM |
| `0x20000000` | 4 MiB LiteRT Micro tensor arena |
| `0x20400000` | 64-byte completion mailbox |

The gem5 NPU device translates the EXTMEM range onto an 8 MiB reserved ARM
physical-memory window beginning at `0x8ff00000`. All EXTMEM traffic continues
through the existing coherent DMA port and SLC/DDR path.

The tensor arena is linked as `NOLOAD .extbss`; Linux clears the shared window
before releasing Coral reset. This avoids trying to initialize external memory
through Coral's AXI slave port, which only exposes TCM and CSR regions.

## Completion Contract

The firmware does not use HTIF semihosting. It writes a versioned mailbox after
operator registration, tensor allocation, and inference. Linux validates:

- firmware reached `MOBILENET_COMPLETE`;
- LiteRT Micro returned no allocation/invoke/output error;
- five output bytes were produced;
- DMA requests equal DMA completions and DMA errors remain zero.

This leaves the standard bridge and Phase 2-5 command tests unchanged.
