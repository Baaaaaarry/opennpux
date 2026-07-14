# RVV Highmem MobileNet Acceptance

## Build

Run on x86 Linux:

```sh
git pull
./tools/coralnpu/build_rvv_mobilenet.sh
```

The build script applies the Coral overlay that enables the pinned
`@tflite_micro` repository, native workspace dependencies, and the pinned
`@tflm_pip_deps` code-generation environment. No separate TFLite or Python
package installation is required.

Expected artifacts:

```text
build/coralnpu/libcoralnpu_gem5_rvv_highmem_bridge.so
build/coralnpu/gem5_mobilenet.elf
```

Do not run the build with `sudo`. If artifacts from an older privileged build
made the output directory read-only, repair it once before rebuilding:

```sh
sudo chown -R "$USER:$(id -gn)" build/coralnpu
```

Update the guest `coralctl` before creating the dedicated checkpoint:

```sh
IMAGE=/home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img \
./tools/coralnpu/phase45_prepare_guest_assets.sh
./sim/gem5/apply_patchset.sh
```

If an existing checkpoint was created before the checksum-capable `coralctl`
was installed, rebuild it once with `CORAL_REBUILD_CKPT=1`. Otherwise the
checkpoint may keep the old `/tmp/coralctl`, and logs will only contain
`mobilenet_output=<first five values>` without `mobilenet_output_checksum`.

## Create Dedicated Checkpoint

The MobileNet configuration reserves an 8 MiB coherent window, so it uses a
separate checkpoint from the 4 KiB command tests:

```sh
CORAL_KERNEL_IMAGE=/home/barry/code/opennpux/build/kernel/vmlinux-4.19.325-opennpux \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
./tools/coralnpu/run_rvv_mobilenet_test.sh
```

The first invocation creates `m5out/coralnpu_mobilenet_ckpt`. Run the same
command again to restore it and execute MobileNet.

Select the operator execution path explicitly:

```sh
# Fast end-to-end functional mode. Coral RTL performs firmware control,
# allocation, TFLM operator scheduling, staging, doorbell, completion, and
# mailbox handling; host reference kernels execute Conv2D + DepthwiseConv2D.
CORAL_OPERATOR_MODE=hybrid ./tools/coralnpu/run_rvv_mobilenet_test.sh

# Full cycle-evaluated RVV RTL mode for hardware performance studies.
CORAL_OPERATOR_MODE=rtl ./tools/coralnpu/run_rvv_mobilenet_test.sh
```

`rtl` is the default for backward compatibility. Both modes use the same gem5
binary, bridge, firmware, Linux driver, mailbox ABI, and boot checkpoint.
Changing modes does not rebuild the checkpoint. Hybrid `mobilenet_npu_cycles`
reports the sum of per-operator modeled accelerator cycles. Host-kernel
nanoseconds remain in the operator descriptor and bridge log and must not be
compared with RTL cycle counts.

Hybrid latency is controlled without rebuilding firmware:

```sh
CORAL_OPERATOR_MODE=hybrid \
CORAL_HYBRID_OPS_PER_CYCLE=64 \
CORAL_HYBRID_BYTES_PER_CYCLE=32 \
CORAL_HYBRID_FIXED_CYCLES=128 \
./tools/coralnpu/run_rvv_mobilenet_test.sh
```

The default model is conservative and deterministic:
`ops_per_cycle=1`, `bytes_per_cycle=16`, `fixed_cycles=0`.
Both modes now share the versioned descriptor in
`hw_sim/gem5_bridge/coral_operator.h`. The partial-MobileNet command is the
first graph opcode; Conv2D, DepthwiseConv2D, and MatMul have reserved per-op
opcodes and use the same tensor/quantization/status layout.

To run both modes and compare the complete output tensor checksum
automatically:

```sh
./tools/coralnpu/run_mobilenet_mode_matrix.sh
```

The script writes:

```text
simout/mobilenet-matrix/mobilenet-hybrid.log
simout/mobilenet-matrix/mobilenet-rtl.log
simout/mobilenet-matrix/mobilenet-compare.log
simout/mobilenet-matrix/mobilenet-report.md
```

Each mode log contains gem5 host output plus the matching guest
`system.terminal` appended immediately after that mode exits. This prevents the
second mode from overwriting the terminal output before checksum comparison.

To compare logs from previous runs:

```sh
./tools/coralnpu/compare_mobilenet_results.sh \
  simout/mobilenet-matrix/mobilenet-hybrid.log \
  simout/mobilenet-matrix/mobilenet-rtl.log
```

Expected comparison output ends with `mobilenet_compare=PASS`. With current
guest tools, the comparison checks `mobilenet_test=PASS`, complete output
tensor checksum equality, and output byte-count equality; it also prints both
NPU cycle counts for reporting. If old logs lack `mobilenet_output_checksum`,
the tool explicitly downgrades to `mobilenet_compare_scope=sample-output` and
compares the first five output values only.

`mobilenet-report.md` is the RTL performance summary artifact. It records
hybrid and RTL cycles, operation count, memory traffic, cycles per operation,
bytes per cycle, and the RTL/hybrid cycle ratio. Use this file for regressions
and design reviews instead of copying terminal fragments manually.

For RTL diagnosis, run the matrix with debug enabled:

```sh
CORAL_MOBILENET_DEBUG=1 ./tools/coralnpu/run_mobilenet_mode_matrix.sh
```

This additionally writes per-mode debug logs and summaries:

```text
simout/mobilenet-matrix/mobilenet-hybrid.debug
simout/mobilenet-matrix/mobilenet-hybrid.summary
simout/mobilenet-matrix/mobilenet-rtl.debug
simout/mobilenet-matrix/mobilenet-rtl.summary
```

The summary reports the last firmware marker, fault CSRs if present, latest
RTL heartbeat, EXTMEM/DMA samples, and per-phase cycle/access deltas.

For a functional run with NPU progress tracing, use:

```sh
CORAL_MOBILENET_DEBUG=1 \
CORAL_RTL_CYCLES_PER_EVENT=1000 \
CORAL_KERNEL_IMAGE=/home/barry/code/opennpux/build/kernel/vmlinux-4.19.325-opennpux \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
./tools/coralnpu/run_rvv_mobilenet_test.sh

tail -F simout/coral-mobilenet.debug
```

To summarize an existing debug log:

```sh
./tools/coralnpu/summarize_mobilenet_progress.sh simout/coral-mobilenet.debug
```

The wrapper prints the effective gem5 options and RTL cycle batch before it
starts gem5. Neither option requires rebuilding the boot checkpoint.

The `NPUDevice` trace reports entry and return for the first 10 RTL step
batches, event 100, and then every 1000 gem5 NPU events. Increasing heartbeat
counters prove that execution continues before the firmware issues its first
external-memory request. A step batch start without its matching heartbeat
localizes a stall inside the bridge; DMA wait/start/complete messages localize
progress through the memory path.

The bridge latches the Coral `io_fault` signal across step batches. A transient
illegal-instruction or memory fault therefore terminates with `Coral RTL raised
a core fault` instead of being hidden between Linux status polls.

MobileNet firmware also writes bridge-local progress markers that do not use
shared DDR:

| Marker | Stage |
| --- | --- |
| `0x4d4e0001` | `_start` entry reached |
| `0x4d4e0002` | scalar setup complete; BSS clear begins |
| `0x4d4e0003` | BSS clear completed |
| `0x4d4e0100` | CRT cleared BSS and entered constructors |
| `0x4d4e0200` | `main()` entered |
| `0x4d4e0201` | initial mailbox writes completed |
| `0x4d4e0300` / `0x4d4e0301` | tensor allocation begin/end |
| `0x4d4e0400` | input tensor initialized |
| `0x4d4e0500` / `0x4d4e0501` | inference begin/end |
| `0x4d4e0600` / `0x4d4e0601` | optimized Conv2D begin/end |
| `0x4d4e0700` / `0x4d4e0701` | optimized DepthwiseConv2D begin/end |

The gem5 backend reads every ELF `PT_LOAD` segment back through the Coral AXI
slave port before boot and compares it byte-for-byte with the ELF image. A bad
ITCM/DTCM load therefore fails before the first RTL heartbeat. The MobileNet
build also prints the ELF entry, LOAD segments, and BSS boundary symbols so a
long pre-`main()` BSS clear can be distinguished from a corrupt firmware load.

The SoC-facing Coral CSR ABI remains at aperture offset `0x30000`. The RVV
highmem RTL moves its internal CSR region to `0x00200000`; the highmem bridge
translates `0x30000..0x30fff` to `0x00200000..0x00200fff`. Without this mapping
gem5 schedules RTL events while the core remains reset and clock-gated.

The MobileNet firmware reserves a 64 KiB DTCM stack instead of Coral's
128-byte default. Its TFLM resolver and `MicroInterpreter` are local objects in
`main()`; the default stack corrupts them before or during `AllocateTensors()`.

On `io_fault`, gem5 reads the Coral CSR window before terminating and reports
`mepc`, `mtval`, and `mcause`. For standard RISC-V exceptions, `mcause=2` is an
illegal instruction, `1` an instruction access fault, `5` a load access fault,
and `7` a store access fault.

MobileNet overrides the weak default exception handler and emits the original
trap CSRs before its final `ebreak`. In the host log, markers `0x4d4eff01`,
`0x4d4eff02`, and `0x4d4eff03` are followed respectively by the original
`mepc`, `mtval`, and `mcause` values. This preserves the first exception that
the default handler would otherwise overwrite with Coral usage fault 25.

MobileNet enables `--npu-fast-dma`. This mode sends each RTL AXI request as a
functional packet through the same coherent SLC-side port, preserving data and
cache visibility while bypassing timing-DMA queue and callback latency. It is
intended for end-to-end functional inference. Omit the option for cycle/timing
studies and DMA protocol acceptance.
Set `CORAL_FAST_DMA=0` when invoking the MobileNet wrapper to select timing
DMA explicitly.
Fast mode also executes multiple RTL batches within one gem5 event, including
synchronous functional DMA completions. This removes hundreds of thousands of
gem5 event-queue round trips while retaining the same Verilated RTL execution
and AXI response sequence. The SimObject default is 1024 batches per event; the
MobileNet wrapper defaults `CORAL_FAST_DMA_EVENT_BATCH=4096` for long inference
runs. Increase it to 8192 or 16384 for throughput sweeps, or reduce it when
debugging fine-grained RTL/DMA progress.
The MobileNet wrapper defaults each batch to 1000 RTL cycles, so one fast-mode
gem5 event can advance up to 4,096,000 cycles with the wrapper default.
Override this with `CORAL_RTL_CYCLES_PER_EVENT` only for debugging.
Fast DMA uses a lazy 4 KiB page cache over the shared EXTMEM window. The first
NPU access fills a page through the coherent functional port, byte/word AXI
requests then hit the local page, and dirty pages are written back before the
RTL halt becomes visible to guest software. Reset invalidates the cache.
Page fills and writebacks are split at gem5 cache-line boundaries so each
functional packet stays within one interleaved memory-channel stripe; issuing
a single 4 KiB packet would be routed to `badaddr_responder`.
The 8 MiB MobileNet window is based at `0x8f000000`, entirely below this
platform's `0x90000000` RAM limit. The earlier `0x8ff00000` base left only
1 MiB of valid RAM and mapped most of the tensor arena into the bad-address
responder. Changing the base automatically rebuilds the dedicated checkpoint
because the reserved-memory address is part of its DTB.

ABI v4 adds a bridge-local EXTMEM fast path for RVV highmem inference. With
fast DMA enabled, the bridge services the 8 MiB Coral EXTMEM aperture directly
inside each Verilator step, so scalar AXI accesses no longer return to gem5 as
individual DMA requests. Before start, gem5 copies the 4 KiB MobileNet mailbox
page into local EXTMEM; after halt it copies that page back through the coherent
functional port. Timing mode leaves local EXTMEM disabled and retains the
original DMA request/completion protocol.
The bridge prints the first ten local EXTMEM accesses and then one cumulative
sample every 100,000 transactions, including read/write counts, bytes, and the
last address. These counters demonstrate allocator activity but are not a
completion percentage because TFLM does not publish its total access count.
Every firmware marker also prints `Coral phase stats` with cumulative and
per-phase RTL cycles, host wall time, EXTMEM accesses/bytes, and a
1/2/4/8/16-byte access-width histogram. The begin/end marker pair for an
operator therefore provides its simulation rate and memory-traffic cost
without enabling high-volume per-transaction tracing.
If `CORAL_FAST_DMA_EVENT_BATCH=4096` and `8192` produce nearly identical phase
times, the bottleneck is no longer gem5 DMA or event scheduling. For example,
the MobileNet Conv2D phase can spend tens of millions of cycles inside the
Verilated RVV RTL while reporting only a handful of EXTMEM accesses. Use
`CORAL_OPERATOR_MODE=sampled` for daily model bring-up: it still boots the
Verilated firmware and exercises the driver, mailbox, EXTMEM, and operator
doorbell path, but routes the currently supported long Conv2D and
DepthwiseConv2D operators through the calibrated hybrid kernel. Use
`CORAL_OPERATOR_MODE=rtl` only for full RTL performance studies.
Summarize the current phase without following the complete trace using:

```sh
./tools/coralnpu/summarize_mobilenet_progress.sh
```

The bridge advances the independent OpenNpuX custom-MAC Verilator model only
during its three active command cycles. Earlier builds evaluated that unused
model on every Coral core cycle, roughly doubling Verilator evaluation work
during MobileNet even though the firmware never addressed the custom MAC.

The first MobileNet run may print `backend=stage-a` while Linux creates the
dedicated boot checkpoint. The wrapper automatically starts a second gem5
process after the checkpoint is saved; that process must print
`backend=verilated-coral`. Existing checkpoints skip the bootstrap process.
The wrapper resolves the validated kernel from `build/kernel/kernel.release`
and uses `/sbin/opennpux-init.sh`; it fails before starting gem5 when that
kernel artifact is missing. Explicit `CORAL_KERNEL_IMAGE` and
`CORAL_KERNEL_INIT` values still override these defaults.
It also defaults to the Phase-3 validated Ubuntu 18.04 image at
`/home/barry/wlk/gem5_arm_linux_images/ubuntu-18.04-arm64-docker.img` instead
of allowing the generic launcher to select an incompatible development image.
Use `CORAL_DISK_IMG` when the validated image is stored elsewhere.

`run_rvv_mobilenet_test.sh` passes the window size through
`CORAL_CONFIG_OPTIONS`, after the gem5 Python configuration path. Keep
`GEM5_OPTIONS` for gem5-global flags such as `--debug-flags`.

## Expected Result

```text
[coral-mobilenet-test] started
mobilenet_prepare=mailbox-only
mobilenet_run=started
status=0x00000001
mobilenet_state=0x00000003
mobilenet_error=0
mobilenet_npu_cycles=<non-zero>
mobilenet_output_checksum=0x<complete output tensor checksum>
mobilenet_output_bytes=<complete output tensor size>
mobilenet_operation_count=<non-zero>
mobilenet_bytes_read=<non-zero>
mobilenet_bytes_written=<non-zero>
mobilenet_dma_requests=<non-zero>
mobilenet_dma_completions=<same-as-requests>
mobilenet_dma_errors=0
mobilenet_output=<five signed int8 values>
mobilenet_test=PASS
[coral-mobilenet-test] PASS
```

Hybrid mode additionally prints:

```text
[coral-mobilenet] operator mode: hybrid
Coral hybrid latency model ops_per_cycle=<...> bytes_per_cycle=<...> fixed_cycles=<...>
Coral hybrid operator complete opcode=2 host_ns=<non-zero> operations=<non-zero> modeled_cycles=<non-zero> bytes=<non-zero>
Coral hybrid operator complete opcode=3 host_ns=<non-zero> operations=<non-zero> modeled_cycles=<non-zero> bytes=<non-zero>
```

`mobilenet_run=started` is flushed immediately before reset is released. If
RTL inference is slow, this distinguishes it from guest-side shared-memory
initialization.

This acceptance runs Coral's upstream
`mobilenet_v1_025_partial_layers.tflite` graph. It contains the initial Conv2D
and optimized DepthwiseConv2D path used by Coral's own RTL test, and proves that
ARM Linux dispatches a LiteRT Micro graph through the driver and coherent
memory path to the official RVV highmem RTL.
The firmware initializes the input tensor using its runtime `input->bytes`
metadata because this partial graph does not have the full model's fixed
`224 * 224 * 3` input shape.
For this partial graph, the 768 KiB tensor arena is placed in the highmem
core's 1 MiB DTCM scratchpad. The measured graph working set is below 512 KiB;
the remaining DTCM accommodates firmware globals and the 64 KiB stack. The
mailbox remains in coherent EXTMEM, so ARM/Linux command and result transport
is unchanged. This removes hundreds of thousands of 1-byte/4-byte AXI
transactions from the Conv2D inner loop while preserving execution of the
actual Coral RVV RTL and optimized kernels. Larger graphs that do not fit DTCM
must use tiled EXTMEM staging or the planned hybrid operator backend rather
than silently increasing this arena.

The full `mobilenet_v1_0.25_224_int8_dummy.tflite` graph is not the Phase-4/5
platform acceptance target. On cycle-accurate RTL it can spend more than 100
million cycles in `AllocateTensors()` before inference begins. Full-graph
execution remains a separate operator-coverage and memory-planning milestone;
heartbeat counts alone are RTL cycles, not DMA counts or completion progress.
