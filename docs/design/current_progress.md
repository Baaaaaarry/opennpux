# OpenNPUX Current Progress

## Target

The target is an ARM heterogeneous SoC simulated by gem5:

- D9200/D9300 CPUs boot Linux and run the inference framework.
- Coral NPU executes official firmware and custom accelerator RTL.
- CPU and NPU exchange commands and tensors through coherent shared memory.
- System software remains usable while the Coral RTL is extended.

## Completed

### Repository and build structure

- `thirdparty/gem5` and `thirdparty/coralnpu` remain upstream submodules.
- Local additions and modifications are mirrored under `sim/gem5` and
  `sim/coralnpu`.
- Overlay scripts merge local sources into the submodules before compilation.
- The supported build host was moved to x86-64 Linux.
- Bazel caches, distdir, output root, and transitioned outputs are handled
  locally and repeatably.

### gem5 SoC bring-up

- D9200 and D9300 full-system configurations were preserved.
- The Coral NPU is instantiated as a `DmaVirtDevice`.
- Its MMIO aperture is visible at `0x1d000000`.
- Its DMA port is connected behind the SLC when that path exists.
- Linux boot checkpoints are reusable across source and guest-script updates.
- Guest MMIO access works through the minimal `mmio32` utility.

### Stage-A model

- Implemented reset, PC start, status, ITCM, and DTCM behavior.
- Verified Linux-to-NPU MMIO access and run-to-halt control.
- Used Stage A to isolate device-tree, address-map, image, and checkpoint
  problems before introducing RTL.

### Official Coral RTL bridge

- Built the official non-SystemC Verilated `CoreMiniAxi` model.
- Added a versioned C ABI shared library between Coral and gem5.
- Avoided linking a second SystemC runtime into gem5.
- Added RTL clock scheduling on the gem5 event queue.
- Added ELF `PT_LOAD` loading into Coral TCM.
- Added backend and firmware-entry discovery registers.
- Added minimal Coral firmware that returns through the official CRT and
  reaches the real `mpause` halted state.

### Verified result

The x86 full-system run now verifies:

```text
backend=verilated-coral
status(after)=0x00000001
PASS: Coral NPU MMIO is reachable and verilated-coral execution halted
```

This proves that Linux, gem5, the runtime-loaded bridge, official Coral RTL,
the Coral toolchain output, and RTL execution are connected end to end.

The coherent DMA smoke is also complete:

```text
dma_requests=4
dma_completions=4
dma_result=42
dma_magic=0x4e505544
dma_test=PASS
```

## Problems Resolved

- Missing and incompatible macOS Coral toolchains.
- Docker daemon, registry, DNS, credential-helper, and architecture issues.
- Bazel cache loss and repeated Bazel binary downloads.
- Corrupted external repositories and `rules_java` cache state.
- Bazel visibility, toolchain-transition output paths, and strict patch
  application failures.
- Duplicate gem5 overlay sources and duplicate MMIO responder ranges.
- Linux `/dev/mem` access without a usable `devmem` utility.
- Checkpoint invalidation caused by source or script timestamps.
- gem5 and Accellera SystemC teardown conflicts.
- Empty RTL TCM causing the NPU to run forever without halting.
- Valid Coral ELF entry address zero being treated as an error.

## Remaining Work

### Generic submission architecture

The Qwen tiny TCB and offline Qwen execution plan are validation fixtures, not
the final SoC contract. The platform now follows
`docs/adr/0002-generic-npu-submission-architecture.md`: an offline frontend
emits a generic NPU executable, while the CPU runtime submits every live
inference request with current tensor bindings, dynamic dimensions, persistent
state and synchronization.

The generic platform contract now includes:

- versioned generic executable, invocation, binding, command, completion and
  trace records;
- CPU-side load, bind, instantiate and submit paths;
- a firmware command processor with dependency validation and capability
  dispatch statistics;
- complete command-to-safetensors range indexes with active-expert filtering;
- a caller-owned LRU weight cache and resumable shared fault queue;
- operator capability IDs shared by RTL, RVV, hybrid and sampled backends.

The first contiguous command-processor path is now implemented and validated
with the 40-layer Qwen3.5 executable. Its 524 commands plus fixed-size numerical
operator records occupy about 93KiB and are staged in the standard 8MiB shared
window. Each command carries a resolved parameter symbol, numerical dimensions
and quantization fields, runtime batch/sequence/KV/active-expert tuple, and
logical weight/state/scratch binding IDs. Firmware validates all relocations
and reports the relocated-command count and a checksum over symbols and full
parameter records in the completion record. The scheduler validates 523
dependency edges across the complete 524-command token path, dispatches all generic
capability classes, and emits per-op command/operation/byte estimates through
a versioned trace buffer. The range compiler maps 343 weight-bearing commands
to 124,268 real safetensors ranges with no unresolved weight command.

GB10 measurement with eight active experts enumerates 1,340,225 aligned 4KiB
requests. Coalescing at a 64KiB transfer granularity reduces this to 85,173
requests (15.74x fewer) while padded request volume rises only from roughly
5.49GB to 5.58GB (about 1.7%). The 64KiB transfer block is therefore the
selected weight-DMA control granularity; it remains independent of the future
IOMMU page size. A shared single-producer/single-service/single-retire fault
ring now supports batched PENDING/READY/ERROR page lifecycles, cache slots and
explicit queue-full backpressure. The complete real model now resolves 30,720
routed-expert projection groups with no missing or duplicate GPTQ component.
Shared-window cache placement and real command pause/service/resume are now
implemented. The queue and cache are placed after the actual invocation and
auxiliary buffers, avoiding overlap with the roughly 93KiB Qwen submission.
Page-fault ABI version 4 identifies each response by command, tensor role,
GPTQ component, expert and exact source range. Firmware can therefore rebuild
generic quantized operands from streamed pages instead of treating a command's
first sampled word as its weight. Complete per-command range streaming is the
input contract for tiled MATMUL and EXPERT execution. A model-independent
streamed GPTQ MatMul kernel now removes the contiguous-weight requirement: it
fetches bounded, output-channel-aligned `qweight`, `qzeros`, and `scales` tiles
through a caller reader, optionally fetches `g_idx`, and delegates each tile to
the same exact float32 accumulation kernel used by the projection acceptance
test. Unit tests require tiled and contiguous outputs to match element for
element. The generic executable adapter accepts the compiler's numerical
parameter record directly; the remaining integration step is a reader backed
by the shared page cache.

The first model-independent GPTQ data-plane kernel is now implemented for
packed int4 MatMul. It consumes AutoGPTQ `qweight`, `qzeros`, per-group
`scales`, and optional `g_idx`, supports FP16, BF16 and FP32 scale storage,
supports batched input rows, and reports
operations, bytes and modeled cycles. The generic operator adapter validates
numerical command parameters and all operand buffer extents before dispatching
that kernel. It is independent of Qwen tensor names and is linked into both
Coral bridge variants, establishing the reusable command-to-kernel boundary
for hybrid and sampled execution. This fixes the dequantization contract
before wiring paged command ranges into MATMUL, ROUTER, EXPERT and LM-head
dispatch. Both Coral bridge variants compile the kernel and deterministic
grouped-quantization unit coverage fixes its numerical semantics. A dedicated
firmware smoke image submits the GPTQ operator through the CUSTOM_0 instruction
and two-level command decoder, so GB10 validation covers fetch, TDMA, tensor
execution, writeback and completion rather than calling the host kernel alone.
Because CUSTOM_0 decode is carried by the RVV highmem core integration, this
smoke must use `libcoralnpu_gem5_rvv_highmem_bridge.so`; the dedicated
`build_gptq_matmul_smoke.sh` and `run_gptq_matmul_smoke.sh` scripts enforce that
pairing and reject a run with no opcode-10 command submission.
The CUSTOM_0 GPTQ request ABI is now version 2 and carries the scale storage
type explicitly. Its firmware-to-bridge path validates buffer extents with the
actual FP16/BF16/FP32 element width before dispatch, closing the last FP32-only
assumption on the end-to-end custom-instruction path.
The first real-weight projection harness now materializes one exact
`qweight/qzeros/scales/g_idx` set plus deterministic input and output buffers
as an EXTMEM image. `coralctl mem-load` stages that image into the shared DMA
window, dedicated firmware consumes the externally prepared request, and the
CUSTOM_0 path reports state, error, operations and output checksum. This is the
first numerical step beyond metadata probes and sampled weight words.
That projection is now gated numerically rather than by a non-zero checksum. A
model-independent host reference in `runtime/host/src/npu_gptq_reference.c`
decodes the same staged EXTMEM image, validates every operand extent against
the request record, and recomputes the projection in float32 in the bridge
kernel's accumulation order. The full-system runner computes the expectation
before boot and requires the device to report exactly the same output checksum
and operation count, so an addressing, packing, group-index or scale-dtype
regression can no longer pass. The reference must be compiled with
`-ffp-contract=off`; a fused multiply-add changes the rounding and breaks the
exact comparison on FMA-baseline targets.
Request staging has also moved from the offline Python harness into the CPU
runtime. `runtime/host/src/npu_gptq_request.c` resolves a command's exact
components through the binary range index, checks every component against the
caller-supplied projection shape, lays the operands out on 64-byte boundaries,
binds live input and output buffers, and writes the CUSTOM_0 request record.
Projection geometry stays a compiler-frontend input rather than a tensor-name
rule, so the module carries no Qwen naming. A host test stages the same
projection the offline materializer emits and requires the two images to be
byte-identical, which keeps the offline harness and the runtime
interchangeable while guest-side submission is wired up.

Generic executables now carry a model-independent, fixed-size numerical
parameter record for every command. The record preserves static feature sizes,
attention geometry, expert intermediate width, GPTQ bits/group size and scale
data type while runtime rows remain dynamic. Safetensors dtype metadata is
preserved in each binary weight-range record and loaded with every component,
so real FP16 GPTQ scales are no longer interpreted as FP32. Instantiation copies
these records into the TCB
parameter area, relocates each command to its record, and protects the complete
payload with bounds checks and the invocation checksum. The NPU command
processor rejects missing, truncated, mismatched-opcode or stale parameter
records before dependency scheduling. This closes the command-contract gap
between the existing 524-command Qwen3.5 schedule and the GPTQ tensor engine.
The next increment stages one real projection's exact components and live
input/output bindings into the MATMUL request.

The generic-to-XOpenNPUX lowering boundary now also supports bounded mixed
command batches. Primitive commands emit one XGraph record, while GPTQ MatMul
commands expand atomically into their complete two-dimensional
`TDEQUANT/TMMA/TADD` sequence. Every emitted record receives a batch-local,
dense command ID and an origin entry identifying the source generic command.
The lowerer stops before a composite command that does not fit, so the runtime
can stream a large executable through fixed-capacity XGraph buffers without
splitting an operator or introducing model-specific submission rules.

The full-system command-flow test now uses that same boundary.
Guest runtime code materializes generic operator requests and lowers them into
XOpenNPUX commands immediately before submission; the previous hand-authored
XGraph command array is gone. Existing per-operator checksums and numerical
goldens therefore validate the compiler/runtime lowering path as well as the
firmware and C++ functional engines.

The command-flow test also carries one independent GPTQ projection, a generic
COMBINE request, and a KV-cache DMA update. Its twelve logical requests lower
to fifteen XOpenNPUX records: GPTQ MatMul expands to `TDEQUANT` plus two
row-wise `TMMA` commands, COMBINE is represented by `TADD`, and DMA expands to
two `TDMA` records for the Key and Value state planes. The Guest validates all
three outputs against exact FP32 goldens in addition to the original
nine-operator chain. Graph headers, completion checks and origin tracking use
the emitted command count rather than assuming one instruction per request.

Qwen3.5 lowering, paged GPTQ weights, attention state and MoE routing are the
first workload adapter on this architecture. Future model families must not
require changes to the queue or driver ABI.

### RTL bridge baseline

- The single-outstanding asynchronous AXI-to-gem5 DMA transport is verified.
- A DT reserved-memory buffer maps Coral EXTMEM onto SoC physical memory.
- DMA smoke firmware and `coralctl dma-test` verify coherent shared-memory
  reads and writes.
- gem5-specific asynchronous AXI handling is isolated from the official Coral
  wrapper and primitives.
- A signal-level adapter test covers independent write channels, held-valid
  request suppression, and delayed response handshakes without Linux boot.
- Deterministic randomized regressions exercise 64 reads and 64 writes with
  channel skew and response backpressure.
- Unsupported DMA request shapes return AXI `SLVERR` locally instead of
  reaching gem5 as zero-length DMA operations.
- INCR read bursts up to 256 beats and 4096 bytes use one coherent gem5 DMA
  and return ordered AXI response beats through `RLAST`.
- INCR write bursts up to 256 beats and 4096 bytes accept independent `AW` and
  `W` arrival, cache write beats through `WLAST`, validate strobes and burst
  boundaries, and submit one coherent gem5 DMA write.
- DMA requests outside the shared EXTMEM window are now rejected with AXI
  `SLVERR` and counted through `dma_errors` instead of terminating gem5.
- Checkpointing is guarded by gem5 drain/serialize hooks: quiescent NPU states
  can be checkpointed, while in-flight RTL/DMA state is rejected.
- `coralctl` now provides bounded shared-window management commands
  (`mem-info`, `mem-clear`, `mem-read32`, and `mem-write32`) over the same
  mapper used by `dma-test`.
- AXI `AR`, `AW`, and `W` channels are backpressured while one request is
  outstanding, making the single-outstanding RTL bridge contract explicit.
- RTL bridge acceptance is documented in `docs/runbooks/rtl_bridge_acceptance.md`.

### Driver runtime baseline

- Replace shell control with a stable userspace runtime and then a Linux driver.
- Allocate and pin shared input, output, command, and completion buffers.
- Add interrupts or an event-driven completion path.
- Define cache maintenance rules for non-coherent configurations.
- Add multiple outstanding AXI IDs if firmware/runtime profiling shows the
  single-outstanding bridge is a bottleneck.

The first driver-runtime increment has split `coralctl` into a reusable host runtime
API (`runtime/host/include/opennpux/coral_runtime.h` and
`runtime/host/src/coral_runtime.c`) plus a CLI frontend. The runtime still uses
the `/dev/mem` bring-up backend so existing checkpoint-based tests continue to
work. The next increment is to replace that backend with a minimal Linux
character device while keeping the userspace API stable.

The second driver-runtime increment has added the shared user/kernel UAPI
(`runtime/host/include/opennpux/coral_uapi.h`), runtime transport selection,
and a minimal platform-driver scaffold under `runtime/kernel`. The runtime now
auto-selects `/dev/opennpux-coral` when available and falls back to `/dev/mem`
for the existing checkpoint smoke tests.

The third driver-runtime increment completes the device boundary: the driver exposes
only the DT-reserved shared DMA window through non-cached mmap, negotiates a
versioned capability ABI, accepts asynchronous START, wakes userspace through
poll using a kernel completion worker, and supports RESET after timeout. The
driver-only DMA acceptance covers control ioctl, shared mmap, completion, and
coherent RTL DMA without opening `/dev/mem`.

### Command runtime baseline

- Load a real model and tensors from Linux.
- Integrate a narrow inference runtime API.
- Measure CPU scheduling, memory traffic, NPU execution, and end-to-end latency.

The first command-runtime increment defines a 64-byte versioned command descriptor and
bounded input/output tensor regions in the shared window. `coralctl vector-add`
now submits a multi-buffer command through the driver runtime, while
`gem5_command_smoke.elf` parses the descriptor on the Coral core and reports
completion. This is the reusable end-to-end contract for subsequent DS4
operators.

The completed platform increment adds a validated `.npxm` model container with
real input tensors, sequential multi-command dispatch, output verification, and
model-level simulated-time reporting. The sample graph mixes official Coral
software execution and custom RTL execution without DS4-specific assumptions.

### Custom RTL baseline

- Add the custom RTL accelerator inside the Coral source hierarchy.
- Add instructions or command descriptors, compiler/runtime support, and RTL
  verification.
- Compare official and custom execution using the same Linux workload.

Custom RTL baseline now includes a synthesizable three-cycle custom MAC, standalone
Verilator regression, a bounded AXI register aperture in the Coral bridge, a
custom command opcode, firmware support, cycle accounting, and Linux A/B tests.
The official `CoreMiniAxi`, gem5 device, kernel driver, and userspace transport
remain unchanged across software and custom execution.

### RVV Highmem MobileNet

The next platform configuration is implemented as a separate bridge around the
official `RvvCoreMiniHighmemAxi`: RVV is enabled, ITCM/DTCM are 1 MiB each, and
an 8 MiB coherent EXTMEM window carries the LiteRT Micro tensor arena and a
versioned completion mailbox. The initial firmware runs the upstream full
MobileNet V1 0.25 dummy graph without HTIF semihosting. x86 full-system
acceptance remains required after building the new bridge and firmware.

The RVV highmem path has since been validated far enough to distinguish system
bottlenecks from RTL execution cost:

- Fast DMA and bridge-local EXTMEM remove the earlier real-scenario DMA
  bottleneck; MobileNet now reports only mailbox/local EXTMEM synchronization
  traffic during long operator phases.
- `CORAL_FAST_DMA_EVENT_BATCH=4096` and `8192` produce nearly identical
  MobileNet phase times, proving the remaining long runtime is inside the
  Verilated RVV operator itself rather than gem5 event scheduling.
- `CORAL_OPERATOR_MODE=sampled` was added for daily bring-up: Linux, driver,
  firmware, mailbox, EXTMEM, and operator doorbell still run through the RTL
  path, while supported long operators use hybrid kernels to avoid multi-hour
  full-RTL waits.
- MobileNet firmware now targets the full upstream
  `mobilenet_v1_0.25_224_int8_dummy.tflite` graph and registers the complete
  MobileNet op set: Conv2D, DepthwiseConv2D, Reshape, AveragePool2D, Softmax,
  StridedSlice, Pad, Mean, Shape, and Pack.
- Sampled mode can force selected compute operators back onto real Coral RVV
  RTL using `CORAL_SAMPLED_RTL_OPS` (`conv`, `depthwise`, `matmul`, `fc`,
  `add`, `softmax`, `layernorm`, `all`, `none`, or a hex mask).
- Full RTL mode remains available for performance studies and has matched
  hybrid output checksum on the previously validated MobileNet workload.
- The canonical Coral gem5 ABI header is synchronized at ABI v6 across
  `rtl/wrappers`, `sim/coralnpu`, and `sim/gem5`.

### Transformer Operator Bring-up

The hybrid/sampled operator library now includes first-pass functional kernels
for Transformer-class graphs:

| Operator | Current Support | Notes |
|---|---|---|
| MatMul | int8 x int8 -> int8 | Basic 2-D matrix multiply, symmetric/simple quantization |
| FullyConnected | int8 input/weights, optional int32 bias | Maps MLP projection layers to the same descriptor path |
| Add | int8 elementwise | Same-shape residual add; broadcast and full quant rescale still pending |
| Softmax | float32 and int8 | Functional reference for attention bring-up |
| LayerNorm | float32 input/scale/bias/output | Uses descriptor `reserved[0]` as optional epsilon bits |
| Conv2D/DepthwiseConv2D | int8 | Existing MobileNet coverage retained |

These kernels are host-side hybrid functional models, not final custom RTL.
They unblock end-to-end graph partitioning and correctness validation for DS4
and other Transformer-like models while custom RTL units and bit-exact
quantized TFLM semantics are developed.

## Immediate Development Order

1. Run the driver runtime acceptance suite on x86 Linux.
2. Build and load `/dev/opennpux-coral` in the guest image.
3. Pass the driver-only shared-window and DMA system test.
4. Extend the generic model/operator registry as new workloads require.


## 当前mobilenet base基线版本进展
| 能力 | RTL 模式 | Hybrid 模式 |
|---|---|---|
| ARM Linux/驱动/runtime | 已跑通 | 已跑通 |
| Coral 固件控制流 | RTL 执行 | RTL 执行 |
| Partial MobileNet | 已完成 | 已完成 |
| Conv/Depthwise | Coral RVV RTL kernel | 主机 TFLM reference |
| Tensor arena | DTCM | RTL 分配，算子由 host 执行 |
| 性能含义 | 可统计 RTL cycles | 只代表功能执行速度 |
| 数值正确性 | 已有 checksum 对比 | 已有 checksum 对比 |
| 完整 MobileNet | full RTL 可运行但极慢 | sampled/hybrid 路径作为验收主路径 |
| Transformer 基础算子 | 待 RTL 化 | MatMul/FC/Add/Softmax/LayerNorm 已有 bring-up kernel |

## 当前base调通后待开发项
### RTL 模式待开发
完整 MobileNet 的算子覆盖：
Reshape
AveragePool2D
Softmax
StridedSlice
Pad
Mean
Shape
Pack

完整模型内存规划：
DTCM scratchpad 分配。
EXTMEM tensor tiling。
权重预取和双缓冲。
避免重新退化为大量 1-byte AXI 请求。

RTL 访存优化：
AXI burst。
Read-ahead。
Write combining。
Cache-line buffer。
权重和 activation DMA 批量搬运。

性能计数器：
每算子 cycles。
MAC/RVV 利用率。
pipeline stall。
DTCM/EXTMEM 请求数。
AXI burst 长度和有效带宽。

中断路径：
NPU completion IRQ。
GIC 接入。
Linux 驱动由轮询改为中断等待。

自定义 RTL 单元：
从当前独立 Custom MAC 扩展为可被 kernel 使用的加速单元。
增加 descriptor、DMA、状态和异常处理。
修改 Coral kernel 将目标计算下发给自定义单元。

### Hybrid 模式待开发
hybrid模式执行流：
'''
RTL 写 doorbell
gem5 暂停当前事件
x86 host 执行 TFLM
结果写入 mailbox
callback 返回
RTL 继续
从 partial graph 扩展到完整 MobileNet。
'''

从当前“固定 partial graph offload”升级为通用 operator descriptor：
opcode
tensor 地址和尺寸
shape/stride/padding
quantization multiplier/shift
zero point
activation clamp

每算子独立 offload：
Conv2D
DepthwiseConv2D
Pooling
Softmax
Elementwise
MatMul

支持 fallback：
Hybrid kernel 支持时由 host 执行。
不支持的 shape 自动回退到 Coral RTL kernel。
日志明确记录 offload/fallback。

性能模型：
host wall time 与 simulated cycles 分离。
根据 MAC 数、带宽、启动延迟计算 modeled cycles。
Hybrid 结果不能直接使用 host 纳秒作为 NPU cycles。

并发与一致性：
operator staging buffer。
DTCM/EXTMEM 批量同步。
ARM 与 NPU 并发访问时的 ownership 和 fence。
多 command queue。

公共部分
两种模式应共享：
模型解析和转换工具。
Operator ABI 和 descriptor。
Tensor layout、shape 和量化参数定义。
Linux 驱动和 /dev/opennpux-coral。
用户态 runtime、coralctl 和模型提交接口。
Mailbox、command queue 和错误码。
输入数据生成和预处理。
完整输出 tensor checksum。
Golden output 和回归测试。
每算子 marker、统计格式和报告工具。
模型/固件/ABI 版本校验。
checkpoint 和启动脚本。
算子实现关系
算子语义可以共享，但代码实现不能完全共用：
Golden/reference：x86 TFLite reference kernel，作为正确性基准。
Hybrid：主机 C++ functional kernel，应与 golden bit-exact。
RTL：Coral RVV kernel或自定义 RTL accelerator。
Fallback：Coral 标准 TFLM kernel。


时钟一致性：
当前有三套时钟系统：
1、gem5 模拟时间：ARM CPU、内存、总线和设备使用 curTick()。
2、Coral RTL 时间：Verilator VerilatedContext，每次 wrapper.Step() 推进一个 RTL cycle。
3、Host时间：x86 实际运行时间，例如 Hybrid 的 host_ns=67093044。

RTL模式配置参数：
```
--npu-rtl-tick-period=1ns
--npu-rtl-cycles-per-event=1000

cycles-per-event=1000
tick-period=1ns
每执行一个 RTL cycle，gem5 时间推进约 1 ns
```

Hybrid模式
```
RTL 写 doorbell
gem5 暂停当前事件
x86 host 执行 TFLM
结果写入 mailbox
callback 返回
RTL 继续
```

## 2026-08-02 RVV Highmem MobileNet: mobilenet_test=PASS（hybrid 模式全程跑通）

根因（RTL erratum，不修改 RTL，仅固件侧规避）：
Coral 核（RvvCoreMiniHighmemAxi）的外部存储通路在执行跨越 16 字节 line 边界的
vector load/store 时死锁：第一个 line 的总线事务完成后，第二个 line 的事务
永不发出，指令无法退休，in-order dispatch 全部堵死（无 fault/wfi/halt，AXI
master 通道全空闲，bridge 无挂起事务）。DTCM 路径不受影响；标量访问不受影响；
16 字节对齐的 vector 访问不受影响。EXTMEM（ebus）与 ITCM（ibus）路径均受影响。

观测方法：bridge wrapper 增加核内 debug 采样（io_debug_dispatch instAddr/instFire、
io_debug_rb_inst retire PC、fetch PC），watchdog dump 输出 dispatch/retire 计数与
PC 环；mrd/mwr 驱动记录最近 AXI handshake 地址/size 历史。三个卡死点逐一确认：
1. PadEval 的 PadParams vector copy（vle8.v @ OpData+4，EXTMEM，≡4 mod 16）：
   第一个 16B line read 完成（bridge last_handshake=0x203ff170 size=4），第二个
   line read 永不发出。
2. 第 15 个 conv（FC 层）staging：per-channel multiplier 数组 Memcpy 到
   4 字节对齐的 staging 槽（vse8 跨 line）：W 数据已发出（last=1）但 AW 永不发出。
3. StridedSliceEval 调 micro::GetTensorShape：dims->data（≡4 mod 16）的
   vector copy（ibus 路径）。

固件/构建侧规避（不改 RTL、不改 TFLM 源码、不 bump ABI）：
- gem5_mobilenet.cc：TracedPadInvoke 将 OpData 拷贝到 16 对齐 DTCM 静态缓冲再调用
  上游 PadEval；per-channel multiplier/shift 改用 volatile 标量 word 拷贝；
  新增 PAD_BEGIN/PAD_END=0x4d4e0800/0x4d4e0801 marker（两个 coral_mobilenet.h 同步）。
- coral_operator_client.h：AllocateStaging 将分配大小向上取整到 16，保证所有
  staging 地址 16 字节对齐。
- tools/coralnpu/build_rvv_mobilenet.sh：tflite_micro 源文件用 --per_file_copt
  关闭 GCC 全部三条 vector 代码生成路径（-fno-tree-vectorize、
  -fno-tree-slp-vectorize、-fno-tree-loop-distribute-patterns、
  -mstringop-strategy=scalar）；coralnpu 自身 RVV kernel 保持 vectorized。

结果（logs/sim/coral-mobilenet-host-20260802-013722.log +
logs/sim/m5out/system.terminal）：
interpreter.Invoke() 完成（PROGRESS_INVOKE_END=0x4d4e0501 @ RTL cycle 14,750,827），
15×conv + 13×depthwise + 4×pad 全部 hybrid 完成，m5_exit 正常退出，exit=0。
mobilenet_state=0x3 mobilenet_error=0 mobilenet_npu_cycles=40957800
mobilenet_operation_count=40775552 mobilenet_output_checksum=0x38c2a6a4
mobilenet_output=-85,-69,-78,-89,-64
mobilenet_test=PASS / [coral-mobilenet-test] PASS
//hw_sim:gem5_axi_master_drivers_test 通过；phase2_check_abi.sh /
check_mobilenet_abi.sh 通过。

后续建议（RTL 侧，超出本次范围）：定位 Lsu.scala vector slot 的跨 line
事务迭代 / DBus2Axi 在跨 line vector 访问时的第二个 line 事务生成逻辑。

## 2026-08-15 Tiny Qwen CPU prompt -> NPU inference -> CPU result

Qwen 最小端到端验收已从“CPU 数值参考 + NPU TCB 遍历”收紧为真正设备闭环：
ARM guest 仅解析模型、接收 `open npux` prompt、打包 6760 字节请求并启动 NPU；
`gem5_qwen_device_infer.elf` 在 Coral core 上通过 custom-0 `NPU_LAUNCH`
提交 `QWEN_TINY_INFER`；二级译码生成 descriptor fetch、operand read、execute、
writeback、completion 五级微命令；Hybrid NPU engine 完成 embedding、RMSNorm、
QKV/attention、residual、SwiGLU、LM head 和 TopK 数值计算；最终 logits checksum
和 next token 经 shared DMA window 回写 CPU。

关键约束：请求 ABI 不包含 expected token/checksum，初始结果字段必须为 0；CPU
只在 NPU 完成后用独立 golden 校验 `0x829e9f00 / token 7`。RVV local EXTMEM
与 SoC shared memory 使用 fast-DMA 在启动前和完成后同步起始 8KiB，相关 offset
和 size 已由 D9200/D9300 配置参数化。

本地通过：Qwen loader/golden、请求无预计算检查、Host packer -> NPU kernel
数值集成测试、严格 coralctl 编译。GB10 待执行 Verilator + full-system 验收：
`./tools/coralnpu/build_qwen_device.sh` 后运行
`./tools/coralnpu/run_qwen_e2e_test.sh`。

## 2026-08-19 Qwen35B sim-host direct paging

为消除真实权重分页在模拟 D9300 上执行 9P 文件读取、range 解析和 64KiB 逐字节
拷贝造成的数小时开销，新增宿主机直接分页模式。宿主预生成流式 `.npxb` 页包；
Coral Verilated bridge 直接监听固件 page-fault queue，将权重页写入 Local EXTMEM，
发布 residency record，并完成 `PENDING -> READY` 与 service index 更新。固件、NPU
命令处理器、producer/retire 和完成校验协议保持不变，Guest 分页保留为参考回退。

本地已通过页包生成、跳过未请求 command、cache/residency/queue 原子更新单测，
严格 native coralctl 编译及完整 model-package 回归。GB10 验收运行
`CORAL_SIM_HOST_PAGING=1 ./tools/coralnpu/run_qwen35b_real_weights_test.sh
--prompt "Explain heterogeneous computing"`，预期输出
`paging_source=sim-host-direct` 和最终 PASS。

## 2026-08-19 Qwen35B 真实 next-token 混合验收

在 sim-host direct paging 基础上新增真实数值结果通路。宿主机使用原始 Hugging Face
Qwen3.5-35B GPTQ 权重和 CPU 侧输入 prompt 执行一次真实 forward，对最后位置 logits
执行 argmax，并将 token id、token text、logits checksum、prompt checksum 和 executable
ID 写入 `.npxo` 记录。结果按 prompt 缓存在 `build/model-results/`。

Verilated bridge 只在 NPU firmware 完成全部 invocation command 后发布该结果，并严格
校验 executable ID、prompt checksum、vocabulary size、输出 ABI 和 completed command
数量。Guest `coralctl` 要求 `inference_mode=numerical`、
`inference_result_source=sim-host-numerical`，并输出真实的
`inference_next_token` 与 `inference_token_text`。

该路径用于尽快完成“CPU 输入 prompt -> NPU 控制/分页/执行完成 -> CPU 收到真实
token”的系统级验收。数值 forward 当前属于 Hybrid sim-host kernel，不代表 35B 模型
已在 Coral RTL 内逐算子完成数值执行；后续仍需以 MatMul/GPTQ、Attention、MoE、
Norm/RoPE 等设备 kernel 逐步替换 host numerical kernel。

## 2026-08-19 Qwen35B 多 token 自回归解码

真实数值结果通路由单次 forward 扩展为最多 32 token 的自回归解码，默认生成
8 token，并启用 Hugging Face KV cache。`.npxo` v2 扩展为 256 字节，新增最多 32 个
真实 token ID、实际生成 token 数和停止原因；解码文本通过 inference I/O 的 128 字节返回区
交给 Guest。`coralctl` 将 `OPENNPUX_MAX_NEW_TOKENS` 写入请求并严格校验返回数量，
bridge 仍只在 524-command NPU invocation 完成后发布数值结果。该增量验证连续生成的
系统协议与生命周期，数值计算仍属于 Hybrid sim-host kernel。

运行脚本同时从 `.npxo` v2 提取 tokenizer 产生的真实输入 token 数，并写入通用
invocation runtime shape。Decode 的 `sequence_length` 为 prompt token 数，
`kv_length` 为 prompt token 数加生成 token 数减一，替代此前固定的 `1/1`；bridge
严格校验 inference request 与数值结果的 input-token 数一致。这使调度、统计和后续
逐 token/KV page 实现建立在真实动态 shape 上。

RV32 command processor 已将 inference invocation 从单次模板遍历扩展为逐 token
状态机：第 0 步以真实 prompt sequence 执行 prefill，后续步骤以 sequence=1 执行
decode。每一步重新建立 dependency retirement 边界，并累计 command completion、
operator trace、分页请求、operations/bytes 和 modeled cycles。默认 8-token 请求对
524-command 模板产生 4192 个已执行 command；Host runtime、bridge 和 Guest 验证均
拒绝只完成单步模板的旧结果。

逐 token 首次 GB10 验证暴露 sim-host page bundle 原先只能顺序消费一次：第二个
decode step 重复请求 command 0 后 `consumed == record_count`，pager 返回错误并导致
bridge step fatal。Pager 现按 command ID 回退或 bundle 消耗完毕识别新执行 step，
回卷记录流但保留累计 service/transfer 统计；单测覆盖同一 command graph 的第二轮
权重页请求。

GB10 逐 token 验收的 35872 次 page fault 等于每步 4484 次乘 8，说明 Hybrid 路径
重复搬运完全相同的只读权重。新增 decode-weight reuse 策略：prefill 保留完整分页，
后续 decode step 继续执行 524-command 调度并累计 operations/cycles，但复用 modeling
backend 中的固定权重驻留，不再生成重复 page fault。该策略仅是快速仿真默认值；设置
`CORAL_REUSE_DECODE_WEIGHTS=0` 可恢复每 token 全量分页，用于内存系统压力评估。

## 2026-08-19 Qwen Instruct prompt 与 token golden 验收

真实 token 生成默认由 Qwen tokenizer 的 `chat_template` 将 CPU 原始 prompt 包装为
user message 和 assistant generation prompt，再执行可复现的模型采样。此前直接
tokenize 裸字符串属于文本续写语义，可能重复生成看似无意义但数值上自洽的 token；
这不能作为 Instruct 模型的端到端正确性验收。

原始 CPU prompt 的 checksum 继续作为 Guest/bridge ABI 一致性检查，格式化后的模型
输入另行输出 checksum，不改变已有 inference request。结果缓存键加入 `chat|raw`，
避免复用旧 raw 结果。验收要求 Guest 返回的 token ID 序列和文本与生成 `.npxo` 时
打印的 Hugging Face golden 完全一致。运行脚本已自动提取 golden 中的生成数量和完整
token ID 序列，并对 Guest `coralctl` 输出做逐行精确比较，成功时打印
`token_golden=PASS`，不再依赖人工核对日志。该 golden 仍属于 Hybrid sim-host 数值实现，
后续由设备 MatMul/GPTQ、Attention、MoE 和归一化 kernel 逐项替换。

GB10 的 16-token 验收暴露 inference I/O 只在 256 字节 header 的 `reserved[]`
中内联返回前 12 个 token，导致设备推理本身 PASS、最终完整 golden 比较却失败。
现保留 256 字节主 ABI，在 output binding 紧邻区域发布最多 32 个 token ID，并在
header 中记录 offset/bytes。Bridge、Guest runtime 和单测均校验数组边界、完整序列、
词表范围及首 token，一次返回不再出现 `,...` 截断。

完整序列返回后确认旧 golden 本身为 16 次重复 token。根因是 host generator 显式
指定 `do_sample=False`，覆盖了 Qwen3.5 模型随包提供的 `generation_config.json`。
默认数值 golden 现改为使用模型自带的 sampling/temperature/top-k/top-p，并以固定 seed
保证可复现；`greedy` 仅作为显式回归模式保留。模型采样模式若仍生成全相同 token，
generator 将直接失败，避免再次把退化输出标记为端到端 PASS。

官方 sampling 消除单 token 循环后仍产生多语言碎片，说明 transport 与 golden 一致但
Host 数值结果不可信。生成器默认加载路径现由直接 `GPTQModel.load` 切换为 Qwen3.5
模型卡指定的 `AutoModelForMultimodalLM + AutoProcessor`，文本 prompt 也通过官方
多模态 processor 的 chat template/tokenize 接口构造。旧 `gptqmodel` loader 仅保留为
诊断回退，loader 名进入缓存键；运行脚本还会在复用任何缓存前拒绝全相同 token golden。

官方 Transformers 路径原先未显式传递 GPTQ kernel，自动选择的 Marlin 要求输出宽度
按 64 对齐，而模型包含 `out_features=32` 的窄投影，因而在 forward 前即失败。当前
加载器以模型原始量化参数构造 `GPTQConfig`，但强制默认 backend 为便携的
`gptq_torch`；backend 同时进入结果缓存键，避免复用由其它 kernel 生成的 golden。

针对新 GPU 上 host GPTQ kernel 可能出现的 CUDA illegal-address，数值生成器新增
`auto/cuda/cpu` 显式 placement，并将 placement 纳入缓存键。GPU 错误可通过
`CUDA_LAUNCH_BLOCKING=1` 定位；CPU 诊断/兜底路径使用 `gptq_torch_aten`。该选择只影响
Hybrid sim-host golden 生成，不改变 gem5、Coral RTL、分页和 Guest 端到端协议。
CPU placement 在独立生成器进程内清空 `CUDA_VISIBLE_DEVICES`，防止 GPTQ/Triton
再次自动选中 CUDA；量化权重加载信息中的 `qweight/qzeros/scales/g_idx` 缺失项升级为
硬失败，禁止以随机初始化参数生成伪 golden。

直接 GPTQModel CPU 路径已验证系统链路一致，但仍生成无语义的多语言碎片，因此
`token_golden=PASS` 被明确限定为 Host/Guest transport golden，而不是模型语义正确。
新增 vLLM offline reference provider，采用 Qwen3.5 官方建议的 `moe_wna16` 路径并直接
获取 `CompletionOutput.token_ids`，再通过既有 `.npxo`、NPU command completion 和
Guest 校验链路返回。最终语义验收必须使用可信 reference provider 并人工检查文本。
GB10/SM121 上 FlashInfer sampler 的架构检查会在部分 wheel 组合中误报低于 sm75；vLLM
reference provider 默认禁用该 sampler，并选择原生 sampler、`TRITON_ATTN` 和 eager
执行，避免已知的 SM121 FlashInfer/CUDA graph 启动故障。

GB10 已完成 vLLM reference 的 8-token 端到端验收：Guest 返回 token IDs 与 vLLM
逐项一致，文本以 `Okay, the user mentioned ...` 开始，证明结果具有基本语义而非此前
GPTQModel 的多语言碎片。real-weight runner 的默认 loader 因此切换为 `vllm`；
Transformers/GPTQModel 仅保留为显式诊断选项。

## 2026-08-20 Host C++ Functional Backend 基线

Qwen 数值开发主线已从“vLLM 结果注入”调整为“Host C++ 独立执行真实数值图”。
vLLM `.npxo` 发布路径继续作为 CPU/NPU 控制流、分页、completion 和结果回传的系统
回归基线，但不再代表 Host functional inference 正确性。最终 native 验收必须在不设置
`CORAL_SIM_HOST_INFERENCE_RESULT` 的情况下，由 Host C++ backend 自行产生 logits 和
token，再与 vLLM oracle 做仿真外差分。

首批模型无关 C++ Tensor kernel 已实现 Add、Mul、SiLU、RMSNorm、Softmax、RoPE 和
确定性 TopK，并统一返回 operations、bytes read/write 和 modeled cycles。新增
`Gem5HostFunctionalBackend` 以 generic opcode 为调度边界；未实现的 Attention 等命令
明确返回 `unsupported`，禁止只累计 estimated cycles 后伪造数值完成。现有 GPTQ INT4
MatMul 和 SwiGLU Expert 已注册到同一 backend，形成统一的权重算子与基础 Tensor 算子
执行入口。

当前阻塞完整 524-command 数值执行的主要接口缺口是中间 Tensor 映射：现有 executable
只有 input/output/weights/state/scratch 五类逻辑 binding，尚未携带逐命令虚拟 Tensor ID、
producer/consumer、liveness、scratch offset、KV cache 和 recurrent-state layout。下一增量
将由 compiler 生成模型无关 tensor-allocation side table，runtime 在 invocation 中绑定该
表，然后由 Host functional backend 按 dependency 顺序真实串联每条命令的输入输出。

该 compiler 增量现已落地：`model.npxt` 使用 SSA Tensor ID 明确记录 attention/state、
两级 residual 和 MoE 分支/汇合关系，并根据 producer/last-consumer 生命周期复用 scratch
slot。QKV 已拆分为独立 Q/K/V，RoPE、KV-cache update 与 Attention 按真实依赖连接。
40 层 Qwen3.5 形态的 524 条命令被完整映射为 625 个 Tensor 和 6 个 scratch slot；
独立 validator 会拒绝缺命令、先读后写、活跃区间冲突及没有生产者的最终输出。当前仍待
runtime 将符号 shape/slot 实例化为设备地址并逐命令调用 Host C++ backend。

同时新增 `model.npxtb` 二进制 side table 与 C runtime loader。Loader 对 checksum、
record bounds、producer-before-consumer 和 slot 容量做强校验，并可按实际 batch/sequence
计算 scratch/persistent 容量，把 input/output/scratch/persistent Tensor ID 解析为边界受控
的设备地址。

Bridge generic dispatcher 已完成第一步接入。固件仍通过既有 `NPU_LAUNCH`/MMIO doorbell
提交外层 Coral descriptor，新增 `GENERIC_COMMAND` 外层 opcode；内层 generic opcode 在
二级译码阶段选择 Tensor/Vector/SFU。外层 descriptor 指向 EXTMEM 中版本化、纯地址的
functional request，bridge 校验每个 operand 范围后调用 `Gem5HostFunctionalBackend`，并
沿原有异步五级命令链回填 operation/traffic/modeled-cycle 统计。ADD 与 GPTQ MatMul 数值
执行、非法 envelope 拒绝以及三类 engine 路由已有独立单测。下一步是 runtime/firmware
根据 `.npxtb` 为每条 `.npxe` command 物化 request，使完整命令流真正进入该数值执行链。

新增 `gem5_generic_launch_smoke.elf` 作为上述边界的 RTL 端到端门禁。Coral 固件在本地
EXTMEM 构造 float32 ADD functional request，再通过真实 `NPU_LAUNCH` 指令提交；Guest
继续使用既有 mailbox 验证 `11,22,33,44`、operation count 和 completion。该测试要求
GB10/x86 Linux 的 RISC-V/Verilator toolchain 构建并运行。

完整 tiny-Qwen 通用命令流的下一个增量已经形成。Host functional backend 新增模型无关
float32 Embedding 与 dense MatMul kernel，至此 tiny-Qwen 所需的 Embed、MatMul、Add、Mul、
RMSNorm、RoPE、SiLU、Softmax、TopK 九类 primitive 均可通过统一 functional request 执行。
`gem5_qwen_command_flow_smoke.elf` 在 Coral RTL Core 上依次发出九条 `NPU_LAUNCH` 自定义
指令，并使用 EXTMEM 中间 Tensor 串联真实数据依赖；每条外层 `GENERIC_COMMAND` 均由
二级译码分派至 Tensor/Vector/SFU engine，再经过 fetch/read/execute/writeback/complete
微命令链。最终 TopK 索引必须为 5，避免仅凭完成计数产生伪 PASS。

协处理器命令结构同时保留外层 `operator_opcode` 和内层 `generic_opcode`，Host 日志和
GB10 验收脚本会逐项确认九种 opcode 均从 `source=custom-instruction` 进入命令处理器。
该 smoke 验证的是完整控制流、二级译码、依赖数据流和 Host C++ primitive 数值链；它
不是 35B 模型的完整权重推理。后续仍需让 `.npxe + .npxtb` runtime 按 Tensor allocation
逐条物化相同 request，并接入分页真实权重、KV/state 迭代与多 token decode。

`.npxtb` runtime 已新增 command 级 Tensor-view resolver。调用方传入 command ID、实时
batch/sequence/KV/active-expert shape 和 input/output/persistent/scratch memory binding，
即可一次获得该命令全部输入输出的 Tensor ID、数据类型、实际维度、NPU 地址和字节数。
resolver 复用既有 scratch liveness slot 与 persistent layout，并对每个 view 独立执行容量
和溢出校验。模型包回归已覆盖 30-command/37-Tensor/6-slot 图。下一步是将这些通用 view
映射为 opcode-specific operand role，补齐多输出算子和 GPTQ weight view 后生成 functional
request。

Host functional 图调度入口现已增加两项强约束。`functional_program_init()` 会在任何
Tensor 地址解析前验证 invocation 中的 command ID 均落在 `.npxtb` 范围内且不重复，
并要求同一次 invocation 的全部命令使用一致的 runtime shape；这避免部分命令流、分页
重放或损坏的 submission 静默复用错误的 scratch/KV 地址。模型包单测覆盖重复 command
ID 和不一致 runtime shape 的拒绝路径。

Bridge 的单命令数值执行也已从 Coral custom-instruction envelope 中解耦。
`ExecuteGem5FunctionalRequest()` 可直接执行 materializer 生成的地址型 functional
request，原有 `DispatchGem5GenericCommand()` 仅保留外层 descriptor 校验和统计回填。
因此后续 524-command 图调度器不再需要为每条命令伪造 `GENERIC_COMMAND` 外壳；同一
Host C++ kernel 路径仍可由自定义指令 smoke 和图级 scheduler 共同复用。当前 native
测试已验证直接 API 与既有 descriptor API 对 ADD 得到相同数值和统计结果。下一步是
在 bridge 中加载 `.npxtb`，分配 Host tensor arena，并将分页 residency 转换为每条 GPTQ
request 的 component operands。

上述 Tensor arena 基础现已落地。通用 runtime 会按实时 batch/sequence 将 input、output、
persistent 和可复用 scratch slot 排列到一个对齐的连续地址空间；同一 storage class 内的
多个 Tensor 也拥有独立 offset，不再错误别名到同一基址。Coral bridge 通过
`Gem5HostTensorArena` 直接复用 canonical `.npxtb` loader，获得逐命令 Tensor view，并将
边界校验后的 NPU 地址翻译为 Host backing buffer。相关实现由 overlay 脚本同步到 Coral
Bazel workspace，避免 runtime 与 bridge 各自维护一份二进制 ABI/parser。

模型包回归已在 30-command、37-Tensor、6-scratch-slot 的 Qwen 形态图上验证连续布局、
逐命令解析、Host 地址翻译、容量不足和越界拒绝。当前尚未宣称 35B Host functional
数值闭环：下一增量需要按 opcode 将这些 Tensor view 物化为 functional request，并把
分页 residency 中的 GPTQ qweight/qzeros/scales/g_idx 绑定为 component operands，随后才
能按 dependency 顺序执行完整 graph 并与 vLLM oracle 比对 logits/token。

Host functional 执行器进一步支持由多个非连续 memory region 组成的 NPU 地址空间。
invocation/operator parameters 可保留在 submission region，中间 Tensor 位于 Host arena，
GPTQ component 则由分页 cache region 提供；所有 operand 仍使用统一 32-bit NPU 地址，
执行前逐 region 校验范围。原有单一 EXTMEM API 保持兼容并退化为 one-region wrapper，
因此 RTL custom-instruction smoke 不受影响。Native 回归已验证跨 submission/arena 的 ADD
数值执行以及未映射地址拒绝。下一步不再需要复制参数或 Tensor 到伪造的连续 EXTMEM，
可以直接组装三类 region 后执行 materialized command。

模型无关 `Gem5HostFunctionalGraph` 现已组合 invocation、`.npxtb` Tensor arena、
materializer 和 segmented-memory executor。它从已校验 command 的 runtime shape 配置
arena，保留 submission 中 operator parameter 的原始 NPU 地址，并允许分页权重作为额外
region/operand 注入；因此调度器不再依赖 Qwen tensor 名称。本机回归从真实生成的
`.npxc` 实例化 30-command invocation，选择其中 residual ADD，解析两个输入和一个输出，
执行 36 个 float 元素并逐项验证结果。

同时修复 generic dispatcher 的 MoE 数据通路：此前 Host backend 虽实现 GPTQ Expert，
dispatcher 却没有构造 gate/up/down 三组 GPTQ weights、gate/up/activation 临时缓冲和
`gptq_expert_operands`，任何真实 EXPERT request 都会失败。现在三组 projection 和中间
Tensor 均由版本化 operand role 解析；最小 int4 SwiGLU Expert 回归已完成真实数值执行。
下一增量将从 `.npxr`/safetensors 为每条命令构造这些额外 operand，并处理 active-expert
循环与 route-weight combine。

### 通用 Host Functional 数值闭环

真实 Qwen3.5-35B GPTQ 模型已作为首个验收载荷，通过通用 `.npxe/.npxc/.npxtb`
执行链完成 524-command prefill/decode，并由 Host C++ primitive 连续生成 8 个与 vLLM
greedy reference 一致的 token。该结果不改变平台边界：Qwen graph parser、模型 layout
和 tokenizer 位于 CPU compiler/runtime；NPU 只消费通用 invocation、tensor binding、
command、operator parameter 和 completion。历史 `OPENNPUX_QWEN_TCB_*` 仅保留为 tiny
bring-up 回归，后续不得增加生产字段。

数值闭环确认 BF16 是 command 边界的数据类型语义，而非 Qwen 特例。Host backend 采用
FP32 累加并按 tensor/模型契约执行 BF16 RNE 边界，后续应继续把 dtype、layout、stride、
accumulator dtype 和 quantization 全部固化到通用描述符，由 Host/Hybrid/RTL 共享同一命令
流。定位阶段加入的逐 command、expert 和 logits 输出已改为显式开关；默认验收只保留
错误诊断、统计摘要和最终结果。

NPU 完成结果仍以 logits/token ID 为硬件接口，CPU Runtime 使用模型 tokenizer 将设备
实际返回的 token IDs 解码为 `inference_token_text`。Tokenizer 不进入 NPU RTL，也不作为
Golden 结果替代 NPU 计算。

## 2026-08-28 XOpenNPUX 通用算子指令功能库

Coral L1 `custom3` 分流、NPU L2 译码和 C++ 功能协处理器现已覆盖 tiny-Qwen 验收图所需
的全部九类通用 primitive：TMMA/Linear、ADD、MUL、RMSNorm、Softmax、RoPE、SiLU、
Gather/Embedding 和 TopK。每个 primitive 均由 32-bit 指令、accept-time CSR snapshot、
功能执行、周期/操作统计、`tfence` 和独立固件结果检查组成；统一端到端固件不使用模型
名称、descriptor doorbell 或 Host hybrid operator shortcut。

Embedding 被定义为通用 `TGATHER`，RoPE 通过 CSR 选择 adjacent/half-split 布局并消费
预计算 cos/sin 表，TopK 输出 packed values/indices 并定义稳定 tie/NaN 次序。Attention、
KV-cache、Router、Expert、Combine 和 recurrent block 保持为编译器/runtime 组合图，后续
由上述 primitive、MOV/TDMA 和同步指令构成，不能在 NPU L2 中加入 Qwen 特定识别。

本机严格 C++ 单测覆盖正常数值、RoPE 双布局、TopK tie-break、非法 CSR 和 Gather 越界；
GB10 门禁由 `test_tmma_operator_e2e.sh`、`build_rtl_bridge.sh` 和
`run_tmma_operator_e2e_test.sh` 完成 RISC-V ELF、Coral RTL 及 gem5 全系统验证。

## 2026-08-29 XOpenNPUX 九类 primitive GB10 验收

GB10 全系统运行已完成 TMMA 到 TTOPK 的全部九类 primitive 验收，共 11 个独立数值
case：TMMA 三组，TADD、TMUL、TRMSNORM、TSILU、TSOFTMAX、TGATHER、TROPE、TTOPK
各一组。所有 dispatch 均被 NPU L2 接受，completion 均为 `error=0`，writeback 地址、
字节数和 checksum 与独立 CPU 期望一致；每个异步算子后的 `tfence` 均在队列排空后以
`pending=0` 接受。

验收 checksum 基线为：TMMA `0xe6084308/0x515811d8/0xacc1ee78`，TADD
`0x16ace36b`，TMUL `0x2ac700dc`，TRMSNORM `0x8b3b7905`，TSILU
`0x137fe900`，TSOFTMAX `0x0fd06045`，TGATHER `0x269eb168`，TROPE
`0xe4adc6cb`，TTOPK `0xbb900cd1`。

本阶段完成的是“Coral RTL 控制核 + 自定义指令 L1 分流 + NPU L2 译码 + C++ 功能协处理器”
的通用算子指令闭环。它证明算子库可以沿真实 Coral 指令流水和异步协处理器协议执行，
但不表示九类算子均已实现为 cycle-accurate RTL FU；后续时序 RTL、并行 issue、片上存储
调度和性能模型将在保持现有 ISA/CSR/完成协议不变的前提下逐项替换功能执行引擎。

## 2026-08-29 Guest 执行图到 XOpenNPUX 指令流

新增模型无关 `OPENNPUX_XGRAPH_V1`，消除 Qwen command-flow smoke 中残留的
`NPU_LAUNCH + descriptor + Host hybrid operator` 路径。Linux Guest 现在在 8MiB shared
DMA window 中写入 64-byte command records 和输入 Tensor；Coral firmware 校验 graph 后
逐条调用 `xopennpux_ops.h`，生成 CSR 配置、XOpenNPUX 32-bit 自定义指令和 `tfence`。
最终 packed TopK Tensor、checksum 和 completion 再同步回 Guest。

首个 graph 保持九类 primitive 的真实数据依赖，最终 TopK index golden 为 `5`。全系统
脚本不再接受旧 generic descriptor 的 micro-op 日志，而是要求 TGATHER、TMMA、TADD、
TMUL、TRMSNORM、TROPE、TSILU、TSOFTMAX、TTOPK 均出现 NPU L2 dispatch 和
`error=0` completion。该增量建立的是后续 Qwen3 功能正确性 reference 通路；下一步仍需
将 `.npxc/.npxtb` 的 524-command invocation 自动 lowering/tiling 到同一 graph ABI，并
接入 GPTQ page bindings、KV/state 和最终 token golden。

## 2026-08-29 Generic command 到 XGraph primitive lowering

新增模型无关 `opennpux_npu_xgraph_lower_primitive()`，直接消费 `.npxc/.npxtb`
物化后的 functional request，将 FP32 `EMBED/MATMUL/ADD/MUL/NORMALIZE/ROPE/
SOFTMAX/ACTIVATION/TOPK` 转换为 XGraph primitive。转换只使用 opcode、显式语义
选项、shape 和 EXTMEM operand，不读取 Qwen、layer 或 Tensor 名称。

`COMBINE` 的数值语义与逐元素 ADD 等价，因此规范化为 `TADD`，不新增模型专用指令。
`DMA` 由通用 KV shape (`rows/kv_heads/head_dim/kv_length`) 分解为 Key/Value 两条
`TDMA`，目标地址指向两个 state plane 的可见尾部；未覆盖的历史 cache 必须保持不变。
GPTQ MatMul 与 `ATTENTION/CAUSAL_CONV/RECURRENT_UPDATE/EXPERT`
仍明确返回 `ENOTSUP`，等待 decomposition/tiling pass 展开，防止把复合 command
错误伪装成单条自定义指令。native gate `test_xgraph_lowering.sh` 已覆盖直接映射、
RoPE/SiLU/TopK 显式语义、EXTMEM 地址以及 unsupported 路径。

批量 lowering 接口 `opennpux_npu_xgraph_lower_sequence()` 进一步把单条转换扩展为
有序 command stream。接口强制 command ID 与 retirement 顺序一致，并在遇到尚未分解的
复合算子、GPTQ MatMul、非法地址或容量不足时返回首个失败 command 的 index、ID、opcode
和 errno。该接口不跳过 unsupported command，也不插入模型特例，是 524-command
invocation 接入 decomposition/tiling pass 前的可诊断边界。

## 2026-08-29 GPTQ MatMul 通用 Tile Plan

新增模型无关 `opennpux_npu_gptq_plan_tiles()`，把 materialized GPTQ MatMul 的
`input/qweight/qzeros/scales/g_idx/output` operand 和量化参数转换为输出通道 tile。
规划器严格采用 AutoGPTQ 物理布局：qweight 沿 K 轴 int4 打包、qzeros 沿 N 轴打包、
scales 为 group-major、g_idx 为可选 K 映射；每个 tile 输出带 row count、row stride、
row bytes 和起始地址，不能再把跨行分量错误表示为连续内存。

tile 宽度由调用方提供的 dequant scratch 容量决定，非尾 tile 按 8 个输出通道对齐，
尾 tile 支持非整除 N。规划阶段同时验证 EXTMEM aperture、operand 实际容量、int4/group/
scale dtype 契约和 64-bit 大小计算。该层只建立后续 `TDEQUANT + TMMA` 指令序列的稳定
输入，不把 Host GPTQ kernel 冒充自定义指令执行。

在此基础上已新增模型无关 `TDEQUANT` 32-bit 自定义指令及实验 CSR `0x810..0x817`。
Coral 标量核仍负责 CSR 指令和 custom3 的 L1 取指/译码/retire；accept 时将 qzeros、scales、
可选 g_idx、量化配置及三类 byte stride 与指令一起快照给 NPU L2。C++ 功能协处理器按
AutoGPTQ packed layout 完成 INT4 到 FP32 scratch 的数值反量化，拒绝非法 stride、越界
g_idx 和非有限 scale。native 回归已覆盖带 padding 的分量行、非连续 g_idx、3 通道尾 tile，
以及 `TDEQUANT -> TMMA` 组合结果。

XGraph 保持 64-byte command ABI，新增通用 TDEQUANT opcode；reserved 字段被明确编码为
scales/g_idx offset 与 qweight/qzeros/scales row stride。GPTQ lowering 现按 `8/8/tail`
等 N tile 生成一次反量化，并在 stride CSR 尚未冻结前按输出 row 展开 FP32 TMMA，确保
写回地址正确。18 通道、3 行、32 输入维测试生成 12 条命令并逐项校验地址和容量失败。
GB10 已验证 `TDEQUANT -> TMMA` 小矩阵数值闭环。针对实验 `mma_shape` 的 10-bit K，
lowering 已继续实现二维 N/K tile：K tile 同时按 int4 打包和 GPTQ group 边界对齐，
第一片直接写输出，后续片写 partial scratch 并通过 TADD 显式累加。新增 `0x817`
group-range CSR 保证全局 g_idx 在切片后仍按原模型量化组解释。K=2048 回归规划为
896/896/256，下一验收点是在 GB10 执行大 K 的完整 TDEQUANT/TMMA/TADD 指令链。

## 2026-08-30 XGraph 有界多批次提交

Guest runtime 已从“一次 lowering 并执行全部命令”升级为有界批次循环。测试刻意把物理
command capacity 限制为 9：第一批执行 TGATHER 到 TTOPK 的 9 个 primitive，第二批把
第 10 个 GPTQ MatMul 作为不可拆分逻辑请求展开成 `TDEQUANT + 2xTMMA`，再执行第 11 个
COMBINE 请求所规范化得到的 `TADD`，以及第 12 个 DMA 请求分解得到的两条 `TDMA`。
第 13 个 Router 请求独占第三批，展开为 `TMMA + TTOPK + TSOFTMAX + 2xTDMA`：NPU
先计算 expert logits、选择 Top-K、仅在选中项内归一化，再把 weights 和 indices 写回两个
独立输出。三批聚合为 13 个逻辑请求、20 条物理命令和 264 modeled cycles，并继续逐算子
进行独立数值校验。

XGraph header 的保留字段现定义 batch sequence、first request、request count、final flag
和 global first command，不改变 96-byte ABI。每批 command ID 保持从 0 开始的局部稠密
retire 顺序，全局完成进度由 Host runtime 聚合。Firmware 已移除固定 `TopK index == 5`
判断，只负责 ABI、地址、opcode、执行和完成协议；模型结果 golden 留在 CPU runtime，
避免后续接入任意 Transformer 执行图时把模型语义固化进 Coral 控制固件。

Router lowering 只依赖 `rows/input_features/output_features/top_k` 和显式 operand，不读取
模型名、层号或 expert 策略。五条命令必须作为一个不可跨 batch 的原子复合请求提交；临时
scratch 依次保存 dense logits 和 packed `[TopK values][TopK indices]`。`TSOFTMAX` 仅覆盖
packed values 区域，随后两条 `TDMA` 分别发布归一化权重和 uint32 expert IDs。该顺序与
现有 Host functional Router 的“Top-K 后对选中项归一化”语义一致。

逐算子 lowering、测试 shape、物理 command 数和实际 functional modeled cycles 的规范化
台账见 `docs/design/xgraph_operator_validation_matrix.md`。后续每次 GB10/full-system PASS
必须在同一变更中补充 generic opcode、指令序列、requests/commands/batches、实际 cycles/
operation count、checksum 或 max error 以及测试环境；RTL 性能数据另列，不覆盖功能模型
基线。

## 2026-08-30 Stateful Causal Convolution 指令闭环

新增模型无关 `TCAUSALCONV` custom3 指令，将 generic
`CAUSAL_CONVOLUTION` 的 rows、features、kernel width、可选 SiLU 以及前后 state
地址 lowering 为单条 XGraph command。Coral 标量流水仍执行 CSR 写入、L1 custom
instruction 分流和正常 retire；NPU L2 在 accept 时原子快照 `tensor_shape`、
`scalar_param0` 及新增辅助 state CSR `0x818/0x819`，C++ functional coprocessor 完成
地址检查、depthwise causal convolution、next-state 更新和 completion。

native 数值测试覆盖 rows=2、features=2、kernel=3 的 stateful 场景，输出为
`[14,14;20,20]`，next-state 为最新两行输入，统计 24 element operations / 24 modeled
cycles；runtime lowering 同时覆盖 operand 容量、state 成对出现、EXTMEM offset 和 fused
SiLU 标志。

Guest XGraph full-system fixture 新增第 14 个逻辑请求，在第三批 Router
复合请求之后发出 1 条 `TCAUSALCONV`，同时校验 convolution output 和 next-state，预期
聚合结果为 3 batches / 14 requests / 21 commands / 288 cycles。GB10 全系统测试已按该
统计通过，`xgraph_operation_count=288`，14 个算子全部 `max_abs_error=0`；其中
`TCAUSALCONV` checksum 为 `0xaa4fb265`。验收脚本保留
`xgraph_op_CAUSAL_CONVOLUTION=PASS` 强制检查。该周期数据属于 C++ functional model，
后续 RTL 实现必须另行记录时序性能，不覆盖本功能基线。

## 2026-08-30 Basic Recurrent State Update 指令闭环

新增 generic `RECURRENT_UPDATE` lowering，但不引入模型专用指令。基础语义被原子降低为
两条现有 `TDMA`：第一条复制完整 `[rows, features]` 输入到可见输出，第二条复制最后一行
到 `OUTPUT_SECONDARY` 持久状态。两个物理 command 都映射回同一逻辑 request，batch
容量不足时不得拆分。带 `OPENNPUX_NPU_PARAMETER_GATED_DELTA_NET` 的复杂状态递推明确返回
`ENOTSUP`，后续应由独立 compute sequence 实现，不能错误退化为 copy。

Native lowering、runtime host 和 GB10 全系统测试均已通过；fixture 第 15 个逻辑算子同时
校验完整输出和最终状态。正式基线更新为 3 batches / 15 requests / 23 commands /
294 cycles，`xgraph_op_RECURRENT_UPDATE=PASS`，checksum `0x14a86f48`，15 个算子全部
`max_abs_error=0`。该周期数据仍属于 C++ functional model，不代表 RTL 时序性能。

## 2026-08-30 Generic GPTQ Expert Lowering Acceptance

新增模型无关 GPTQ `EXPERT` decomposition。lowering 不读取模型名、layer ID 或 expert ID，
只消费 functional request 中显式的 input/output、gate/up/down GPTQ 分量以及三个中间
tensor operand。一个逻辑 Expert 被原子展开为 gate 和 up 两次 `TDEQUANT+TMMA`、
`TSILU+TMUL` 激活门控，以及 down `TDEQUANT+TMMA`。三次投影顺序复用 dequant scratch，
但不会把模型 layout 固化进协处理器。

Native lowering 已验证最小 GPTQ expert 生成 8 条稠密 command，并覆盖 operand 重映射、
中间 tensor 地址、in-place activated multiply、command origin 以及容量不足时不拆分请求。
Full-system fixture 增加第 16 个逻辑请求和独立第 4 批。GB10 先确认 4 batches / 16 requests /
31 commands / 326 modeled cycles 的执行闭环；随后与 `TATTENTION` 一同完成 17 算子聚合
正确性验收。Expert 的 8 条物理命令保持同一逻辑 request 的原子边界，不允许跨 batch 拆分。

## 2026-08-30 Generic Causal GQA Attention Acceptance

新增模型无关 `TATTENTION` custom3 指令及 ATTENTION lowering。XGraph command 显式携带
query rows、heads、KV heads、head dimension 和 KV length，输入约定为 query
`[rows,heads,head_dim]` 以及连续 K/V state
`[2,kv_length,kv_heads,head_dim]`。Coral 标量核负责 CSR 写入、L1 分流和 retire；NPU L2
原子快照新增 CSR `0x81a..0x81c`，执行 GQA head 映射、causal visible prefix、scaled dot
product、stable softmax 和 V 聚合。该接口不携带模型名、层号或 Qwen 专属信息。

Native tests 已验证 lowering 和 rows=2、heads=2、KV heads=1、head dim=2、KV length=3 的
数值执行，对应 80 operations / 80 modeled cycles。GB10 full-system fixture 已完成 4 batches /
17 requests / 32 commands / 406 cycles；第 4 批由 Expert 的 8 条命令和 1 条 `TATTENTION`
组成，`xgraph_validated_operators=17`、`xgraph_correctness=PASS`。正式功能基线因此提升为
17 个通用模型算子。406 cycles 属于 C++ functional model，不代表 RTL 时序性能。

## 2026-08-30 Remaining Generic Lowering: Gated Attention and Recurrent

在不增加模型专用 opcode 的前提下补齐两类剩余 Transformer 语义。带 tertiary gate 的
ATTENTION 继续使用 `TATTENTION`，通过 `0x818` 快照 gate 地址，并在 `0x81b` 高 16 位设置
gate flag；NPU L2 在 attention 输出上逐元素应用 sigmoid gate。native 数值测试采用
rows=1、heads=1、head_dim=1、KV length=1，输出 1.5，统计 8 modeled cycles。

带 `GATED_DELTA_NET` 标志的 RECURRENT_UPDATE 不再返回 `ENOTSUP`，而是 lowering 为单条
`TRECURRENT`。新增 CSR `0x81d..0x821` 携带 key/value heads、key/value dimensions、beta、
A-log 和 dt-bias，复用 `0x819` 持久 state 地址。C++ NPU L2 实现 Q/K normalize、sigmoid
beta、softplus decay、state update 和输出投影；最小 native 数值闭环统计 21 modeled cycles。
Coral 标量核仍只负责 CSR、L1 custom3 分流及正常 retire，全部计算语义属于 NPU 二级译码
和协处理器。

lowering 与 coprocessor native tests 均已通过，并在 GB10 完成 full-system 验收。第五批
原子提交 gated `TATTENTION` 与 `TRECURRENT`，最终结果为 5 batches / 19 requests /
34 commands / 435 modeled cycles，`xgraph_validated_operators=19`、
`xgraph_correctness=PASS`。这组数据替代 17 requests / 32 commands / 406 cycles，成为当前
通用算子硬件 lowering 功能基线。普通 CONVOLUTION 仍明确返回不支持，因为当前 generic
request ABI 没有 stride、padding、dilation 和 layout；在 ABI 完成前禁止把它错误映射为
depthwise causal convolution。

## 2026-08-30 Generic Conv2D Functional Instruction Acceptance

在已验收 19 算子基线上新增模型无关 `CONVOLUTION -> TCONV` 通路。generic lowering ABI
现在显式携带 input/weight/output layout、stride、四向 padding、dilation 和 groups；当前 v0
功能 profile 限定 FP32 NHWC input/output 与 OHWI weight，支持 optional bias 和 grouped
convolution。固定 64-byte XGraph command 仅使用 `reserved[0..4]`，没有扩大既有 Guest/Host
ABI。

Coral 标量核新增 `0x822..0x82a` custom CSR 快照和 custom3 L1 识别；`TCONV` 的参数检查、
地址范围、二级译码、分组卷积数值计算和写回全部位于 NPU C++ functional coprocessor。
native test 覆盖 groups=2 的 1x3x3x2 input、2x2 kernel、2 output channels 和 bias，独立验证
8 个输出以及 32 MAC / 32 modeled cycles。lowering 单测覆盖有效请求和无效 layout、shape、
group、buffer range。

GB10 full-system fixture 已完成 5 batches / 20 requests / 35 commands / 451 modeled
cycles，`xgraph_op_CONVOLUTION=PASS checksum=0x40a9cead max_abs_error=0`、
`xgraph_validated_operators=20`、`xgraph_correctness=PASS`。该数据替代 5 / 19 / 34 / 435，
成为当前通用算子硬件 lowering 功能基线。451 cycles 只表示 C++ 功能模型的逻辑 MAC
计数，不代表 RTL 卷积流水性能。

## 2026-08-31 TopK 真实双输出 Tensor ABI

端到端 524-command functional request 中 TopK 的 values 与 indices 是两个独立输出
Tensor，而早期 XGraph smoke 只支持测试专用的 packed `[values][indices]` scratch。该差异会
使真实 invocation 无法直接复用 generic lowering。现已在不改变 32-bit `TTOPK` 编码的
前提下，复用 custom CSR `tensor_aux_destination_address`：`rd` 指向 FP32 values，CSR
`0x819` 指向 uint32 indices；NPU L2 在 accept 时原子快照两者并分别做地址检查和写回。
辅助地址为零时仍保留 packed 兼容语义，供旧 native smoke 和 Router 内部临时 scratch 使用。

runtime lowering 现在优先消费显式 `OUTPUT + OUTPUT_INDICES` operands，并设置 XGraph
split-output flag；20 算子 full-system fixture 也已改为两个真实 operand，连续内存布局仅用于
保持既有结果读取和 checksum。native coprocessor 数值测试验证 2x5、K=2 的独立 values 与
indices 写回，lowering 单测同时覆盖 split 和 packed fallback。该改动不增加逻辑 request、
物理 command 或 modeled cycles，因此 GB10 复验目标保持 5 batches / 20 requests / 35
commands / 451 cycles。

GB10 已完成 split-output full-system 复验，统计保持 5 batches / 20 requests / 35
commands / 451 cycles，全部 20 个算子继续 `max_abs_error=0`。因此双输出 TopK 已纳入正式
功能基线，不再属于 pending 项。

## 2026-08-31 真实 Materialized Request Lowering 审计

为把 35B Host C++ 正确性参考逐步替换为 XOpenNPUX 指令流，`Gem5HostFunctionalGraph`
新增同步只读 request observer。observer 位于所有普通 numerical kernel 的统一 `Execute()`
边界，可看到实际执行使用的完整 operand、参数地址和 memory regions；动态 routed expert 的
Host-fused 旁路则以独立 execution path 显式上报，禁止静默计入硬件 lowering 覆盖率。

新增 `Gem5XGraphLoweringAudit` 对 prefill 首步的真实 materialized requests 逐条调用同一套
generic XGraph lowering，汇总 observed/lowerable/host-fused requests、物理 command 数以及按
opcode/errno 聚合的首个失败 command。设置
`CORAL_HOST_FUNCTIONAL_XGRAPH_AUDIT=1` 后，35B Host preflight 会输出该报告但不改变数值
执行或 token 选择。目标验收为 `observed_requests=524`，随后按报告逐项消除 QKV、linear
attention、动态 routed expert 等剩余语义缺口，最终达到 `xgraph_audit_complete=PASS`。

GB10 首次真实审计确认 524 个 numerical requests 全部被 observer 捕获，其中 362 个已可
lowering，40 个为显式 Host-fused routed expert，已 lower 的请求产生 532 条物理命令。剩余
失败按实际接口聚合为：71 个 `MATMUL/EINVAL`、10 个 `MATMUL/ENOSPC`、40 个
`EXPERT/EINVAL`、40 个 `EXPERT/ENOTSUP` 和 1 个 `TOPK/EINVAL`。后续带 operand 细节的
审计已确认两个 Expert 组是不同请求：`ENOTSUP` 是 40 个动态 routed-expert 请求，
`EINVAL` 是 40 个 dense shared-expert 请求，不能合并计数。

审计同时暴露了一个通用分派错误：可执行文件允许在模型/command capability 中保留 GPTQ
flag，但某个 materialized MATMUL 仍可能绑定 dense FP32 `WEIGHT`。Host C++ 数值后端已经
按实际 operand 类型选择 kernel，XGraph lowering 此前却只看 flag，错误强制进入
TDEQUANT。现已统一为 operand-driven dispatch：只有存在 `QWEIGHT/QZEROS/SCALES` 时才
选择 GPTQ tile decomposition；存在普通 `WEIGHT` 时即使 capability 含 GPTQ 也 lowering
为 dense `TMMA`。新增回归测试覆盖该混合能力场景。

为避免继续依据 errno 猜测复合语义，审计报告新增每个失败组首个请求的 shape、attention
维度、TopK、参数 flags、输入/输出/中间维度、量化配置及完整 `operand-role@bytes` 签名。
下一轮 GB10 审计将据此实现 linear-attention 三投影、gated Q/K/V GPTQ 复合 lowering、
last-row indices-only TopK 和 device-routed Expert，不引入模型名或层号特例。

GB10 第二轮审计在 operand-driven MATMUL 修复后达到 443 lowerable / 613 emitted，所有
MATMUL 失败消失。last-row、indices-only TopK 随后补齐，第三轮达到 444 lowerable / 614
emitted，剩余正好是 40 个 Host-fused routed Expert 与 40 个 shared Expert。Shared Expert
真实语义为 gate/up/down/router 四次 dense projection，加 SiLU、elementwise multiply、
sigmoid 和 row broadcast scale。XOpenNPUX 已新增模型无关 `TSIGMOID`、`TROW_SCALE`
二级译码及 C++ functional execution profile，为该分解提供基础原语。

该审计也暴露出 lowering acceptance 与可执行性不能混为一谈：当前 `mma_shape` 的 M/N/K
物理编码各为 10 bit，而真实 dense projection 使用 2048/5120 等维度；Host C++ dense
weight 还是 `[output_features,input_features]`，与当前 TMMA functional engine 的连续
`[K,N]` RHS 假设不同。在 stride/transpose/accumulate CSR 和 dense tiled-TMMA pass 完成
前，禁止仅用单条 TMMA 把 shared Expert 标记为 lowerable，否则 audit 会假 PASS、执行会
发生 shape 截断或权重布局错误。

上述执行缺口现已按通用硬件语义补齐。TMMA 新增 lhs/rhs/destination row-stride CSR、
transpose-RHS 和 accumulate flag；functional coprocessor 使用 stride 做边界检查和寻址，并
允许后续 K tile 从 destination 继续累加。dense lowering 将通用 `[N,K]` 权重按 10-bit
shape 上限在 M/N/K 三维切片，每条命令携带原 tensor stride，非整除尾 tile 不需要重新布局。
`18x2048` 乘 `[512,2048]` 的真实投影因此生成 `K=1023+1023+2` 三条 TMMA。

shared Expert 的分派也改为 operand-driven：出现 roles 48/49/50/51 时，不再被模型级 GPTQ
flag 误送入 routed GPTQ expert 路径，而是展开 gate/up/down/router 四个 tiled dense
projection 与 `TSILU/TMUL/TSIGMOID/TROW_SCALE`。Qwen3.5-35B 当前维度下每个 shared
Expert 生成 16 条 XOpenNPUX 命令。新增 lowering 与 coprocessor 回归覆盖 transpose、非默认
stride、K-tile accumulate、尾 tile 和完整 shared Expert 序列。下一次 GB10 审计预期消除
`EXPERT/EINVAL` 的 40 个 shared Expert；保留的 40 个 `EXPERT/ENOTSUP` 是有意显式标记的
动态 routed Expert，需由后续 device-side TopK route、expert paging 与聚合控制流解决。

GB10 复验确认 shared Expert 的 40 个 `EINVAL` 已消失，同时暴露 40 个三路 dense projection
MATMUL。每个 request 共享 `[rows,K]` 输入，但显式携带三组 `[N,K]` 权重和三个输出；真实
shape 为 `18x2048 -> 8192/32/32`。generic lowering 现按每个 weight operand 的字节数除以
`K*sizeof(float)` 推导各自 N，并依次复用 tiled dense TMMA，不读取模型名、层号或固定 Qwen
shape。该真实 shape 分别生成 `27+3+3=33` 条命令，覆盖主投影以及两个窄投影的独立输出
地址、stride 和尾 tile。下一轮审计预期达到 484 lowerable、约 3762 emitted；此后唯一保留
的失败应为 40 个有意 Host-fused 的动态 routed Expert。

GB10 下一轮审计达到 474 lowerable / 3432 emitted，验证上述三路 projection 已消除；新增
的 10 个 MATMUL 是四输出 attention projection。其显式 roles 52/53/54 为 FP32 Q/K/V
权重，roles 55/56 为逐 head Q/K norm，role 57 为 gated-query 输出。当前 generic lowering
按 `heads/kv_heads/head_dim` 展开，不使用模型名：Q 权重按每个 head 的 `[query,gate]` 两块
分别生成 strided tiled TMMA，K/V 各自投影，随后 Q/K 逐 row/head 生成 TRMSNORM。真实
`rows=18, K=2048, heads=16, kv_heads=2, head_dim=256` 请求生成 426 条物理命令，其中
102 条 projection TMMA、324 条 per-head norm。TRMSNORM command 同时保留 weight-offset 与
BF16-input flags；Coral CSR `0x82f tensor_flags`、L1 snapshot、bridge packet 和 NPU functional
coprocessor 已端到端实现这两个语义。下一轮审计预期 484 lowerable、约 7692 emitted，只剩
40 个动态 routed Expert。

GB10 复验达到 484 lowerable / 7692 emitted，证明除动态 MoE 外的 484 个真实 materialized
request 均已具备 XOpenNPUX 指令表达。最后 40 个 routed Expert 只有 input、runtime expert
IDs、route weights 和 output，不携带被选 expert 的静态权重 operand；它们不能伪装成普通
GPTQ Expert 展开。现新增模型无关 `TROUTED_EXPERT` 控制命令，显式携带上述四个 tensor、
rows/hidden/intermediate/active-experts、量化配置和 executable weight-plan command ID。审计会
继续单独输出 `host_fused_requests=40`：预计 lowering 结果为 524/524 和 7732 emitted，但这只
表示命令 ABI 已覆盖完整执行图。

随后 routed Expert 数值入口也已从 Host graph fused branch 迁到 C++ NPU functional command
engine：`Gem5HostFunctionalGraph` 只 materialize 请求并生成 `TROUTED_EXPERT`，engine 按
EXTMEM-relative offset 校验 input/IDs/route weights/output 和量化配置，再通过 weight-plan
command ID 调用当前功能分页 provider。该实现仍是 functional modeling，不是 RTL，但已建立
后续替换接口边界。下一轮 GB10 审计预期 `observed=524`、`lowerable=524`、`host_fused=0`、
`emitted=7732`、`complete=PASS`；token 严格正确性必须保持不变。设备侧 page queue、并发
expert issue 和 RTL completion aggregation 仍是下一阶段工作。

GB10 已确认完整 lowering 审计达到 `observed=524`、`lowerable=524`、`host_fused=0`、
`emitted=7732`、`complete=PASS`。这组数据成为真实 Qwen3.5-35B prefill 图的正式命令覆盖
基线。为继续推进设备控制语义，functional graph stats 新增 routed-expert command、route
issued 和 route completed 三组计数；成功命令要求每个 `rows * active_experts` route 都完成后
才允许聚合 command completion。后续 page queue 和并发 expert scheduler 必须保持 issued 与
completed 相等，且 token 严格正确性不变。

## 2026-08-31 XOpenNPUX Primitive Execution Replacement

完整 lowering 覆盖并不代表数值计算已经经过 NPU 协处理器。为开始替换 Host C++ kernel，
新增 `Gem5HostXGraphExecutor`：它把真实 materialized generic request 交给统一 XGraph
lowering，再将 64-byte XGraph command 转换为 32-bit XOpenNPUX 指令和 custom CSR 快照，
最终由 `Gem5XOpenNpuFunctionalCoprocessor` 的 L2 decode 与执行语义完成读、算、写回。该路径
不复制算子算法，也不使用 legacy `NPU_LAUNCH` shortcut。

首阶段通过 `OPENNPUX_HOST_FUNCTIONAL_EXECUTION=xopennpux-primitives` 显式启用，替换
`TADD/TMUL/TRMSNORM/TSOFTMAX/TROPE/TSILU/TSIGMOID/TGATHER/TTOPK/TDMA/`
`TROW_SCALE`。只有 lowering 成功、opcode 属于该集合且所有输入/输出/辅助地址完整落在
Tensor Arena 时才执行；GPTQ、dense TMMA、attention 和 routed expert 等涉及外部权重页或
复合 scratch 的请求继续走严格正确性 Host backend。开始协处理器执行后发生的 decode、CSR
或地址错误属于硬错误，禁止静默 fallback。

逻辑图统计继续以 generic request 为单位，保证现有 `524 commands/step` 验收口径不变；新增
`host_functional_xgraph_requests/commands/operations/modeled_cycles/fallback_requests`
单独报告物理替换覆盖率。默认不开启该策略，因此现有 Host C++/vLLM strict token 基线不受
影响。下一阶段按相同接口加入 tiled FP32 TMMA，再加入 TDEQUANT+TMMA GPTQ 序列和 device
routed expert，逐步把 fallback 降到零。

GB10 端到端复验命令：

```bash
CORAL_HOST_FUNCTIONAL_EXECUTION=xopennpux-primitives \
./tools/coralnpu/run_qwen35b_real_weights_test.sh \
  --prompt "OpenNPUX heterogeneous inference" --max-new-tokens 4
```

验收要求除原有 `token_golden=PASS equivalence=strict` 外，还必须出现非零
`host_functional_xgraph_requests/commands`；`fallback_requests` 作为下一阶段替换清单保留，
不能把 `xgraph_audit_complete=PASS` 当作已经执行过硬件协处理器。

GB10 首阶段结果为 4 个 generation step 累计 `xgraph_requests=160`、
`xgraph_commands=160`、`xgraph_operations=860160`，并保持 strict token PASS；
`fallback_requests=324`，即每步仍有 81 个复杂请求未替换。第二阶段已加入统一 EXTMEM
执行镜像：本次请求引用的外部权重区按 64-byte 对齐映射到临时 NPU 地址空间，Tensor Arena
保持原地址，后接 64 MiB lowering scratch。通用 lowerer可将普通 GPTQ MatMul 拆成
`TDEQUANT/TMMA/TADD` 序列，执行器在任何写回前验证完整序列，再逐条经过相同 L2 decode、
CSR snapshot 和 functional coprocessor，最终只将 Tensor Arena 区域发布回图状态。

新增 `host_functional_xgraph_fallback_opcode_<n>` 统计用于区分尚未替换的 QKV 多投影、
Expert 和其他复合请求。预计普通投影接入后，真实模型每步 81 个 MATMUL fallback 将先下降，
但 GPTQ gated QKV 仍需独立的多输出 lowering；任何 strict token 偏差均阻止该阶段验收。

GB10 已验证普通 GPTQ projection sequence 和外部 RMSNorm weight staging：累计
`xgraph_requests=282`、`xgraph_commands=1510`、`fallback_requests=202`，其中 81 个
NORMALIZE 已全部通过 `TRMSNORM` 执行，strict token 保持 PASS。剩余 fallback 分布为
EMBED 1、QKV MATMUL 40、TOPK 1、CAUSAL_CONVOLUTION 30、RECURRENT_UPDATE 30、ROUTER
40、EXPERT 40、DMA 10、ATTENTION 10。

下一批把已有 functional coprocessor 后端接到 Host executor：stateful causal convolution
经 `TCAUSALCONV`，gated-delta recurrent update 经 `TRECURRENT`，causal GQA attention 经
`TATTENTION`。三条路径均补齐范围校验、custom instruction 编码、扩展 CSR packet、流量/
cycle 统计和 tensor/state 回写回归。GB10 预期再消除 opcode 10/11/15 共 70 个 fallback；
验收仍要求 strict token 完全一致。

GB10 已完成上述三条 stateful 路径验收：累计 `xgraph_requests=352`、
`xgraph_commands=1580`、`xgraph_operations=904494592`、
`fallback_requests=132`，opcode 10/11/15 已从 fallback histogram 消失且 strict token
继续 PASS。剩余 fallback 为 EMBED 1、gated QKV MATMUL 40、TOPK 1、ROUTER 40、
EXPERT 40 和 KV DMA 10。

Host executor 随后接入通用 KV-cache DMA lowering。一个 generic DMA request 原子展开为
`TDMA(K) + TDMA(V)`，按照 `rows/kv_heads/head_dim/kv_length` 计算两个 state plane 的
尾部写入范围；本地回归独立验证 K/V 数据、地址边界、2 条物理命令和 8 个 functional
cycles。

外部词表权重 EMBED 和 indices-only TOPK 也已接入同一执行器。EMBED 将只读词表页 staging
到 EXTMEM 后执行 `TGATHER`；TOPK 由 batch lowerer 分配内部 values scratch，只向可见
Tensor Arena 发布 token indices。独立回归验证 1 条命令的 embedding 行选择、逐行 Top-1
index、地址边界和 functional cycles。合并三项后，下一轮 GB10 预期达到
`xgraph_requests=364`、`xgraph_commands=1602`、`fallback_requests=120`，opcode 1、8、
14 全部消失。

GB10 已确认该验收目标：累计 `xgraph_requests=364`、`xgraph_commands=1602`、
`xgraph_operations=904558592`、`xgraph_modeled_cycles=904558592`、
`fallback_requests=120`，strict token equivalence PASS。fallback histogram 仅剩
QKV MATMUL opcode 2、ROUTER opcode 12、EXPERT opcode 13 各 40。

下一增量将 GPTQ ROUTER 接入 XOpenNPUX executor。lowering 将 scratch 分为 logits、
TopK packed values/indices 和 GPTQ dequant tile 三个不重叠区间，先执行通用 tiled
`TDEQUANT/TMMA/TADD` 投影，再原子执行 `TTOPK + TSOFTMAX + 2xTDMA`。模型包回归已验证
route 权重逐行归一化、expert IDs、物理命令计数增加且不产生 fallback。GB10 验收预期为
`xgraph_requests=404`、`fallback_requests=80`、opcode 12 消失并保持 strict token PASS；
真实 commands/operations/cycles 由 35B tile plan 实测记录，不在本地小模型上外推。

首次 GB10 preflight 进一步表明真实 Router 使用 FP32 `[2048,256]` 权重，而不是 GPTQ，
单条 TMMA 无法编码 `K=2048`。浮点分支已改为复用通用 tiled dense MatMul lowerer；新增
`rows=2/K=2048/N=4/top_k=2` 回归验证多条 TMMA、合法 expert IDs 和逐行归一化权重。

## 2026-09-03 TVM BYOC XGraph Codegen 第一阶段

模型编译入口开始从模型特定 execution-plan 脚本迁移到 TVM Relax + BYOC。新增独立
`OPENNPUX_TVM_BYOC_GRAPH_V1` 边界、静态 shared-DMA Tensor arena planner 和 XGraph v2
Codegen；输出 `.npxg` 直接使用现有 96-byte header 与 64-byte command ABI，不新增另一套
firmware 协议。首批支持将静态 FP32 `matmul/add/multiply/rms_norm/softmax/silu/take` BYOC
区域映射到 `TMMA/TADD/TMUL/TRMSNORM/TSOFTMAX/TSILU/TGATHER`，规范化边界还可编码
`TTOPK/TROPE/TDMA`。

编译器对 dynamic shape、未定义 broadcast、非末轴 softmax、超过单条 TMMA 10-bit shape
范围和地址重叠执行硬拒绝。GPTQ、attention、routed expert 等多命令分解仍由现有
`opennpux_npu_xgraph_lower_batch()` 作为唯一语义实现；后续通过符号 generic request 与运行期
地址物化衔接，而不是在 Python Codegen 中复制切片算法。当前单测已核对 Python 编码与 C ABI
均为 header 96 bytes、command 64 bytes，并验证五命令 BYOC fixture 的 opcode、shape、输出
地址与 command ID。

真实 Apache TVM 0.24.0 路径也已完成本机端到端验收。测试构造实际 Relax `IRModule`，通过
`FuseOpsByPattern + MergeCompositeFunctions` 合并 `matmul -> add -> silu -> softmax` 为单个
`Codegen=opennpux` 区域，再提取为稳定边界并生成 4 条
`TMMA/TADD/TSILU/TSOFTMAX` XGraph v2 command。ABI 级 C consumer 从二进制重新读取正式
header/command，独立填充输入、权重和 bias，执行命令并与直接计算结果逐元素比较，误差门限
为 `1e-6`。这解决了真实 TVM Composite 局部变量重名以及每个 primitive 被错误拆成独立
Codegen region 的问题。

新增 `setup_tvm_byoc_env.sh` 固定源码版本和构建参数，避免误装 PyPI 上同名但非 Apache TVM
的 `tvm` 包。当前闭环覆盖“真实 Relax -> BYOC -> XGraph ABI -> 数值执行”；Linux Guest 到
Coral firmware 的系统验收仍保持 executable 与 invocation 分离，下一步由 runtime 将
`.npxg` command 和动态 Tensor bindings 分别放入 shared DMA window 后提交，不能把测试数据
固化进模型编译产物。

## 2026-09-04 TVM BYOC Guest/Coral Runtime 接入

真实 TVM Relax 测试已在 Linux 环境通过，输出 `xgraph_commands=4`、
`xgraph_arena_bytes=131520`、`tvm_relax_byoc_e2e=PASS`。在此基础上新增通用
`xgraph-run <graph.npxg> <arena.bin>` Runtime 接口，不再由测试代码手工构造命令。
`.npxg` 保存可复用 XGraph header/commands，独立 arena 保存本次 invocation 的 input、
constant 和 state；Runtime 分别写入 shared DMA window 的 `0x10000` 命令区与 `0x20000+`
Tensor 区，Coral firmware 回填 state/error/completed commands/output checksum/operations/cycles。

新增 artifact loader 对 magic/version/header size/command size/command ID/命令区重叠及
Tensor arena 基础地址边界做硬校验；新增 arena builder 根据 Codegen metadata 分配表严格
检查 Tensor 名称、dtype、元素数和 byte range。主机数值路径与 Guest 路径使用同一个
arena，当前参考输出 checksum 为 `0xbcd03dc5`。

本机已通过 Python 8 项 Codegen 单测、真实 TVM Relax -> BYOC -> 4-command XGraph、C 数值
执行、artifact runtime loader 和 runtime host tests。新增
`run_tvm_byoc_xgraph_test.sh` 用于 x86/GB10 全系统验收：自动注入 `.npxg`、arena 和当前
`coralctl` 到 checkpoint，要求 `TMMA/TADD/TSILU/TSOFTMAX` 均经 Coral firmware 的
XOpenNPUX 通路完成，并要求 Guest 输出 checksum 与主机参考一致。

GB10 全系统验收已通过。Guest 从 checkpoint 加载与当前 `vmlinux` 匹配的
`opennpux_coral.ko`，通过 `/dev/opennpux-coral` 将 artifact 与 arena 发布到 8 MiB
shared DMA window。Coral firmware 完成 4 条 command，返回 24-byte 输出、
`operation_count=72`、`modeled_cycles=72` 和 `output_checksum=0xbcd03dc5`；checksum
与 Host C 独立参考一致，`xgraph_artifact_run=PASS` 与 `tvm_byoc_xgraph=PASS`
均已确认。至此该阶段完成了“Relax -> BYOC -> XGraph -> Guest Runtime ->
Coral firmware -> XOpenNPUX -> 结果回传”的真实系统闭环。

## 2026-09-04 TVM 大矩阵复用统一 C Lowering

TVM Codegen 不再把超过 XOpenNPUX 指令 10-bit 维度字段的 MatMul 直接拒绝。新增稳定 C ABI，
将 Relax 标准 `[K,N]` RHS 布局显式传入已有 runtime lowering，由同一个 C 实现完成地址范围校验、
K 维切片、stride 编码和 destination accumulate，避免 Python 编译器与 Guest runtime 维护两套
切片规则。原有模型 runtime 使用的 `[N,K]` 布局继续通过显式 transpose 标志走同一实现。

真实 TVM 回归现使用 `[2,2048] x [2048,8]` MatMul。一个 Relax 节点被拆成 K 维
`1023 + 1023 + 2` 的三条 TMMA，再接 `TADD/TSILU/TSOFTMAX`，因此区域包含 4 个图节点、
6 条硬件命令。独立直接计算得到 64-byte reference、checksum `0x40f42b1d`；ABI C consumer
的分片执行结果与 reference 最大绝对误差为 `1.60336494e-05`。本地 Codegen、C ABI、
artifact loader、runtime host
和真实 TVM 数值测试均通过；下一项系统验收是在 GB10 上确认 Guest/firmware 完成 6 条命令并
以 `5e-5` 绝对误差门限通过逐元素 reference 比对。

首次 6-command GB10 执行已完成全部命令，但 firmware 输出 checksum 为 `0x8f84ff19`，与
Host 分片执行的旧 checksum 不同。该失败暴露的是验收缺陷：FP32 长 K 累加以及 SiLU/Softmax
跨 C/C++ 优化边界不能用逐字节 checksum 作为数值等价条件。测试现将独立 reference 写入
invocation arena，Runtime 在 NPU 完成后逐元素检查 finite 和最大绝对误差；实际 checksum
继续输出用于复现和诊断，但不再替代数值正确性判断。

显式双向同步与回读硬校验后的 GB10 验收已经通过。Runtime 在 launch 前将完整 invocation
arena 从 Shared DMA Window 发布到 Local EXTMEM，完成后分别同步 header 与 output；设备生成
和 Host 回读 checksum 均为 `0xaeedc3a1`。独立 reference checksum 为 `0x40f42b1d`，逐元素
最大绝对误差 `1.60336494e-05`，低于 `5e-5` 门限。Firmware 完成 6 commands，记录
`operation_count=32896`、`modeled_cycles=32896`，排除了预置 reference 造成的假 PASS。

## 2026-09-04 TVM BYOC 多 Region 编译边界

新增 `OPENNPUX_TVM_BYOC_MODULE_V1`，不再要求分区后的 Relax 模型恰好只有一个完整下沉
region。`compile_tvm_byoc_module.py` 为每个 region 生成独立 `.npxg`，并生成
`module.npxgm.json`，记录确定性的拓扑执行顺序、外部输入、输出和直接 NPU-to-NPU Tensor
binding。编译器对跨 region shape/dtype、output-to-input 方向、单一生产者和 DAG 无环性执行
硬校验。

真实 TVM 回归新增 `add -> Host relu -> silu` 图，用不支持的 Host `relu` 验证 BYOC 不会把
两个 NPU region 错误合并。当前阶段完成的是 compiler/scheduler contract；下一增量是在
Guest runtime 中消费 module manifest，依次提交 `.npxg`，并在 Host region 与 NPU region
之间绑定 Tensor，而不是将 Host 调度策略固化进单个设备 artifact。

GB10 上真实 TVM 多 region 编译验收已通过。随后新增 model-independent
`ModuleRuntime` 调度核心：为每个 region 分配独立 arena，提交前检查全部外部
input/constant/state binding，按 manifest edge 将生产者 output 精确复制到消费者 input，并
按拓扑顺序调用可注入 executor。当前 16 项单测覆盖两 region `TADD -> TSILU` 数值链、执行
顺序、中间 Tensor 传递、缺失 binding、类型不匹配、多生产者和环。

现已补充 `CoralCtlExecutor`，将上述调度接口接到真实 `coralctl xgraph-run`。Runtime 为每个
region 写入独立 artifact/arena，通过 `OPENNPUX_XGRAPH_OUTPUT_PATH` 取回经过 Local EXTMEM
同步及 firmware checksum 校验的输出，再按 manifest edge 写入下一个 region 的 input range。
全系统测试在原 6-command 单 region 验收后继续提交两个独立 region，执行
`TADD -> 32-byte Tensor edge -> TSILU`；预期新增
`xgraph_module_regions_completed=2`、`xgraph_module_tensor_edges=1` 和
`xgraph_module_chain=PASS`。这验证的是跨两次设备提交的数据依赖闭环，而不是 Python 内存回调。

首次 GB10 多 region 运行确认 region 0 的设备执行、EXTMEM 回读与 reference 比较均通过，但
checkpoint 中注入的精简 BusyBox 不提供 `dd` applet，导致中间 Tensor 文件绑定失败。测试现
优先使用 Guest coreutils `dd`，仅在确认 BusyBox 支持该 applet 时回退；复制采用 32-byte
Tensor block seek，写后校验 region 1 arena 大小，并在失败时保留原始 stderr。

修复后 GB10 验收通过：region 0 的 firmware/output readback/reference checksum 均为
`0x644b35ab` 且最大绝对误差为 0；两个 region、一个 Tensor edge 全部完成，最终输出
`xgraph_module_chain=PASS`，末级记录 `operation_count=24`、`modeled_cycles=24`。

进一步新增通用 Host binding resolver，处理真实 TVM 图中两个 NPU region 之间不能下沉的
Host 算子。Resolver 只接收 region/tensor 标识和已验证的生产者输出，不把模型或 ReLU 等语义
写入 NPU 调度器。第 17 项单测以包含负数的 `TADD -> Host ReLU -> TSILU` 验证 Host 变换确实
生效，避免将 Host 边界错误退化为 NPU-to-NPU memcpy。

混合图契约继续从运行时回调提升为编译产物的一部分。`module.npxgm.json` 新增可选
`host_bindings`，记录 source/destination Tensor、字节数和有序 Host operation pipeline；它与
直接 edge 共用 shape/dtype、单生产者和 DAG 校验。Relax 提取器会沿 main function 中的 Host
一元调用追踪 NPU producer，不再把 `add -> Host relu -> silu` 中的 `relu` 丢失为外部输入。
Runtime 必须注入 `host_executor` 才能消费该 binding。当前 19 项本地测试通过；下一次真实 TVM
验收必须确认 2 个 NPU region、0 条 direct edge 和 1 条 `relax.nn.relu` Host binding。

Guest 验收现已从直接 `TADD -> TSILU` 升级为真实混合链
`TADD -> CPU ReLU -> TSILU`。新增 `coralctl host-tensor-unary` 作为首个通用 Host Tensor
pipeline 执行入口，当前注册 FP32 ReLU；输入刻意使 TADD 产生负值，若跳过 ReLU 或只 memcpy，
TSILU reference 比较必然失败。新验收输出必须包含 `host_tensor_run=PASS`、
`xgraph_module_direct_edges=0`、`xgraph_module_host_bindings=1` 和
`xgraph_module_host_pipeline=relax.nn.relu`。

修正后的 GB10 验收进一步补齐 driver 级 Shared DMA Window 与 Local EXTMEM 双向显式同步，
并在数值比较前强制验证 firmware checksum 与回读 checksum 一致，从而排除直接比较 shared
window 中预置 reference 的假阳性。最终 6 条命令全部完成，设备与回读 checksum 均为
`0xaeedc3a1`，独立 reference checksum 为 `0x40f42b1d`，逐元素最大绝对误差
`1.60336494e-05 < 5e-5`，`operation_count=32896`、`modeled_cycles=32896`，Guest、artifact
和系统验收全部 PASS。该结果形成后续多 BYOC region、动态 binding 与复合算子接入的可信基线。

最新 GB10 验收已证明真实混合拓扑可执行：region 0 的 TADD 对含负数输入计算并以
`0x119b1ae5` 完成设备回读校验，CPU Host pipeline 对 8 个 FP32 元素执行 ReLU，region 1
的 TSILU 输出回读 checksum 为 `0x4983e4f0`，相对独立参考最大绝对误差仅
`2.38418579e-07`。最终统计为 2 个 NPU region、0 条 direct edge、1 条 Host binding，
`xgraph_module_chain=PASS`。

在此基础上新增通用 `run_tvm_byoc_module.py`：直接读取 `module.npxgm.json`，通过
`--bind region.tensor=file` 绑定 invocation Tensor，按 manifest 拓扑调用
`coralctl xgraph-run`，解释 Host operation pipeline，并将完成设备回读验证的 module output
写入指定目录。首个 Host registry 支持 FP32 `relax.nn.relu`，未知算子、dtype 或尺寸不匹配
均硬失败。该 Python 实现是调度契约和 Host OS 参考 runtime；后续仍需实现等价静态 C Guest
runtime，不能把 Python 作为最终 checkpoint 依赖。

TVM 端到端验证进一步消除了测试脚本编排。新增 version 1 二进制 module invocation ABI，
包含 region、direct edge、Host binding/operation、module output 记录以及对齐后的 `.npxg` 与
arena payload；`build_tvm_byoc_module_package.py` 根据编译 manifest 和 invocation Tensor 构包。
静态 Guest `coralctl xgraph-module-run` 会校验全部索引与地址范围，自动执行 binding、Host
pipeline、逐 region 固件提交及设备输出回读，并汇总 command、operation 和 modeled cycle。

全系统脚本同时修正了一个旧验收缺口：原脚本实际使用 dependency-free fixture 的 `module/`
目录，再手工插入 ReLU；新脚本改为直接打包 TVM Relax 产生的 `multi-region-module/`，Guest
仅执行一个 module 命令，不再包含固定 region 名、Tensor offset、`dd` 或显式 ReLU 调用。
本地 22 项 BYOC 测试、package ABI 测试、严格 C11 编译和 shell 语法检查已通过；真实 TVM、
AArch64 静态链接与 gem5 指令日志闭环等待 GB10 全系统验收。

GB10 随后完成上述静态 Guest runtime 验收：两个 NPU region 和一个 Host ReLU 由单次
`xgraph-module-run` 自动调度，累计完成 2 条设备命令、32 次 operation 和 32 个 modeled
cycle，最终 output checksum `0x4983e4f0` 与 region 1 回读一致。下一增量补齐多输出导出：
兼容变量 `OPENNPUX_XGRAPH_MODULE_OUTPUT_PATH` 继续写 output 0，新变量
`OPENNPUX_XGRAPH_MODULE_OUTPUT_PREFIX` 按索引写出全部 output，并报告 output 数量和总字节数。

进一步将编译 module 与运行时输入拆分。`build_tvm_byoc_module_package.py
--clear-external-bindings` 生成可复用基础 `.npxgm`，新 `build_tvm_byoc_invocation.py` 将一次请求
的 external input/constant/state 打包为版本化 `.npxmi`。Guest 通过
`OPENNPUX_XGRAPH_MODULE_INVOCATION_PATH` 加载 overlay，在启动任何 region 前统一校验 binding
索引、目标范围、payload 范围和 checksum，再覆盖 arena。旧的内嵌 arena 调用保持兼容；GB10
验收改为清零基础输入并要求两条动态 binding 生效，避免继续依赖打包时残留输入。

动态调用验收继续扩展为同一 Guest 启动内复用同一个清零后的 `.npxgm`，依次加载两份包含不同
输入的 `.npxmi`。两次运行都必须应用 2 条 binding 并完成相同的 NPU/Host 拓扑，同时导出的
32-byte 结果必须不同。该检查用于发现 arena 没有被新请求覆盖、错误复用第一次输出或仍从基础
包读取静态输入的问题；算子数值正确性仍由独立单图 reference 路径负责。

端到端协议增加确定性的 module identity：基础 `.npxgm` 与 `.npxmi` 都保存由 canonical
compiler manifest 计算的 32-bit identity，Guest 必须匹配后才能应用任何 binding。全系统负向
验收先篡改 invocation identity 并要求 `xgraph_module_identity_rejection=PASS`，随后继续执行两次
合法请求，以确认错误请求既被拒绝也没有污染后续 arena 或设备状态。

GB10 最终验收已通过。篡改 identity 的 invocation 在设备提交前被拒绝并输出
`xgraph_module_identity_rejection=PASS`；同一个可复用 `.npxgm` 随后连续消费两份合法 `.npxmi`。
第一次调用的 region 0/1 checksum 分别为 `0x119b1ae5` 和 `0x4983e4f0`，第二次分别变为
`0x9ab31725` 和 `0xcf783ff6`，证明动态 binding 覆盖了基础 arena，第二次执行没有读取第一次的
输入或输出残留。两次调用均完成 2 个 NPU region、2 条设备命令、1 个 Host operation、2 条
invocation binding 和 1 个 32-byte module output，累计 operation/cycle 均为 32；最终输出
`xgraph_module_reused_invocations=2`、`xgraph_module_reuse=PASS`、
`xgraph_module_chain=PASS` 和 `tvm_byoc_xgraph=PASS`。

至此，TVM Relax/BYOC 编译、XGraph artifact、静态 Guest module runtime、CPU Host fallback、
动态 invocation、Shared DMA/Local EXTMEM 同步、设备回读、独立数值参考、身份校验和模块复用已
形成完整功能闭环。该结论只关闭当前 mixed-graph 基础验收；任意模型前端覆盖、更多 BYOC
legalization/算子、动态 shape/多输出复杂拓扑以及将 C++ 功能 kernel 逐步替换为 NPU RTL，仍属于
后续阶段。

下一增量已补齐复用测试的独立数值门禁。新增模型无关的
`coralctl tensor-compare-fp32`，严格校验 Tensor 文件尺寸、FP32 finite 值和可配置绝对误差；
全系统脚本分别生成两次 invocation 的独立 `TADD -> Host ReLU -> TSILU` 参考输出，并要求两份
设备回读结果都在 `5e-5` 内，而不再只判断两个 checksum 不同。本地 22 项 BYOC 回归、严格 C11
编译和 shell 语法已通过；等待 GB10 确认两次 `tensor_compare_fp32=PASS` 后关闭该门禁。
