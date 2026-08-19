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
`CORAL_SIM_HOST_PAGING=1 ./tools/coralnpu/run_qwen35b_real_weights_test.sh`，预期输出
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

真实数值结果通路由单次 forward 扩展为最多 32 token 的贪心自回归解码，默认生成
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
