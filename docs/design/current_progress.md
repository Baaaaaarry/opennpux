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

GPTQ MatMul 与 `ATTENTION/CAUSAL_CONV/RECURRENT_UPDATE/ROUTER/EXPERT/COMBINE`
暂时明确返回 `ENOTSUP`，等待 decomposition/tiling pass 展开，防止把复合 command
错误伪装成单条自定义指令。native gate `test_xgraph_lowering.sh` 已覆盖直接映射、
RoPE/SiLU/TopK 显式语义、EXTMEM 地址以及 unsupported 路径。

批量 lowering 接口 `opennpux_npu_xgraph_lower_sequence()` 进一步把单条转换扩展为
有序 command stream。接口强制 command ID 与 retirement 顺序一致，并在遇到尚未分解的
复合算子、GPTQ MatMul、非法地址或容量不足时返回首个失败 command 的 index、ID、opcode
和 errno。该接口不跳过 unsupported command，也不插入模型特例，是 524-command
invocation 接入 decomposition/tiling pass 前的可诊断边界。
