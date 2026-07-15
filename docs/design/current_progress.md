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

### Phase 2 completion

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
  outstanding, making the single-outstanding Phase-2 contract explicit.
- Phase-2 acceptance is documented in `docs/runbooks/phase2_acceptance.md`.

### Phase 3

- Replace shell control with a stable userspace runtime and then a Linux driver.
- Allocate and pin shared input, output, command, and completion buffers.
- Add interrupts or an event-driven completion path.
- Define cache maintenance rules for non-coherent configurations.
- Add multiple outstanding AXI IDs if firmware/runtime profiling shows the
  single-outstanding bridge is a bottleneck.

The first Phase-3 increment has split `coralctl` into a reusable host runtime
API (`runtime/host/include/opennpux/coral_runtime.h` and
`runtime/host/src/coral_runtime.c`) plus a CLI frontend. The runtime still uses
the Phase-2 `/dev/mem` backend so existing checkpoint-based tests continue to
work. The next increment is to replace that backend with a minimal Linux
character device while keeping the userspace API stable.

The second Phase-3 increment has added the shared user/kernel UAPI
(`runtime/host/include/opennpux/coral_uapi.h`), runtime transport selection,
and a minimal platform-driver scaffold under `runtime/kernel`. The runtime now
auto-selects `/dev/opennpux-coral` when available and falls back to `/dev/mem`
for the existing checkpoint smoke tests.

The third Phase-3 increment completes the device boundary: the driver exposes
only the DT-reserved shared DMA window through non-cached mmap, negotiates a
versioned capability ABI, accepts asynchronous START, wakes userspace through
poll using a kernel completion worker, and supports RESET after timeout. The
driver-only DMA acceptance covers control ioctl, shared mmap, completion, and
coherent RTL DMA without opening `/dev/mem`.

### Phase 4

- Load a real model and tensors from Linux.
- Integrate a narrow inference runtime API.
- Measure CPU scheduling, memory traffic, NPU execution, and end-to-end latency.

The first Phase-4 increment defines a 64-byte versioned command descriptor and
bounded input/output tensor regions in the shared window. `coralctl vector-add`
now submits a multi-buffer command through the Phase-3 driver, while
`gem5_command_smoke.elf` parses the descriptor on the Coral core and reports
completion. This is the reusable end-to-end contract for subsequent DS4
operators.

The completed platform increment adds a validated `.npxm` model container with
real input tensors, sequential multi-command dispatch, output verification, and
model-level simulated-time reporting. The sample graph mixes official Coral
software execution and custom RTL execution without DS4-specific assumptions.

### Phase 5

- Add the custom RTL accelerator inside the Coral source hierarchy.
- Add instructions or command descriptors, compiler/runtime support, and RTL
  verification.
- Compare official and custom execution using the same Linux workload.

Phase 5 now includes a synthesizable three-cycle custom MAC, standalone
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

1. Run the Phase-3 runtime acceptance suite on x86 Linux.
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
