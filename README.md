# gem5 Coral x86 Superproject

This repository is the x86-oriented superproject for system-level Coral NPU
simulation work.

## Quick Start

Set the path to your gem5 ARM disk images and kernels once per session:

```sh
export IMAGE_PATH="${IMAGE_PATH:-$HOME/wlk/gem5_arm_linux_images}"
```

If your images live elsewhere, set `IMAGE_PATH` explicitly. The examples below use `$IMAGE_PATH` for all disk image paths.

The gem5 run script (`run_multicore.sh`) also respects `IMAGE_PATH`, and most test wrappers accept `CORAL_DISK_IMG` and `CORAL_KERNEL_IMAGE` overrides.

## Layout

- `thirdparty/gem5`: pinned gem5 submodule
- `thirdparty/coralnpu`: pinned official Coral submodule
- `sim/gem5`: gem5-side local source deltas, stored with the same directory
  structure as upstream gem5
- `sim/coralnpu`: Coral-side local source deltas, stored with the same
  directory structure as upstream Coral
- `runtime/host`: host-side bootscripts and runtime bring-up assets
- `runtime/npu`: NPU-side program placeholders and notes
- `rtl/wrappers`: wrapper code and integration shims for RTL co-simulation
- `docs`: ADRs, design notes, and runbooks
- `tools/coralnpu`: Coral standalone validation helpers

## Workflow

1. Update submodules to the pinned commits recorded in
   `thirdparty/PINNED_COMMITS.md`.
2. Keep local gem5 and Coral modifications under `sim/gem5/` and
   `sim/coralnpu/` using the same relative paths as their upstream trees.
3. Merge `sim/gem5` into `thirdparty/gem5` and `sim/coralnpu` into
   `thirdparty/coralnpu` before compilation.
4. Use the runbooks under `docs/runbooks` to validate baseline system flows on an x86 Linux host.

## Team Collaboration

This project is intended for multi-developer work with separate Codex accounts.
Do not share one working directory or one branch between multiple people. Each
developer should claim a GitHub Issue, create an independent branch or worktree,
keep Codex scoped to that Issue, and merge through Pull Request review.

Start here before development:

- `AGENTS.md`: repository-wide Codex rules, ownership boundaries, and validation
  expectations.
- `sim/gem5/AGENTS.md`: gem5 overlay rules.
- `sim/coralnpu/AGENTS.md`: CoralNPU overlay rules.
- `docs/process/team_collaboration.md`: team workflow, design review, branch,
  PR, and ownership process.
- `docs/process/task_template.md`: standard Issue/Codex task template.
- `docs/process/review_checklist.md`: human and Codex review checklist.

Shared interfaces such as SoC memory map, NPU CSR/MMIO, shared DMA window,
TCB/operator descriptors, kernel UAPI, Verilated bridge C ABI, checkpoint flow,
and operator semantics must be documented and reviewed before merge.

## Guest MMIO Tool

The Coral bring-up scripts need a guest-side MMIO helper because `dd` reads from
`/dev/mem` can fail on device memory with `Bad address`. On the x86 Linux host:

```sh
tools/guest_tools/build_mmio32.sh
sudo tools/guest_tools/install_mmio32_to_image.sh \
  $IMAGE_PATH/ubuntu-18.04-arm64-docker.img
```

Rebuild the boot checkpoint after modifying the disk image so Linux sees the
updated contents.

The userspace control utility can be built and installed similarly:

```sh
tools/guest_tools/build_coralctl.sh
sudo tools/guest_tools/install_coralctl_to_image.sh \
  $IMAGE_PATH/ubuntu-18.04-arm64-docker.img
```

Use `coralctl info` to inspect the active backend and `coralctl run` to start
the staged firmware and wait for completion.
Use `coralctl dma-test` with the DMA smoke firmware to verify coherent
shared-memory reads and writes.

## Coral RTL Bridge

On the supported x86 Linux host, build the official `CoreMiniAxi` Verilator
model and the OpenNPUX C ABI adapter:

```sh
./tools/coralnpu/prepare_coral_bazel.sh
./tools/coralnpu/check_rtl_bridge_abi.sh
./tools/coralnpu/build_rtl_bridge.sh
```

If the x86 host cannot resolve `releases.bazel.build`, download
`bazel-8.6.0-linux-x86_64` on another machine and install it with:

```sh
BAZEL_BINARY=/path/to/bazel-8.6.0-linux-x86_64 \
  ./tools/coralnpu/prepare_coral_bazel.sh
```

The staged RTL bridge artifacts are:

```sh
build/coralnpu/libcoralnpu_gem5_bridge.so
build/coralnpu/gem5_smoke_halt.elf
build/coralnpu/gem5_dma_smoke.elf
build/coralnpu/gem5_command_smoke.elf
```

Apply the gem5 overlay and run the existing full-system flow with the RTL
backend:

```sh
./sim/gem5/apply_patchset.sh
CORAL_NPU_BACKEND=verilated-coral ./thirdparty/gem5/run_multicore.sh
```

The default remains `stage-a`. Set `CORAL_RTL_TICK_PERIOD` and
`CORAL_RTL_CYCLES_PER_EVENT` to tune the RTL scheduling quantum.

The boot checkpoint is stored at `checkpoint/coralnpu_ckpt` relative to the
superproject root. Pulling code or changing the injected resume script does
not rebuild it. Use `CORAL_REBUILD_CKPT=1` only when the booted guest state
must be recreated.

## Command Submission Runtime

Build the bridge and command firmware, rebuild/install `coralctl`, and run the
driver-backed multi-buffer command test:

```sh
./tools/coralnpu/build_rtl_bridge.sh
./tools/guest_tools/build_coralctl.sh
sudo ./tools/guest_tools/install_coralctl_to_image.sh \
  $IMAGE_PATH/ubuntu-18.04-arm64-docker.img

cd thirdparty/gem5
CORAL_KERNEL_IMAGE=../../build/kernel/vmlinux-4.19.325-opennpux \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
CORAL_REBUILD_CKPT=1 \
./run_multicore.sh
cd ../..

CORAL_KERNEL_IMAGE=./build/kernel/vmlinux-4.19.325-opennpux \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
./tools/coralnpu/run_command_test.sh
```

The acceptance result is `vector_add=PASS`, `completed_elements=16`, and
`output_checksum=0x00000198`.

## General Model Packages

Real Hugging Face safetensors models use the versioned, externally sharded
`OPENNPUX_MODEL_PACKAGE_V2` format. Import validates metadata without copying
or loading weight payloads:

```sh
./tools/models/prepare_hf_model_package.sh /data/models/Qwen3.5-35B
```

See `docs/runbooks/qwen35b_model_package.md`. The deterministic
`qwen-tiny.npxm` remains the golden control/TCB test and is not a pretrained
model.

## Heterogeneous Platform

The generic model runtime and custom RTL accelerator use the same driver and
command ABI as the official Coral path. Build and install the sample assets:

```sh
./tools/coralnpu/build_rtl_bridge.sh
IMAGE=$IMAGE_PATH/ubuntu-18.04-arm64-docker.img \
  ./tools/coralnpu/prepare_guest_assets.sh
```

After rebuilding the boot checkpoint once, run:

```sh
CORAL_KERNEL_IMAGE=./build/kernel/vmlinux-4.19.325-opennpux \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
./tools/coralnpu/run_model_test.sh
```

See `docs/runbooks/heterogeneous_platform_acceptance.md` for the model and explicit
official/custom RTL acceptance criteria.


## 当前SOC memmap

当前 D9200 / D9300 全系统脚本基于 gem5 `VExpress_GEM5_V1` 平台，并在此基础上增加 Coral NPU aperture。默认内存大小是 `4GiB`，DRAM 从 `0x80000000` 开始。

| 地址范围 | 归属 | 用途 |
| --- | --- | --- |
| `0x00000000-0x03ffffff` | VExpress off-chip CS0 | Flash / boot 相关保留窗口 |
| `0x08000000-0x0bffffff` | VExpress off-chip CS0 | 默认 bootloader 装载区域，`boot.arm64` 常用 |
| `0x0c000000-0x0fffffff` | VExpress off-chip CS0 | 保留窗口 |
| `0x10000000-0x13ffffff` | VExpress off-chip CS4 | gem5 平台扩展外设窗口 |
| `0x14000000-0x17ffffff` | VExpress off-chip CS1 | 保留 / PSRAM 窗口 |
| `0x18000000-0x1bffffff` | VExpress off-chip CS2 | VRAM / 保留窗口 |
| `0x1c000000-0x1fffffff` | VExpress off-chip CS3 | 外设窗口，UART、RTC、VirtIO 等 |
| `0x1c010000-0x1c01ffff` | RealView IO | VE system control registers |
| `0x1c090000-0x1c09ffff` | UART0 | Linux 主串口，命令行使用 `console=ttyAMA0 earlycon=pl011,0x1c090000` |
| `0x1c0a0000-0x1c0cffff` | UART1-3 | 额外串口 / fake UART |
| `0x1c130000-0x1c14ffff` | VirtIO | gem5 VirtIO 扩展窗口，块设备会映射为 guest `/dev/vda*` |
| `0x1c170000-0x1c17ffff` | RTC | guest RTC |
| `0x1d000000-0x1d030fff` | Coral NPU | 当前 NPU MMIO aperture，默认 `--npu-pio-addr=0x1D000000 --npu-pio-size=0x31000` |
| `0x1d000000-0x1d001fff` | Coral NPU ITCM | stage-a 模型默认 ITCM window，大小由 `itcmSize=8KiB` 控制 |
| `0x1d020000-0x1d027fff` | Coral NPU DTCM | stage-a 模型默认 DTCM window，大小由 `dtcmSize=32KiB` 控制 |
| `0x1d030000-0x1d030fff` | Coral NPU CSR | reset / pc_start / status 等 CSR window |
| `0x1d030000` | Coral NPU CSR | `RESET_CONTROL`，bit0 reset，bit1 clock gate |
| `0x1d030004` | Coral NPU CSR | `PC_START` |
| `0x1d030008` | Coral NPU CSR | `STATUS`，bit0 halted，bit1 fault |
| `0x20000000-0x207fffff` | Coral NPU EXTMEM 8M |
| `0x20800000-0x3fffffff` | VExpress on-chip | 、GIC、timer、HDLCD、SMMU、PCI IO/config 等 |
| `0x2c001000-0x2c001fff` | GIC | GIC distributor |
| `0x2c002000-0x2c003fff` | GIC | GIC CPU interface |
| `0x2c1c0000-0x2c1cffff` | GICv2m | MSI frame |
| `0x2f000000-0x2fffffff` | PCI IO | PCI IO space |
| `0x30000000-0x3fffffff` | PCI config | PCI config space |
| `0x40000000-0x7fffffff` | PCI memory | External AXI / PCI memory window |
| `0x80000000-0xffffffff` | DRAM | 默认 `4GiB` 配置下的 guest 物理内存窗口 |

实现位置：

- 平台地址图：`src/dev/arm/RealView.py`
- 系统内存范围：`configs/example/arm/devices.py`
- D9300/NPU 参数：`configs/example/arm/arm_multicore_d9300.py`
- NPU MMIO aperture：`src/dev/npu/NPUDevice.py`、`src/dev/npu/npu_device.cc`
- stage-a NPU 内部 ITCM/DTCM/CSR 分段：`src/dev/npu/coral_stagea_backend.*`

注意：`0x1d000000-0x1d030fff` 选在 VExpress CS3 外设窗口内，避免和 DRAM、GIC、PCI config/mem 空间冲突。若修改 `--npu-pio-addr` 或 `--npu-pio-size`，需要同时检查 `membus/iobus` bridge range，避免出现 “two ports responding within range”。

## RVV Highmem MobileNet

The optional highmem path retains the standard bridge and adds the official
`RvvCoreMiniHighmemAxi` configuration with LiteRT Micro MobileNet firmware:

```sh
./tools/coralnpu/build_rvv_mobilenet.sh
IMAGE=$IMAGE_PATH/ubuntu-18.04-arm64-docker.img \
./tools/coralnpu/prepare_guest_assets.sh
CORAL_KERNEL_IMAGE=./build/kernel/vmlinux-4.19.325-opennpux \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
./tools/coralnpu/run_rvv_mobilenet_test.sh
```

The first run creates a dedicated 8 MiB-window checkpoint; the second restores
it and executes MobileNet. See
`docs/runbooks/rvv_mobilenet_acceptance.md` for the complete procedure.
### Qwen3.5 real-weight paging

The control-only acceptance repeats one synthetic page. To stream the actual
GPTQ shards through the CPU-owned pager, export the prepared model directory to
the guest through the read-only VirtIO 9P device:

```bash
sudo apt-get install diod
CORAL_MODEL_DIR=/data/models/Qwen3.5-35B \
  ./tools/coralnpu/run_qwen35b_real_weights_test.sh
```

The guest kernel must provide `CONFIG_NET_9P`, `CONFIG_NET_9P_VIRTIO`, and
`CONFIG_9P_FS`. The test deliberately fails before NPU launch when 9P is absent;
it never copies the multi-gigabyte model into the boot image or checkpoint.
