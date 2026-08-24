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
generic invocation/command/tensor descriptors, kernel UAPI, Verilated bridge C
ABI, checkpoint flow, and operator semantics must be documented and reviewed
before merge. Model-specific TCBs are compatibility tests, not platform ABIs.

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

## Qwen3.5 End-to-End Setup On A New Host

This section is the reproducible path from a fresh x86-64 Ubuntu host to the
real-weight Qwen3.5 end-to-end verdict. The current acceptance path is:

```text
CPU prompt and tokenizer
  -> generic NPU executable/invocation
  -> gem5 NPUDevice and Verilated Coral command processor
  -> Host C++ functional kernels with real GPTQ weights
  -> NPU completion and token IDs
  -> CPU tokenizer and decoded text
```

It validates the heterogeneous SoC control path, command scheduling, weight
paging, functional kernels, completion ABI and CPU/NPU data exchange. It is
not a claim that all Qwen operators execute in full RTL. The Host C++ kernels
are the correctness baseline that will be replaced incrementally by timing
models and RTL execution units without changing the generic invocation ABI.

### 1. Clone Or Update The Repository

Use one working directory and branch per developer. Until the current Qwen
changes are merged into `main`, use `codex_dev`:

```sh
git clone --recurse-submodules \
  --branch codex_dev \
  git@github.com:Baaaaaarry/opennpux.git
cd opennpux
git submodule sync --recursive
git submodule update --init --recursive
```

For an existing checkout:

```sh
git switch codex_dev
git pull --ff-only origin codex_dev
git submodule sync --recursive
git submodule update --init --recursive
```

Verify that the checkout contains the current Qwen options:

```sh
grep -n -- '--thinking-mode' \
  tools/coralnpu/run_qwen35b_real_weights_test.sh
```

An `unknown option: --thinking-mode` error means the host is running an older
branch or checkout.

### 2. Install Host Dependencies

The supported build host is x86-64 Linux. On Ubuntu install the native gem5,
ARM64 cross-build, kernel, 9P and Python prerequisites:

```sh
sudo apt-get update
sudo apt-get install -y \
  git curl wget rsync ca-certificates patch unzip xz-utils \
  build-essential scons pkg-config python3 python3-dev python3-venv \
  gcc-aarch64-linux-gnu libc6-dev-arm64-cross \
  bc bison flex libssl-dev libelf-dev libncurses-dev \
  device-tree-compiler diod \
  zlib1g-dev libprotobuf-dev protobuf-compiler \
  libgoogle-perftools-dev libboost-all-dev libhdf5-dev \
  libpng-dev libcapstone-dev
```

Do not run two gem5 Qwen tests against the same checkout, checkpoint or output
directory. They share UART port `4567`, the 9P export and
`logs/sim/m5out/`.

### 3. Prepare The ARM64 Disk Image

Use an existing known-good Ubuntu 18.04 gem5 image, or download the baseline
assets:

```sh
export IMAGE_PATH="${IMAGE_PATH:-$HOME/wlk/gem5_arm_linux_images}"
./tools/kernel/download_gem5_arm_images.sh

export CORAL_DISK_IMG="$IMAGE_PATH/ubuntu-18.04-arm64-docker.img"
test -s "$CORAL_DISK_IMG"

sudo ./tools/kernel/install_opennpux_init_to_image.sh "$CORAL_DISK_IMG"
```

The multi-gigabyte model is never copied into this image. It is exported
read-only from the host through VirtIO 9P at run time. The installed
`/sbin/opennpux-init.sh` is the stable PID 1 that mounts the minimal guest
filesystems and executes the current gem5 `readfile` resume script. Reinstall
it after pulling changes to `runtime/host/init/opennpux-init.sh`, then rebuild
the checkpoint.

### 4. Build The 9P-Enabled ARM64 Kernel

The official baseline kernel is useful for initial boot, but the real-weight
test requires `CONFIG_NET_9P=y`, `CONFIG_NET_9P_VIRTIO=y` and
`CONFIG_9P_FS=y`. Build the repository kernel:

```sh
./tools/kernel/build_arm64_kernel.sh
./tools/kernel/check_gem5_kernel_config.sh build/linux-arm64/.config

export CORAL_KERNEL_IMAGE="$PWD/build/kernel/vmlinux-$(cat build/kernel/kernel.release)"
test -s "$CORAL_KERNEL_IMAGE"
```

The default `linux-4.19.y` build uses the bundled gem5 4.18 configuration as a
boot-compatible base. Do not replace it with arm64 `defconfig` unless the new
configuration has independently reached the PL011 console. A
`kernel lacks CONFIG_9P_FS` verdict means this step was skipped or a different
kernel was supplied.

### 5. Build The Coral Bridge And Guest Tool

The helper installs the Bazel version pinned by the Coral submodule, applies
the `sim/coralnpu` overlay, builds the bridge and stages the command processor
firmware:

```sh
./tools/coralnpu/prepare_coral_bazel.sh
./tools/coralnpu/check_rtl_bridge_abi.sh
./tools/coralnpu/build_rtl_bridge.sh
./tools/guest_tools/build_coralctl.sh
```

Verify the required artifacts:

```sh
test -s build/coralnpu/libcoralnpu_gem5_bridge.so
test -s build/coralnpu/gem5_npu_command_processor_smoke.elf
test -x build/guest-tools/coralctl-aarch64
```

The first Bazel build downloads pinned toolchains and repositories. On a
restricted network, use Bazel's populated `--distdir` cache or pre-download
the dependencies before running this step. Do not commit the synchronized
dirty state under `thirdparty/coralnpu`; the source of truth remains
`sim/coralnpu`.

### 6. Download And Compile The Model Metadata

The default test model is
`Qwen/Qwen3.5-35B-A3B-GPTQ-Int4`. The weight download is approximately 24 GB;
reserve additional space for metadata, caches and temporary files. Create a
user-writable destination and run the package compiler:

```sh
export CORAL_MODEL_DIR="${CORAL_MODEL_DIR:-/data/models/Qwen3.5-35B}"
sudo install -d -o "$(id -un)" -g "$(id -gn)" "$CORAL_MODEL_DIR"

HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
  ./tools/models/prepare_hf_model_package.sh "$CORAL_MODEL_DIR"
```

The script resumes partial downloads and reuses existing shards. Expected
final verdicts are:

```text
model_package=PASS
qwen_execution_plan=PASS
npu_executable=PASS
npu_weight_plan=PASS
hf_model_package_prepare=PASS
```

At minimum the directory must then contain `model.npxm`, `model.npxe`,
`model.npxc`, `model.npxr`, `model.npxtb`, `execution-plan.npxp`, tokenizer
assets, the safetensors index and every referenced shard.

### 7. Prepare The Numerical Reference Environment

The NPU result is produced by the Host C++ functional backend. vLLM is used
only as an external semantic reference and to diagnose token differences; it
does not replace the result returned by the simulated NPU.

```sh
./tools/models/setup_hf_numerical_env.sh
export CORAL_HF_PYTHON="$PWD/.venv/hf-numerical/bin/python"

if ! "$CORAL_HF_PYTHON" -c 'import vllm' >/dev/null 2>&1; then
  "$CORAL_HF_PYTHON" -m pip install -U vllm
fi

"$CORAL_HF_PYTHON" -c \
  'import torch, transformers, vllm; print(torch.__version__, transformers.__version__, vllm.__version__)'
```

The setup requires a Python interpreter with the `_ssl` extension. If pip
reports `SSL module is not available`, verify both the selected interpreter
and any existing virtual environment:

```sh
python3 -c 'import ssl; print(ssl.OPENSSL_VERSION)'
/usr/bin/python3 -c 'import ssl; print(ssl.OPENSSL_VERSION)'

sudo apt-get update
sudo apt-get install -y ca-certificates openssl python3-full python3-venv

rm -rf .venv/hf-numerical
OPENNPUX_PYTHON=/usr/bin/python3 ./tools/models/setup_hf_numerical_env.sh
```

If `/usr/bin/python3` works but `python3` does not, the latter is normally a
custom interpreter earlier in `PATH`; select `/usr/bin/python3` explicitly as
shown above. If both fail, repair the distribution Python installation. A
Python built from source must be rebuilt after installing `libssl-dev`; pip
cannot install or repair Python's `_ssl` extension.

`transformers>=4.57` also requires Python 3.9 or newer. If pip only lists old
Transformers releases (for example, it stops at `4.46.3`), first check the
interpreter and then bypass a stale configured mirror explicitly:

```sh
/usr/bin/python3 --version

rm -rf .venv/hf-numerical
OPENNPUX_PYTHON=/usr/bin/python3 \
OPENNPUX_PYPI_INDEX_URL=https://pypi.org/simple \
./tools/models/setup_hf_numerical_env.sh
```

If `/usr/bin/python3` is older than 3.9, install a supported Python before
creating the virtual environment. Merely changing the package index cannot
make an incompatible Python install a newer Transformers wheel.

On GB10, prefer the platform-compatible CUDA/PyTorch and vLLM packages already
validated for SM121. CUDA is used only to accelerate reference generation;
gem5, Verilator, AXI and the Host C++ functional backend do not require CUDA.
The prompt-specific `.npxo` reference under `build/model-results/` is reused
on subsequent runs.

### 8. Run The First End-to-End Test

The first run creates a dedicated checkpoint with the custom kernel and 8 MiB
NPU shared window, then automatically restores it and executes the test:

```sh
CORAL_MODEL_DIR="$CORAL_MODEL_DIR" \
CORAL_DISK_IMG="$CORAL_DISK_IMG" \
CORAL_KERNEL_IMAGE="$CORAL_KERNEL_IMAGE" \
CORAL_KERNEL_INIT=/sbin/opennpux-init.sh \
CORAL_HF_PYTHON="$CORAL_HF_PYTHON" \
CORAL_REBUILD_CKPT=1 \
./tools/coralnpu/run_qwen35b_real_weights_test.sh \
  --prompt "Who are you" \
  --max-new-tokens 12 \
  --prompt-format chat \
  --thinking-mode off \
  --decode-mode greedy \
  --model-loader vllm
```

`--thinking-mode off` uses Qwen3.5's official non-thinking chat-template path
for a direct answer. Use `--thinking-mode on` to include reasoning. The mode is
applied consistently to vLLM, the CPU tokenizer and Host C++ input, and is part
of the reference-cache key.

Do not set `CORAL_REBUILD_CKPT=1` for normal repeats:

```sh
./tools/coralnpu/run_qwen35b_real_weights_test.sh \
  --prompt "Explain heterogeneous computing in one sentence." \
  --max-new-tokens 16 \
  --thinking-mode off
```

Rebuild the checkpoint only after changing the disk image, kernel,
`CORAL_KERNEL_INIT`, kernel command line or shared-memory layout. Pulling code,
rebuilding the bridge, changing the prompt or changing the injected resume
script does not by itself require a new boot checkpoint.

### 9. Acceptance And Logs

A structurally successful run ends with:

```text
inference_result_source=host-functional-cpp
inference_token_ids=<IDs returned by the tested NPU path>
inference_text_source=cpu-tokenizer
inference_token_text=<decoded answer>
inference_run=PASS
executable_run=PASS
[coral-qwen35b-real-weights-test] functional_backend=host-cpp
[coral-qwen35b-real-weights-test] PASS
```

`token_golden=PASS` means the generated IDs match vLLM. A numerically close
but different greedy branch is reported as
`token_golden=WARN equivalence=diverged`; it is diagnostic and no longer
invalidates command execution, memory safety or completion acceptance.
Operator failure, invalid memory access, incomplete command count, device
fault and missing output remain hard failures.

Host and Guest logs are stored at:

```text
simout/qwen35b-host-functional-preflight.log
logs/sim/m5out/system.terminal
logs/sim/m5out/stats.txt
```

Monitor the Guest UART without moving the log file:

```sh
mkdir -p logs/sim/m5out
touch logs/sim/m5out/system.terminal
tail -F logs/sim/m5out/system.terminal
```

The interactive UART is available only while gem5 is running and listening:

```sh
cd thirdparty/gem5
./util/term/gem5term localhost 4567
```

If port `4567` is unavailable, check `pgrep -af gem5.opt` and
`ss -ltnp | grep ':4567'`. A completed `m5_exit` closes the port; the complete
serial transcript remains in `system.terminal`.

### Qwen3.5 real-weight paging

The control-only acceptance repeats one synthetic page. To stream the actual
GPTQ shards through the CPU-owned pager, accept a CPU prompt, and return a
modeled next token, export the prepared model directory to the guest through
the read-only VirtIO 9P device:

```bash
sudo apt-get install -y diod gcc-aarch64-linux-gnu build-essential \
  bc bison flex libssl-dev libelf-dev
./tools/kernel/build_arm64_kernel.sh

CORAL_MODEL_DIR=/data/models/Qwen3.5-35B \
  CORAL_KERNEL_IMAGE="$PWD/build/kernel/vmlinux-$(cat build/kernel/kernel.release)" \
  CORAL_REBUILD_CKPT=1 \
  ./tools/coralnpu/run_qwen35b_real_weights_test.sh \
    --prompt "Explain heterogeneous computing in one sentence."
```

The guest kernel must provide `CONFIG_NET_9P`, `CONFIG_NET_9P_VIRTIO`, and
`CONFIG_9P_FS`. `build_arm64_kernel.sh` now enables all three as built-ins, so
no 9P modules need to be installed in the minimal guest image. The test
deliberately fails before NPU launch when 9P is absent; it never copies the
multi-gigabyte model into the boot image or checkpoint. When no explicit
`CORAL_KERNEL_IMAGE` is supplied, the wrapper selects the kernel recorded in
`build/kernel/kernel.release`. A rebuilt kernel invalidates the dedicated
checkpoint; the command above rebuilds it and automatically resumes the test.
The guest mount supplies both the fixed VirtIO mount tag `gem5` and the host
export path as the diod `aname`; these identify the device and exported tree,
respectively.
Pass each test prompt with `--prompt`; this overrides the legacy
`CORAL_QWEN_PROMPT` environment variable. The default run validates the fast
functional-model endpoint: CPU prompt -> generic NPU executable -> token
completion. The NPU returns token IDs; after guest validation, the CPU-side
tokenizer decodes those returned IDs and prints `inference_token_text=<text>`.
Set `CORAL_QWEN35B_NUMERICAL=1` to enable incremental GPTQ
numerical kernels; that stricter mode is complete only when every required
operator has a numerical backend.
