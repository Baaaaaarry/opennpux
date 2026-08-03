# MobileNet 仿真 bring-up 修复全记录（2026-07-31 ~ 2026-08-02）

本文记录 RVV Highmem MobileNet 仿真从"30+ 小时卡死无输出"到
"14 分钟 PASS"过程中发现并修复的全部问题，每个改动点附原理讲解，
可供 code review、提交整理，以及向不熟悉 RTL/AXI 的同事讲解。

最终结果（guest 串口输出见 `logs/sim/m5out/system.terminal`；host 侧
运行日志与 footer 见 `logs/sim/coral-mobilenet-host-20260802-013722.log`）：

```
mobilenet_state=0x00000003  mobilenet_error=0
mobilenet_npu_cycles=40957800  mobilenet_operation_count=40775552
mobilenet_output_checksum=0x38c2a6a4  mobilenet_test=PASS
exit=0，wall time 852 秒
```

修复清单总览：

| # | 问题 | 性质 | 主要文件 |
|---|---|---|---|
| 1 | AXI 握手死锁（仿真冻结） | 桥 testbench bug | `gem5_axi_master_drivers.h` |
| 2 | 部分掩码写被拒绝（store fault） | 桥功能缺口 | `gem5_dma_request_builder.h`、`coralnpu_gem5_abi.cc` |
| 3 | 跨行向量访存死锁（第 3 层沉默） | **RTL erratum**（固件绕过） | `gem5_mobilenet.cc`、`coral_operator_client.h`、`build_rvv_mobilenet.sh` |
| 4 | 桥 -O0 编译（1.3 kHz） | 构建配置 | `build_rvv_mobilenet.sh` |
| 5 | 缺乏可观测性（盲跑 30 小时） | 基础设施 | `coralnpu_gem5_abi.cc`、`gem5_core_mini_axi_wrapper.h` |
| 6 | 脚本输出不落盘、无计时 | 体验 | `run_rvv_mobilenet_test.sh` |

---

## 0. 背景知识速成（理解所有修复的三把钥匙）

### 0.1 AXI 的 valid/ready 握手

AXI 总线把主设备（CPU 核）和从设备（内存/桥）之间的对话拆成五个独立
通道：写地址 AW、写数据 W、写响应 B、读地址 AR、读数据 R。每个通道只有
一条规则：

> 发送方举 `valid`（"我有货"），接收方举 `ready`（"我能收"）；
> **某个时钟上升沿上两者同时为 1，这一拍才算成交。**
> 成交前发送方不许撤销、不许改内容。

类比：快递柜交接——你举手（valid）表示包裹到了，柜门开（ready）表示
能放；只有"柜门开着的那个瞬间你把手伸进去"才算完成交付。任何一个没
对上，货就一直举在原地。

### 0.2 Verilator testbench 的"半拍时差"

我们的 Coral RTL 被 Verilator 编译成 C++。RTL 在时钟上升沿打拍（更新
输出、采样输入）；桥里的"驱动"代码在时钟低电平阶段（falling edge）
干活。于是：

```
posedge N:   RTL 更新输出（比如举起 awvalid）
falling N:   桥驱动能看到 awvalid 了 —— 但 RTL 要等 posedge N+1 才评估握手
```

**桥驱动在 falling N 看到的"valid && ready=1"，要到 posedge N+1 才被
RTL 正式确认。** 这半个时钟周期的时差，是修复 1 的全部故事。

### 0.3 向量访存、行与掩码

- RVV 向量指令一次可以搬 16 字节（VLEN=128）。内存按 16 字节对齐切分
  叫"行"（line）。
- AXI 写数据通道带一个 **strobe（写掩码）**：16 个比特，每一位表示
  "这 16 个字节里这一字节是否真的要写"。部分掩码完全合法——比如
  `strb=0x000F` 表示"只写低 4 字节"。
- **跨行访问**：向量访存起始地址不是 16 的倍数时，数据横跨两行，
  硬件要拆成两笔总线事务。

---

## 1. AXI 握手死锁：桥在裁判吹哨前签了收

### 现象

仿真跑到固件写完 `ALLOCATE_BEGIN` 进度标记后彻底冻结：gem5 忙、时钟
照走，但 490M 周期里固件一步不动。watchdog（见修复 5）抓到稳定现场：

```
mwr[awvalid=1 awready=0 awaddr=0x30000020 wvalid=0]  ← RTL 举着写地址，桥不接收
deferred_request=0 bvalid=0
```

### 根因

桥的写驱动在 falling N 采样到 `awvalid && awready=1` 后，**立即捕获
地址并在同一个低电平阶段把 awready 拉低**。但 RTL 要到 posedge N+1 才
评估握手——此时 awready 已经是 0，RTL 认为"没成交"，继续举着地址等；
桥却认为"已收到"，开始等 W 数据（RTL 永远不会发）。双方互相死等。

类比：裁判规定"握手以哨声（posedge）为准"。桥在哨响前一瞬看到对方
伸手就签了收据，还顺手把门关了；对方没听见哨响，认为交易没发生，
一直伸手站在原地。

### 修复（`gem5_axi_master_drivers.h`）

原则：**捕获可以照旧，但 ready 必须等对方过完一个 posedge 才允许落下。**

- 写驱动 AW 捕获：新增一次性标志 `hold_addr_ready_`，让 awready 多保持
  一个 falling edge 再落 0；
- 写驱动 W 末拍：同理 `hold_data_ready_`（否则死锁会挪到 W 最后一拍）；
- 读驱动 AR：删掉捕获后立刻清零 arready 的两行，arready 自然延迟到下个
  falling edge 重算——语义等价、改动最小。

代价：每笔事务最多多 1 拍延迟，对仿真无感；换来"桥只消费 RTL 确实
交出过的东西"。`gem5_axi_master_drivers_test.cc` 两处断言按新时序顺延
一拍（那本来就是旧 bug 时序的编码），9 个用例全绿。

### 一句话讲给别人

"仿真器的总线模型在半拍时差里抢签了收据，导致 RTL 和桥对'这单生意
成没成'产生分歧、互相死等；修复就是让桥的 ready 信号晚半拍收兵，
保证对方的握手先被时钟正式确认。"

---

## 2. 部分掩码写被拒绝：桥只认"全页有效"的信封

### 现象

死锁修好后固件继续推进 390 万周期，然后在 `StridedSlicePrepare` 里一条
向量 store（`vse8.v`）收到总线错误响应，触发 store access fault
（mcause=7），gem5 fatal 退出。诊断打印给出实锤：

```
Coral AXI reject write addr=0x203fe8a0 len=0 size=4 burst=1
  beat[0] strb=0x000f last=1     ← 16 字节行事务，只使能低 4 字节
```

### 根因

桥的请求构造器 `BuildGem5DmaWriteRequest` 是为"标量形状"的事务写的，
其中一条检查要求 **strobe 必须全 1**（`gem5_dma_request_builder.h:87-91`
的旧逻辑）。而 Coral 的 LSU 把向量 store（及非对齐标量 store）统一发成
16 字节行事务 + 逐字节掩码，尾拍只使能几个字节——完全合法的 AXI，却被
桥拒绝（SLVERR）。

类比：桥是个只收"16 页全满"信封的收发室；RTL 寄来一个"16 页里只有
前 4 页有效"的合规信封，收发室直接拒收并回了张"地址错误"的条子，
RTL 当场异常死亡。

### 修复（`gem5_dma_request_builder.h`、`coralnpu_gem5_abi.cc`）

- 写回调流程重构：**先判断事务是否落在桥本地 EXTMEM 窗口**，是则走新的
  byte-enable 感知吸收路径——按掩码逐字节合并进本地 buffer，不再要求
  全掩码；全掩码写的行为逐字节不变；
- 严格检查（burst!=INCR / size>4 / WLAST / beat 数）保留，因为它们现在
  只守护 gem5 DMA 路径（ABI v6 的 `coral_gem5_dma_request` 没有掩码字段，
  部分掩码写无法表达——不在窗口内的掩码写拒绝并 loud-log，ABI v7 列为
  后续）；
- 所有拒绝点带**原因码**打印（`partial-strb`、`window-overflow` 等
  10 种），下次同类问题一眼定位。

### 一句话讲给别人

"桥把'只有部分字节有效的合法写操作'当成坏包拒收；修复是让桥学会按
掩码逐字节合并，只写该写的字节。"

---

## 3. 第 3 层沉默：RTL 跨行向量访存死锁（erratum）

### 现象

修复 1、2 后，固件跑完初始化、前两层卷积（hybrid host kernel 首次真正
执行），然后在第 3 个 `CONV_END` 之后核"沉默"：所有 AXI 通道空闲、
无 fault、无 wfi、无 halt，时钟照走，1500 万周期无任何总线活动。

### 根因（RTL erratum，证据链完整但未修 RTL）

用 dispatch/retire PC 探针（修复 5 的扩展）定位到：**Coral 核对"跨
16 字节行、且目标不是 DTCM 的向量 load/store"会死锁**——第一行的总线
事务正常完成，第二行的事务永远不发，该指令永不退休，顺序流水线永久
停摆。运行中共命中三个触发点：

1. `PadEval`：GCC 自动向量化 `PadParams` 拷贝 → 从 ≡4 mod 16 的 EXTMEM
   地址 `vle8.v`（跨行）——即第 3 层沉默点；
2. FC 层 staging 拷贝：4 对齐目标的跨行 `vse8.v`；
3. `StridedSliceEval`：`GetTensorShape` 从 ITCM/flatbuffer 区跨行向量读。

规律：标量访问任意地址 ✅；16 对齐向量访问 ✅；跨行向量访问 DTCM ✅；
跨行向量访问 EXTMEM/ITCM ❌。嫌疑位置（留给上游）：
`thirdparty/coralnpu/hdl/chisel/src/coralnpu/scalar/Lsu.scala` 或
`DBus2Axi.scala` 的向量多行迭代逻辑。

### 修复（固件/构建侧绕过，不动 RTL）

- `gem5_mobilenet.cc`：Pad 算子包一层 `TracedPadInvoke`——先把 OpData
  拷到 16 对齐的 DTCM 静态缓冲再走原 kernel；4 对齐的 per-channel
  multiplier/shift 改标量字拷贝；新增 PAD_BEGIN/END 进度标记
  （`coral_mobilenet.h` 两个副本同步更新）；
- `coral_operator_client.h`：staging 分配统一向上取整到 16 字节，保证
  所有 staging 地址 16 对齐；
- `build_rvv_mobilenet.sh`：新增 `TFLM_NOVEC_COPTS`，对 tflite_micro 全库
  禁用 GCC 自动向量化的全部三条路径（loop/SLP 向量化、loop→memcpy 惯
  用法、内联 movmem 展开）——coralnpu 自家 RVV kernel 保持向量化。

类比：这辆车的变速箱在"跨越两条车道线变道"时会卡死。修车（改 RTL）
不归我们；我们给导航重新规划路线（固件对齐纪律 + 关闭自动向量化），
永远不走跨线变道。

### 一句话讲给别人

"芯片设计里有个 erratum：向量指令跨 16 字节行访问外部内存会卡死流水
线。我们通过内存对齐纪律和关掉编译器自动向量化，保证固件永远不触发
它，并把 erratum 本身留给 RTL 上游修。"

---

## 4. 性能：桥是用 -O0 编译的（14 倍提速）

### 现象与根因

最初仿真速率 ~1.3 kHz（RTL cycle/秒），30 小时只推进 1.5 亿周期。排查
发现 `build_rvv_mobilenet.sh` 的 `bazel build` 没传 `-c opt`，产物落在
`bazel-out/k8-fastbuild`——即 **-O0 编译**。Verilated 模型是全系统最
重的 C++，-O0 → -O2 实测 **~14x**（1.3 kHz → 19.2 kHz）。

### 修复（`build_rvv_mobilenet.sh`）

- 构建加 `-c opt`（脚本本就支持 `"$@"` 透传）；
- **顺手修了一个隐藏 bug**：脚本的 `resolve_output()` 用 `bazel cquery`
  解析产物路径时不透传构建标志——导致主构建编的是 opt、安装的却是默认
  fastbuild 产物（自定义标志被静默忽略）。现在 `BUILD_EXTRA_ARGS` 同时
  转发给 build 和 cquery。

### 一句话讲给别人

"仿真慢首先不是模型复杂，而是它用调试档（-O0）编译；切到优化档并修
好'编A装B'的安装脚本后，速度提升 14 倍，调试迭代从 20 小时一轮变成
4 分钟一轮。"

---

## 5. 可观测性：watchdog 与现场打印（贯穿全案的基础设施）

没有这些仪器，上面每个 bug 都只能靠猜。全部保留在代码里：

- **AXI watchdog**（`coralnpu_gem5_abi.cc`）：RTL 连续 N 周期无 master
  活动（阈值 `CORAL_AXI_WATCHDOG_CYCLES`，默认 5M，已入 `AGENTS.md`
  环境变量表）时，dump 全部 AXI 通道握手电平
  （`gem5_core_mini_axi_wrapper.h::DumpMasterChannels`）+ 两个驱动的
  内部状态（`DumpState`）+ dispatch/retire PC 环与 AXI 握手历史
  （修复 3 的定位全靠它们）；
- **拒绝现场打印**：桥每次回 SLVERR 前打印事务完整参数 + 原因码；
- **heartbeat**（gem5 侧，原有）：每 1M 周期打 `requested_cycles` 与
  step 返回码。

### 一句话讲给别人

"硬件仿真调 bug 和修真机一样：先装示波器再修电路。watchdog 就是我们
的示波器——它把'卡住了'变成'卡在哪个信号的哪一拍'。"

---

## 6. 脚本：输出落盘 + 中断计时（`run_rvv_mobilenet_test.sh`）

- 全部控制台输出（脚本/gem5/桥 stderr）经 tee 同时写入
  `CORAL_MOBILENET_HOST_LOG`（默认 `logs/sim/coral-mobilenet-host-<ts>.log`，
  有 latest 软链）且控制台照常显示；
- 退出（正常/出错/Ctrl+C/SIGINT）必打 footer：起止时间、耗时、退出码，
  并指向 `m5out/stats.txt`（gem5 在 SIGINT 时正常 dump）；
- 末尾 `exec` 改为普通调用，退出码才能穿过 tee 传播；
- 修正头注释里"sampled 是默认"的过时说法（实际默认 `rtl`）。

---

## 7. 已定性但未修 / 后续事项

1. **RTL erratum**（修复 3 的根源）：跨行非 DTCM 向量访存死锁，留上游；
   若未来固件引入新的跨行向量访存（tflite_micro 的 novec 范围之外），
   erratum 会复发。
2. **gem5 时间域脱钩**：fast-DMA 功能模式下 gem5 全局时间推进比 RTL
   cycle 域慢 ~119 倍（30 小时 wall 里 gem5 时间只走 4.19ms）——设计
   使然（功能模式不给每拍赋予时间语义）；做周期研究请用
   `CORAL_FAST_DMA=0` 的 timing 模式。
3. **ABI v7**：`coral_gem5_dma_request` 加掩码字段，让部分掩码写也能走
   gem5 DMA 路径（当前仅 EXTMEM 吸收路径支持）。
4. **性能第二梯队**：空闲周期冗余 Eval 裁剪（每 RTL 周期 ~15-19 次全
   模型求值，空闲期可降到 2 次）、Verilator `--threads`、`-O3/-march=
   native`——rtl 模式全图运行前值得做。
5. **驱动 Reset 缺口**（评审 MAJOR-2）：`wrapper.Reset()` 时驱动的
  内部标志不清空，IOC_RESET 中途复位可能失同步——当前测试路径不触发，
   建议补 driver `Reset()`。

## 验证记录

- 单元测试 `//hw_sim:gem5_axi_master_drivers_test`：全绿；
- ABI 校验：`check_mobilenet_abi.sh`、`phase2_check_abi.sh`：通过；
- 端到端：`CORAL_MOBILENET_DEBUG=1 CORAL_OPERATOR_MODE=hybrid
  ./tools/coralnpu/run_rvv_mobilenet_test.sh` → `mobilenet_test=PASS`，
  852 秒，exit=0。

## 提交记录（本地仓，分支 bugfix/env-fixes-and-comments）

按依赖顺序逐笔提交（每笔可独立编译）：

| Commit | 主题 | 对应章节 | 文件 |
|---|---|---|---|
| `af684868234796b98598a059cf381e57cf920759` | fix: defer AXI ready drop after capture to close handshake deadlock | §1 | `gem5_axi_master_drivers.h`、`gem5_axi_master_drivers_test.cc` |
| `63a01fe5abf5ef4e2df315f8200d2efe01d4d6d9` | feat: dump AXI channel, driver, and core PC state in bridge watchdog | §5 | `gem5_core_mini_axi_wrapper.h` |
| `d9649ad9ba1d261ada3a52ba03b182e992971bc2` | fix: absorb partial-strobe EXTMEM writes; add watchdog and reject reasons | §2、§5 | `coralnpu_gem5_abi.cc`、`gem5_dma_request_builder.h` |
| `9677dcf1f5afd101027402208089f8e2347de0f4` | fix: work around Coral RTL erratum on line-crossing vector access | §3 | `gem5_mobilenet.cc`、`coral_mobilenet.h`（×2）、`coral_operator_client.h` |
| `807c6975393dd7199edc977d8736f6da19e3518d` | build: forward extra bazel flags to cquery; disable TFLM autovectorization | §3、§4 | `build_rvv_mobilenet.sh` |
| `b87966ef3dc1cbb0c99d7439d36e9f810fc81124` | feat: tee host output to a log file and print timing footer on exit | §6 | `run_rvv_mobilenet_test.sh` |
| `00359c784ca1d11aec88fd8383e83ddf5ada1e0f` | docs: record watchdog env var, MobileNet e2e analysis, erratum progress | §5 及配套文档 | `AGENTS.md`、`mobilenet_test_e2e_analysis.md`、`current_progress.md` |
| （见 git log 最新一笔） | docs: add MobileNet bring-up fixes writeup with commit index | 本文档 | `mobilenet_bringup_fixes.md` |

说明：`gem5_axi_master_drivers.h` 的 `DumpState()`/握手历史随 §1 提交
（watchdog 依赖）；`.gitignore`、`tools/guest_tools/build_busybox_aarch64.sh`
为会话前已存在的本地改动，未纳入本系列；`sim/coralnpu/MODULE.bazel*`、
`sim/coralnpu/rules/BUILD` 为来源待确认的未跟踪文件，未提交。
