# Copyright (c) 2016-2017, 2019-2021 ARM Limited
# All rights reserved.
#
# The license below extends only to copyright in the software and shall
# not be construed as granting a license to any other intellectual
# property including but not limited to intellectual property relating
# to a hardware implementation of the functionality of the software
# licensed hereunder.  You may use the software subject to the license
# terms below provided that you ensure that this notice is replicated
# unmodified and in its entirety in all distributions of the software,
# modified or unmodified, in source code or in binary form.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met: redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer;
# redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution;
# neither the name of the copyright holders nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# This is an example configuration script for full system simulation of
# a generic ARM bigLITTLE system.

import argparse
import os
import sys


import m5
import m5.util
from m5.objects import *
from m5.objects import Cache

m5.util.addToPath("../../")

import devices
from common import (
    FSConfig,
    MemConfig,
    ObjectList,
    Options,
    SysPaths,
)

from devices import (
    AtomicCluster,
    FastmodelCluster,
    KvmCluster,
)

from common.cores.arm import (
    O3_ARM_Cortex_x4 as x4_core,
    O3_ARM_Cortex_A720 as a720_core,
)

default_disk = "ubuntu-18.04-arm64-docker.img"

default_mem_size = "4GiB"


def _to_ticks(value):
    """Helper function to convert a latency from string format to Ticks"""

    return m5.ticks.fromSeconds(m5.util.convert.anyToLatency(value))


def _using_pdes(root):
    """Determine if the simulator is using multiple parallel event queues"""

    for obj in root.descendants():
        if (
            not m5.proxy.isproxy(obj.eventq_index)
            and obj.eventq_index != root.eventq_index
        ):
            return True

    return False


class BigCluster(devices.ArmCpuCluster):
    def __init__(self, system, num_cpus, cpu_clock, cpu_voltage="1.0V"):
        cpu_config = [
            ObjectList.cpu_list.get("O3_ARM_v7a_3"),
            devices.L1I,
            devices.L1D,
            devices.L2,
        ]
        super().__init__(system, num_cpus, cpu_clock, cpu_voltage, *cpu_config)

class LittleCluster(devices.ArmCpuCluster):
    def __init__(self, system, num_cpus, cpu_clock, cpu_voltage="1.0V"):
        cpu_config = [
            ObjectList.cpu_list.get("MinorCPU"),
            devices.L1I,
            devices.L1D,
            devices.L2,
        ]
        super().__init__(system, num_cpus, cpu_clock, cpu_voltage, *cpu_config)

class X4BigCluster(devices.L2PrivCluster):
    def __init__(self, system, num_cpus, cpu_clock, cpu_voltage="1.0V"):
        cpu_config = [
            ObjectList.cpu_list.get("O3_ARM_Cortex_x4"),
            x4_core.O3_ARM_Cortex_x4_ICache,
            x4_core.O3_ARM_Cortex_x4_DCache,
            x4_core.O3_ARM_Cortex_x4L2,
        ]
        super().__init__(system, num_cpus, cpu_clock, cpu_voltage, *cpu_config)

class A720MiddleCluster(devices.L2PrivCluster):
    def __init__(self, system, num_cpus, cpu_clock, cpu_voltage="1.0V"):
        cpu_config = [
            ObjectList.cpu_list.get("O3_ARM_Cortex_A720"),
            a720_core.O3_ARM_Cortex_A720_ICache,
            a720_core.O3_ARM_Cortex_A720_DCache,
            a720_core.O3_ARM_Cortex_A720L2,

        ]
        super().__init__(system, num_cpus, cpu_clock, cpu_voltage, *cpu_config)

class A720LittleCluster(devices.L2PrivCluster):
    def __init__(self, system, num_cpus, cpu_clock, cpu_voltage="1.0V"):
        cpu_config = [
            ObjectList.cpu_list.get("O3_ARM_Cortex_A720"),
            a720_core.O3_ARM_Cortex_A720_ICache,
            a720_core.O3_ARM_Cortex_A720_DCache,
            a720_core.O3_ARM_Cortex_A720L2,

        ]
        super().__init__(system, num_cpus, cpu_clock, cpu_voltage, *cpu_config)

class L3Cache(Cache):
    size = "8MiB"
    assoc = 16
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 48
    tgts_per_mshr = 16
    writeback_clean = True
    clusivity = "mostly_incl"
    prefetcher = NULL

def createSystem(
    caches,
    kernel,
    bootscript,
    machine_type="VExpress_GEM5",
    disks=[],
    mem_size=default_mem_size,
    bootloader=None,
    options=None,
):
    platform = ObjectList.platform_list.get(machine_type)
    m5.util.inform("Simulated platform: %s", platform.__name__)

    sys = devices.SimpleSystem(
        caches,
        mem_size,
        platform(),
        workload=ArmFsLinux(object_file=SysPaths.binary(kernel)),
        readfile=bootscript,
    )

    # sys.mem_ctrls = [
    #     SimpleMemory(range=r, port=sys.membus.mem_side_ports)
    #     for r in sys.mem_ranges
    # ]
    MemConfig.config_mem(options, sys)

    sys.connect()

    # Attach disk images
    if disks:

        def cow_disk(image_file):
            image = CowDiskImage()
            image.child.image_file = SysPaths.disk(image_file)
            return image

        sys.disk_images = [cow_disk(f) for f in disks]
        sys.pci_vio_block = [
            PciVirtIO(vio=VirtIOBlock(image=img)) for img in sys.disk_images
        ]
        for dev in sys.pci_vio_block:
            sys.attach_pci(dev)

    sys.realview.setupBootLoader(sys, SysPaths.binary, bootloader)

    sys.exit_on_work_items = True
    return sys


cpu_types = {
    "atomic": (AtomicCluster, AtomicCluster, AtomicCluster),
    "timing": (BigCluster, LittleCluster),
    "Heng": (X4BigCluster, A720MiddleCluster, A720LittleCluster),
}

# Only add the KVM CPU if it has been compiled into gem5
if devices.have_kvm:
    cpu_types["kvm"] = (KvmCluster, KvmCluster)

# Only add the FastModel CPU if it has been compiled into gem5
if devices.have_fastmodel:
    cpu_types["fastmodel"] = (FastmodelCluster, FastmodelCluster)


def addOptions(parser):
    parser.add_argument(
        "--restore-from",
        type=str,
        default=None,
        help="Restore from checkpoint",
    )
    parser.add_argument(
        "--dtb", type=str, default=None, help="DTB file to load"
    )
    parser.add_argument(
        "--kernel", type=str, required=True, help="Linux kernel"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/dev/vda1",
        help="Specify the kernel CLI root= argument",
    )
    parser.add_argument(
        "--machine-type",
        type=str,
        choices=ObjectList.platform_list.get_names(),
        default="VExpress_GEM5",
        help="Hardware platform class",
    )
    parser.add_argument(
        "--disk",
        action="append",
        type=str,
        default=[],
        help="Disks to instantiate",
    )
    parser.add_argument(
        "--bootscript", type=str, default="", help="Linux bootscript"
    )
    parser.add_argument(
        "--cpu-type",
        type=str,
        choices=list(cpu_types.keys()),
        default="timing",
        help="CPU simulation mode. Default: %(default)s",
    )
    parser.add_argument(
        "--kernel-init", type=str, default="/sbin/init", help="Override init"
    )
    parser.add_argument(
        "--big-cpus",
        type=int,
        default=2,
        help="Number of big CPUs to instantiate",
    )
    parser.add_argument(
        "--middle-cpus",
        type=int,
        default=4,
        help="Number of middle CPUs to instantiate",
    )
    parser.add_argument(
        "--little-cpus",
        type=int,
        default=2,
        help="Number of little CPUs to instantiate",
    )
    parser.add_argument(
        "--caches",
        action="store_true",
        default=True,
        help="Instantiate caches",
    )
    parser.add_argument(
        "--last-cache-level",
        type=int,
        default=3,
        help="Last level of caches (e.g. 3 for L3)",
    )
    parser.add_argument(
        "--l3-size",
        type=str,
        default="8MiB",
        help="Memory size of L3 Cache",
    )
    parser.add_argument(
        "--slc-size",
        type=str,
        default="8MiB",
        help="Memory size of SLC Cache",
    )
    parser.add_argument(
        "--big-cpu-clock",
        type=str,
        default="3.3GHz",
        help="Big CPU clock frequency",
    )
    parser.add_argument(
        "--middle-cpu-clock",
        type=str,
        default="2.9GHz",
        help="Big CPU clock frequency",
     )
    parser.add_argument(
        "--little-cpu-clock",
        type=str,
        default="1.8GHz",
        help="Little CPU clock frequency",
    )
    parser.add_argument(
        "--sim-quantum",
        type=str,
        default="1ms",
        help="Simulation quantum for parallel simulation. "
        "Default: %(default)s",
    )
    parser.add_argument(
        "--mem-size",
        type=str,
        default=default_mem_size,
        help="System memory size",
    )
    parser.add_argument(
        "--mem-type",
        default="LPDDR5_8533_1x16_BG_BL32",
        choices=ObjectList.mem_list.get_names(),
        help="type of memory to use",
    )
    parser.add_argument(
        "--mem-channels", type=int, default=4, help="number of memory channels"
    )
    parser.add_argument(
        "--kernel-cmd",
        type=str,
        default=None,
        help="Custom Linux kernel command",
    )
    parser.add_argument(
        "--bootloader",
        action="append",
        help="executable file that runs before the --kernel",
    )
    parser.add_argument(
        "--kvm-userspace-gic",
        action="store_true",
        default=False,
        help="Use the gem5 GIC in a KVM simulation",
    )
    parser.add_argument(
        "-P",
        "--param",
        action="append",
        default=[],
        help="Set a SimObject parameter relative to the root node. "
        "An extended Python multi range slicing syntax can be used "
        "for arrays. For example: "
        "'system.cpu[0,1,3:8:2].max_insts_all_threads = 42' "
        "sets max_insts_all_threads for cpus 0, 1, 3, 5 and 7 "
        "Direct parameters of the root object are not accessible, "
        "only parameters of its children.",
    )
    parser.add_argument(
        "--vio-9p", action="store_true", help=Options.vio_9p_help
    )
    parser.add_argument(
        "--dtb-gen",
        action="store_true",
        help="Doesn't run simulation, it generates a DTB only",
    )

    # 新增功能 #
    parser.add_argument(
        "--slice-run",
        action="store_true",
        default=False,
        help="Run Simulation in time slices and periodically dump stats"
    )
    parser.add_argument(
        "--slices",
        type=int,
        default=100000,
        help="启动切片运行的总slice数(0 表示不用切片)"
    )
    parser.add_argument(
        "--slice-ticks",
        type=str,
        default="1s",
        help="Length of each simulation slice(e.g., 1ms, 1s)"
    )
    parser.add_argument(
        "--save-slices",
        action="store_true",
        default=False,
        help="在每个slice结束时保存断点"
    )
    parser.add_argument(
        "--restore-slice",
        type=int,
        default=-1,
        help="从指令slice的checkpoint恢复"
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=os.path.abspath(os.path.join(m5.options.outdir, "ckpts")),
    )
    parser.add_argument(
        "--cpu-monitor",
        action="store_true",
        default=False,
        help="add mem monitor to out port of cpu"
    )
    parser.add_argument(
        "--l1-monitor",
        action="store_true",
        default=False,
        help="add mem monitor to out port of l1"
    )
    parser.add_argument(
        "--l2-monitor",
        action="store_true",
        default=False,
        help="add mem monitor to out port of l2"
    )
    parser.add_argument(
        "--l3-monitor",
        action="store_true",
        default=False,
        help="add mem monitor to out port of l3"
    )
    parser.add_argument(
        "--slc-monitor",
        action="store_true",
        default=False,
        help="add mem monitor to out port of slc"
    )
    return parser


def build(options):
    m5.ticks.fixGlobalFrequency()

    kernel_cmd = [
        "earlyprintk",
        "earlycon=pl011,0x1c090000",
        "console=ttyAMA0",
        "lpj=19988480",
        "norandmaps",
        "loglevel=8",
        f"mem={options.mem_size}",
        f"root={options.root}",
        "rw",
        f"init={options.kernel_init}",
        "vmalloc=768MB",
        "swiotlb=force,swiotlb=512M",
    ]

    root = Root(full_system=True)

    disks = [default_disk] if len(options.disk) == 0 else options.disk
    system = createSystem(
        options.caches,
        options.kernel,
        options.bootscript,
        options.machine_type,
        disks=disks,
        mem_size=options.mem_size,
        bootloader=options.bootloader,
        options=options
    )
    
    root.system = system
    if options.kernel_cmd:
        system.workload.command_line = options.kernel_cmd
    else:
        system.workload.command_line = " ".join(kernel_cmd)

    if options.big_cpus + options.middle_cpus + options.little_cpus == 0:
        m5.util.panic("Empty CPU clusters")

    big_model, middle_model, little_model = cpu_types[options.cpu_type]

    all_cpus = []
    # big cluster
    if options.big_cpus > 0:
        system.bigCluster = big_model(
            system, options.big_cpus, options.big_cpu_clock
        )
        system.mem_mode = system.bigCluster.memory_mode()
        all_cpus += system.bigCluster.cpus

    # middle cluster
    if options.middle_cpus > 0:
        system.middleCluster = middle_model(
            system, options.middle_cpus, options.middle_cpu_clock
        )
        system.mem_mode = system.middleCluster.memory_mode()
        all_cpus += system.middleCluster.cpus

    # little cluster
    if options.little_cpus > 0:
        system.littleCluster = little_model(
            system, options.little_cpus, options.little_cpu_clock
        )
        system.mem_mode = system.littleCluster.memory_mode()
        all_cpus += system.littleCluster.cpus

    # Figure out the memory mode
    if (
        options.big_cpus > 0
        and options.middle_cpus > 0
        and options.little_cpus > 0
        and system.bigCluster.memory_mode()
        != system.middleCluster.memory_mode()
        or system.bigCluster.memory_mode()
        != system.littleCluster.memory_mode()
    ):
        m5.util.panic("Memory mode missmatch among CPU clusters")

    # add L3 & SLC inside
    system.addCaches(options.caches, options.last_cache_level, options.l3_size, options.slc_size, options=options)

    # Create a KVM VM and do KVM-specific configuration
    if issubclass(big_model, KvmCluster):
        _build_kvm(options, system, all_cpus)

    # Linux device tree
    if options.dtb is not None:
        system.workload.dtb_filename = SysPaths.binary(options.dtb)
    else:
        system.workload.dtb_filename = os.path.join(
            m5.options.outdir, "system.dtb"
        )
        system.generateDtb(system.workload.dtb_filename)

    if devices.have_fastmodel and issubclass(big_model, FastmodelCluster):
        from m5 import arm_fast_model as fm
        from m5 import systemc as sc

        # setup FastModels for simulation
        fm.setup_simulation("cortexa76")
        # setup SystemC
        root.systemc_kernel = m5.objects.SystemC_Kernel()
        m5.tlm.tlm_global_quantum_instance().set(
            sc.sc_time(10000.0 / 100000000.0, sc.sc_time.SC_SEC)
        )

    if options.vio_9p:
        FSConfig.attach_9p(system.realview, system.iobus)

    return root


def _build_kvm(options, system, cpus):
    system.kvm_vm = KvmVM()
    system.release = ArmDefaultRelease.for_kvm()

    if options.kvm_userspace_gic:
        # We will use the simulated GIC.
        # In order to make it work we need to remove the system interface
        # of the generic timer from the DTB and we need to inform the
        # MuxingKvmGic class to use the gem5 GIC instead of relying on the
        # host interrupt controller
        GenericTimer.generateDeviceTree = SimObject.generateDeviceTree
        system.realview.gic.simulate_gic = True

    # Assign KVM CPUs to their own event queues / threads. This
    # has to be done after creating caches and other child objects
    # since these mustn't inherit the CPU event queue.
    if len(cpus) > 1:
        device_eq = 0
        first_cpu_eq = 1
        for idx, cpu in enumerate(cpus):
            # Child objects usually inherit the parent's event
            # queue. Override that and use the same event queue for
            # all devices.
            for obj in cpu.descendants():
                obj.eventq_index = device_eq
            cpu.eventq_index = first_cpu_eq + idx

def _resolve_slice_dir(options):
    base = os.path.abspath(os.path.normpath(options.ckpt_dir))
    cdir = os.path.join(base, f"slice_{options.restore_slice:04d}")
    cdir = os.path.abspath(cdir)
    print(f"[restore] trying slice dir: {cdir}")

    if not os.path.isdir(cdir):
        parent = os.path.dirname(cdir)
        try:
            listing = ", ".join(sorted(os.listdir(parent)))
        except Exception as e:
            listing = f"<cannot list {parent}: {e}>"
        m5.util.panic(
            f"Restore slice directory not found:\n"
            f" expected:{cdir}\n"
            f" parent:  {parent}\n"
            f" entries: {listing}\n"
        )
    return cdir
def instantiate(options, checkpoint_dir=None):
    # 按slice保存恢复
    if options.restore_slice is not None and options.restore_slice >= 0:
        options.restore_from = _resolve_slice_dir(options)

    root = Root.getInstance()
    if root and _using_pdes(root):
        m5.util.inform(
            "Running in PDES mode with a %s simulation quantum.",
            options.sim_quantum,
        )
        root.sim_quantum = _to_ticks(options.sim_quantum)

    # Get and load from the chkpt or simpoint checkpoint
    if options.restore_from:
        if checkpoint_dir and not os.path.isabs(options.restore_from):
            cpt = os.path.join(checkpoint_dir, options.restore_from)
        else:
            cpt = options.restore_from

        m5.util.inform("Restoring from checkpoint %s", cpt)
        m5.instantiate(cpt)
    else:
        m5.instantiate()

def _to_tick(s):
    return int(m5.ticks.fromSeconds(m5.util.convert.anyToLatency(s)))

def _dump_stats_with_marker(s, prefix="slice"):
    print(f"{prefix} {s} Begin Simulation Statistics")
    m5.stats.dump()
    m5.stats.reset()
    print(f"[{prefix} {s}] End Simulation Statistics")


def run(options):
    os.makedirs(options.ckpt_dir, exist_ok=True)
    #起始slice序号：指定 id + 1开始
    start_slice = options.restore_slice + 1 if options.restore_slice >= 0 else 0
    if start_slice < 0:
        m5.util.panic("Using slice-run, but --slices and --slice-ticks must be > 0")

    if options.slice_run:
        options.slice_ticks = int(_to_ticks(options.slice_ticks))
        for s in range(start_slice, options.slices):
            slice_end_tick = (s + 1) * options.slice_ticks
            now = m5.curTick()
            delta = max(0, slice_end_tick - now)
            print(f"[slice {s} run {delta} tick (from {now} to {slice_end_tick})]")
            ev = m5.simulate(delta)
            cause = ev.getCause()
            print(f"[slice {s} cause={cause} @ {m5.curTick()}]")
            _dump_stats_with_marker(s)
            if options.save_slices:
                cpt_dir = os.path.join(options.ckpt_dir, f"slice_{s:04d}")
                print(f"[slice {s} save checkpoint --> {cpt_dir}]")
                m5.checkpoint(cpt_dir)

    # start simulation (and drop checkpoints when requested)
    else:
        while True:
            event = m5.simulate()
            exit_msg = event.getCause()
            if exit_msg == "checkpoint":
                print("Dropping checkpoint at tick %d" % m5.curTick())
                cpt_dir = os.path.join(options.ckpt_dir, "cpt.%d" % m5.curTick())
                m5.checkpoint(cpt_dir)
                print("Checkpoint done.")
            elif "workbegin" in exit_msg:
                print("m5 work begin dump and reset")
                m5.stats.reset()
            elif "workend" in exit_msg:
                print("m5 work end dump and reset")
                m5.stats.dump()
                m5.stats.reset()
            else:
                print(exit_msg, " @ ", m5.curTick())
                break

    sys.exit(event.getCode())


def generateDtb(root):
    root.system.generateDtb(os.path.join(m5.options.outdir, "system.dtb"))


def main():
    parser = argparse.ArgumentParser(
        description="Generic ARM big.LITTLE configuration"
    )
    addOptions(parser)
    options = parser.parse_args()
    root = build(options)
    root.apply_config(options.param)
    instantiate(options)
    if options.dtb_gen:
        generateDtb(root)
    else:
        run(options)


if __name__ == "__m5_main__":
    main()
