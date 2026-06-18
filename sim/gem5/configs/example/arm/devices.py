# Copyright (c) 2016-2017, 2019, 2021-2023 Arm Limited
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

# System components used by the bigLITTLE.py configuration script

import m5
from m5.objects import *

m5.util.addToPath("../../")
from common import ObjectList
from common.Caches import *

have_kvm = "ArmV8KvmCPU" in ObjectList.cpu_list.get_names()
have_fastmodel = "FastModelCortexA76" in ObjectList.cpu_list.get_names()


class L1I(L1_ICache):
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 4
    tgts_per_mshr = 8
    size = "48KiB"
    assoc = 3


class L1D(L1_DCache):
    tag_latency = 2
    data_latency = 2
    response_latency = 1
    mshrs = 10
    tgts_per_mshr = 8
    size = "32KiB"
    assoc = 2
    write_buffers = 16


class L2(L2Cache):
    tag_latency = 12
    data_latency = 12
    response_latency = 5
    mshrs = 24
    tgts_per_mshr = 16
    size = "1MiB"
    assoc = 16
    write_buffers = 8
    clusivity = "mostly_excl"


class L3(Cache):
    assoc = 16
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 128
    tgts_per_mshr = 16
    write_buffers = 32
    clusivity = "mostly_excl"
    # prefetcher = StridePrefetcher(degree=32, latency=1, prefetch_on_access=True)
    prefetcher = AMPMPrefetcher(
        on_miss=False, on_read=True, on_write=False, on_data=True, on_inst=False,
        latency=6, queue_size=192, queue_filter=True, queue_squash=True,
        cache_snoop=True, tag_prefetch=True,

        ampm=AccessMapPatternMatching(
            # 顺序流：起始度数适中，限制 stride 在可用范围（以 cacheline 为单位）
            start_degree=24,
            limit_stride=32,
            hot_zone_size="16kB",
            # 提升表容量与相联，增强学习覆盖
            access_map_table_entries="2048",
            access_map_table_assoc=16,
            # 加快自适应节流（缩短 epoch）
            epoch_cycles=50000,
            high_coverage_threshold=0.60,
            low_coverage_threshold=0.35,
            high_accuracy_threshold=0.40,
            low_accuracy_threshold=0.20,
            high_cache_hit_threshold=0.95,
            low_cache_hit_threshold=0.60,
            # 抬高“可用带宽”估计，避免 degree 过早被压
            offchip_memory_latency="100ns",
        ),
    )

class SLC(Cache):
    assoc = 16
    tag_latency = 22
    data_latency = 25
    response_latency = 1
    mshrs = 128
    tgts_per_mshr = 16
    write_buffers = 64
    writeback_clean = False
    clusivity = "mostly_incl"
    # prefetcher = StridePrefetcher(degree=32, latency=1, prefetch_on_access=True)
    prefetcher = AMPMPrefetcher(
        on_miss=False, on_read=True, on_write=False, on_data=True, on_inst=False,
        latency=8, queue_size=192, queue_filter=True, queue_squash=True,
        cache_snoop=True, tag_prefetch=True,

        ampm=AccessMapPatternMatching(
            start_degree=20,
            limit_stride=32,
            hot_zone_size="16kB",
            access_map_table_entries="2048",
            access_map_table_assoc=16,
            epoch_cycles=60000,
            high_coverage_threshold=0.60,
            low_coverage_threshold=0.35,
            high_accuracy_threshold=0.40,
            low_accuracy_threshold=0.20,
            high_cache_hit_threshold=0.95,
            low_cache_hit_threshold=0.60,
            offchip_memory_latency="100ns",
        ),
    )


class MemBus(SystemXBar):
    badaddr_responder = BadAddr(warn_access="warn")
    default = Self.badaddr_responder.pio
    snoop_filter = NULL
    width = 128
    response_latency = 3
    frontend_latency = 7
    forward_latency = 7


class ArmCpuCluster(CpuCluster):
    def __init__(
        self,
        system,
        num_cpus,
        cpu_clock,
        cpu_voltage,
        cpu_type,
        l1i_type,
        l1d_type,
        l2_type,
        tarmac_gen=False,
        tarmac_dest=None,
    ):
        super().__init__()
        self._cpu_type = cpu_type
        self._l1i_type = l1i_type
        self._l1d_type = l1d_type
        self._l2_type = l2_type

        assert num_cpus > 0

        self.voltage_domain = VoltageDomain(voltage=cpu_voltage)
        self.clk_domain = SrcClockDomain(
            clock=cpu_clock, voltage_domain=self.voltage_domain
        )

        self.generate_cpus(cpu_type, num_cpus)

        for cpu in self.cpus:
            if tarmac_gen:
                cpu.tracer = TarmacTracer()
                if tarmac_dest is not None:
                    cpu.tracer.outfile = tarmac_dest

        system.addCpuCluster(self)

    def addL1(self):
        for cpu in self.cpus:
            l1i = None if self._l1i_type is None else self._l1i_type()
            l1d = None if self._l1d_type is None else self._l1d_type()
            cpu.addPrivateSplitL1Caches(l1i, l1d)

    def addL2(self, clk_domain):
        if self._l2_type is None:
            return
        self.toL2Bus = L2XBar(width=64, clk_domain=clk_domain)
        self.l2 = self._l2_type()
        for cpu in self.cpus:
            cpu.connectCachedPorts(self.toL2Bus.cpu_side_ports)
        self.toL2Bus.mem_side_ports = self.l2.cpu_side
        self.toL2Bus.point_of_coherency = False
        self.toL2Bus.point_of_unification = False

    def addPMUs(
        self,
        ints,
        events=[],
        exit_sim_on_control=False,
        exit_sim_on_interrupt=False,
    ):
        """
        Instantiates 1 ArmPMU per PE. The method is accepting a list of
        interrupt numbers (ints) used by the PMU and a list of events to
        register in it.

        :param ints: List of interrupt numbers. The code will iterate over
            the cpu list in order and will assign to every cpu in the cluster
            a PMU with the matching interrupt.
        :type ints: List[int]
        :param events: Additional events to be measured by the PMUs
        :type events: List[Union[ProbeEvent, SoftwareIncrement]]
        :param exit_sim_on_control: If true, exit the sim loop when the PMU is
            enabled, disabled, or reset.
        :type exit_on_control: bool
        :param exit_sim_on_interrupt: If true, exit the sim loop when the PMU
            triggers an interrupt.
        :type exit_on_control: bool

        """
        assert len(ints) == len(self.cpus)
        for cpu, pint in zip(self.cpus, ints):
            int_cls = ArmPPI if pint < 32 else ArmSPI
            for isa in cpu.isa:
                isa.pmu = ArmPMU(interrupt=int_cls(num=pint))
                isa.pmu.exitOnPMUControl = exit_sim_on_control
                isa.pmu.exitOnPMUInterrupt = exit_sim_on_interrupt
                isa.pmu.addArchEvents(
                    cpu=cpu,
                    itb=cpu.mmu.itb,
                    dtb=cpu.mmu.dtb,
                    icache=getattr(cpu, "icache", None),
                    dcache=getattr(cpu, "dcache", None),
                    l2cache=getattr(self, "l2", None),
                )
                for ev in events:
                    isa.pmu.addEvent(ev)

    def connectMemSide(self, bus):
        try:
            self.l2.mem_side = bus.cpu_side_ports
        except AttributeError:
            for cpu in self.cpus:
                cpu.connectCachedPorts(bus.cpu_side_ports)

class ArmCpuClusterWithMonitor(ArmCpuCluster):
    def addL1(self, options = None):
        for cpu in self.cpus:
            l1i = None if self._l1i_type is None else self._l1i_type()
            l1d = None if self._l1d_type is None else self._l1d_type()
            cpu.addPrivateSplitL1Caches(l1i, l1d, options = options)

    def addL2(self, clk_domain, options = None):
        if self._l2_type is None:
            return
        self.toL2Bus = L2XBar(width=64, clk_domain=clk_domain)
        self.l2 = self._l2_type()
        for cpu in self.cpus:
            cpu.connectCachedPorts(self.toL2Bus.cpu_side_ports, options = options)
        self.toL2Bus.mem_side_ports = self.l2.cpu_side
        self.toL2Bus.point_of_coherency = False
        self.toL2Bus.point_of_unification = False

    def connectMemSide(self, bus, options = None):
        opt_l2_monitor = getattr(options, "l2_monitor", False)
        
        try:
            if opt_l2_monitor:
                self.l2.monitor = CommMonitor()
                self.l2.monitor.footprint = MemFootprintProbe()
                self.l2.mem_side = self.l2.monitor.cpu_side_port
                self.l2.monitor.mem_side_port = bus.cpu_side_ports
            else:
                self.l2.mem_side = bus.cpu_side_ports
        except AttributeError:
            for cpu in self.cpus:
                cpu.connectCachedPorts(bus.cpu_side_ports, options = options)
class L2PrivCluster(ArmCpuClusterWithMonitor):
    def addL1(self, options=None):
        for cpu in self.cpus:
            l1i = None if self._l1i_type is None else self._l1i_type()
            l1d = None if self._l1d_type is None else self._l1d_type()

            if (
                options
                and getattr(options, "enable_gemmini", False)
                and getattr(options, "gemmini_cpu", None) is cpu
                and not getattr(options, "gemmini_mmio_iobus", False)
            ):
                opt_cpu_monitor = getattr(options, "cpu_monitor", False)

                cpu.icache = l1i
                cpu.dcache = l1d

                if hasattr(cpu, "connectMayWithMonitor"):
                    cpu.connectMayWithMonitor(
                        cpu,
                        "self.icache_port",
                        l1i.cpu_side,
                        opt_cpu_monitor,
                        name="iportmonitor",
                    )
                    cpu.connectMayWithMonitor(
                        cpu,
                        "self.dcache_port",
                        options.gemmini_dev.cpu_side,
                        opt_cpu_monitor,
                        name="dportmonitor",
                    )
                else:
                    cpu.icache_port = l1i.cpu_side
                    cpu.dcache_port = options.gemmini_dev.cpu_side

                options.gemmini_dev.mem_side = l1d.cpu_side

                cpu._cached_ports = ["icache.mem_side", "dcache.mem_side"]
                cpu._cached_ports += cpu.ArchMMU.walkerPorts()

                if cpu.checker != NULL:
                    cpu._cached_ports += [
                        "checker." + port
                        for port in cpu.ArchMMU.walkerPorts()
                    ]
            else:
                cpu.addPrivateSplitL1Caches(l1i, l1d, options=options)

    def addL2(self, clk_domain, options = None):
        for cpu in self.cpus:
            cpu.privL2 = self._l2_type()
            cpu.toL2Bus = CoherentXBar(width=64,
                                    clk_domain=clk_domain,
                                    frontend_latency=1,
                                    forward_latency=0,
                                    response_latency=1,
                                    header_latency=1,
                                    snoop_response_latency=1)

            cpu.connectCachedPorts(cpu.toL2Bus.cpu_side_ports, options)
            cpu.toL2Bus.mem_side_ports = cpu.privL2.cpu_side

    def connectMemSide(self, bus, options = None):
        opt_l2_monitor = getattr(options, "l2_monitor", False)
        
        for cpu in self.cpus:
            if opt_l2_monitor:
                cpu.privL2.monitor = CommMonitor()
                cpu.privL2.monitor.footprint = MemFootprintProbe()
                cpu.privL2.mem_side = cpu.privL2.monitor.cpu_side_port
                cpu.privL2.monitor.mem_side_port = bus.cpu_side_ports
            else:
                cpu.privL2.mem_side = bus.cpu_side_ports      

class AtomicCluster(ArmCpuCluster):
    def __init__(
        self,
        system,
        num_cpus,
        cpu_clock,
        cpu_voltage="1.0V",
        tarmac_gen=False,
        tarmac_dest=None,
    ):
        super().__init__(
            system,
            num_cpus,
            cpu_clock,
            cpu_voltage,
            cpu_type=ObjectList.cpu_list.get("AtomicSimpleCPU"),
            l1i_type=None,
            l1d_type=None,
            l2_type=None,
            tarmac_gen=tarmac_gen,
            tarmac_dest=tarmac_dest,
        )

    def addL1(self):
        pass


class KvmCluster(ArmCpuCluster):
    def __init__(
        self,
        system,
        num_cpus,
        cpu_clock,
        cpu_voltage="1.0V",
        tarmac_gen=False,
        tarmac_dest=None,
    ):
        super().__init__(
            system,
            num_cpus,
            cpu_clock,
            cpu_voltage,
            cpu_type=ObjectList.cpu_list.get("ArmV8KvmCPU"),
            l1i_type=None,
            l1d_type=None,
            l2_type=None,
            tarmac_gen=tarmac_gen,
            tarmac_dest=tarmac_dest,
        )

    def addL1(self):
        pass


class FastmodelCluster(CpuCluster):
    def __init__(self, system, num_cpus, cpu_clock, cpu_voltage="1.0V"):
        super().__init__()

        # Setup GIC
        gic = system.realview.gic
        gic.sc_gic.cpu_affinities = ",".join(
            ["0.0.%d.0" % i for i in range(num_cpus)]
        )

        # Parse the base address of redistributor.
        redist_base = gic.get_redist_bases()[0]
        redist_frame_size = 0x40000 if gic.sc_gic.has_gicv4_1 else 0x20000
        gic.sc_gic.reg_base_per_redistributor = ",".join(
            [
                "0.0.%d.0=%#x" % (i, redist_base + redist_frame_size * i)
                for i in range(num_cpus)
            ]
        )

        gic_a2t = AmbaToTlmBridge64(amba=gic.amba_m)
        gic_t2g = TlmToGem5Bridge64(
            tlm=gic_a2t.tlm, gem5=system.iobus.cpu_side_ports
        )
        gic_g2t = Gem5ToTlmBridge64(gem5=system.membus.mem_side_ports)
        gic_g2t.addr_ranges = gic.get_addr_ranges()
        gic_t2a = AmbaFromTlmBridge64(tlm=gic_g2t.tlm)
        gic.amba_s = gic_t2a.amba

        system.gic_hub = SubSystem()
        system.gic_hub.gic_a2t = gic_a2t
        system.gic_hub.gic_t2g = gic_t2g
        system.gic_hub.gic_g2t = gic_g2t
        system.gic_hub.gic_t2a = gic_t2a

        self.voltage_domain = VoltageDomain(voltage=cpu_voltage)
        self.clk_domain = SrcClockDomain(
            clock=cpu_clock, voltage_domain=self.voltage_domain
        )

        # Setup CPU
        assert num_cpus <= 4
        CpuClasses = [
            FastModelCortexA76x1,
            FastModelCortexA76x2,
            FastModelCortexA76x3,
            FastModelCortexA76x4,
        ]
        CpuClass = CpuClasses[num_cpus - 1]

        cpu = CpuClass(
            GICDISABLE=False, BROADCASTATOMIC=False, BROADCASTOUTER=False
        )
        for core in cpu.cores:
            core.semihosting_enable = False
            core.RVBARADDR = 0x10
            core.redistributor = gic.redistributor
            core.createThreads()
            core.createInterruptController()
        self.cpus = [cpu]

        self.cpu_hub = SubSystem()
        a2t = AmbaToTlmBridge64(amba=cpu.amba)
        t2g = TlmToGem5Bridge64(tlm=a2t.tlm, gem5=system.membus.cpu_side_ports)
        self.cpu_hub.a2t = a2t
        self.cpu_hub.t2g = t2g

        system.addCpuCluster(self)

    def require_caches(self):
        return False

    def memory_mode(self):
        return "atomic_noncaching"

    def addL1(self):
        pass

    def addL2(self, clk_domain):
        pass

    def connectMemSide(self, bus):
        pass


class ClusterSystem:
    """
    Base class providing cpu clusters generation/handling methods to
    SE/FS systems
    """

    def __init__(self, **kwargs):
        self._clusters = []

    def numCpuClusters(self):
        return len(self._clusters)

    def addCpuCluster(self, cpu_cluster):
        self._clusters.append(cpu_cluster)
    
    def setL3Type(self, l3_type):
        self._l3_type = l3_type

    def addCaches(self, need_caches, last_cache_level, l3_size=None, slc_size=None, options = None):
        if not need_caches:
            # connect each cluster to the memory hierarchy
            for cluster in self._clusters:
                cluster.connectMemSide(self.membus)
            return
        if l3_size is None:
            l3_size = 0
        if slc_size is None:
            slc_size = 0
        cluster_mem_bus = self.membus
        assert last_cache_level >= 1 and last_cache_level <= 3
        for cluster in self._clusters:
            if issubclass(type(cluster), ArmCpuClusterWithMonitor):
                cluster.addL1(options)
            else:
                cluster.addL1()
        if last_cache_level > 1:
            for cluster in self._clusters:
                if issubclass(type(cluster), ArmCpuClusterWithMonitor):
                    cluster.addL2(cluster.clk_domain, options)
                else:
                    cluster.addL2(cluster.clk_domain)
        if last_cache_level > 2:
            max_clock_cluster = max(
                self._clusters, key=lambda c: c.clk_domain.clock[0]
            )
            self.l3 = L3(clk_domain=max_clock_cluster.clk_domain)
            self.l3.size = l3_size
            self.toL3Bus = CoherentXBar(
                clk_domain = max_clock_cluster.clk_domain,
                frontend_latency = 1,
                forward_latency = 0,
                response_latency = 1,
                snoop_response_latency = 1,
                width = 128,
                point_of_coherency = False,
                point_of_unification = False,
                max_outstanding_snoops = 490,
            )
            self.toL3Bus.snoop_filter = SnoopFilter(lookup_latency=0, max_capacity="16MiB")
            self.l3.cpu_side = self.toL3Bus.mem_side_ports
            cluster_mem_bus = self.toL3Bus
            # connect each cluster to the memory hierarchy
            for cluster in self._clusters:
                if issubclass(type(cluster), ArmCpuClusterWithMonitor):
                    cluster.connectMemSide(cluster_mem_bus, options)
                else:
                    cluster.connectMemSide(cluster_mem_bus)

            self.toSLCBus = CoherentXBar(
                clk_domain = max_clock_cluster.clk_domain,
                frontend_latency = 1,
                forward_latency = 0,
                response_latency = 1,
                snoop_response_latency = 1,
                width = 128,
                point_of_coherency = False,
                point_of_unification = False,
                max_outstanding_snoops = 84
            )
            opt_l3_monitor = getattr(options, "l3_monitor", False)
            if opt_l3_monitor:
                self.l3.monitor = CommMonitor()
                self.l3.monitor.footprint = MemFootprintProbe()
                self.l3.mem_side = self.l3.monitor.cpu_side_port
                self.l3.monitor.mem_side_port = self.toSLCBus.cpu_side_ports
            else:
                self.l3.mem_side = self.toSLCBus.cpu_side_ports

            self.slc = SLC()
            self.slc.size = slc_size
            self.slc.cpu_side = self.toSLCBus.mem_side_ports

            opt_slc_monitor = getattr(options, "slc_monitor", False)
            if opt_slc_monitor:
                self.slc.monitor = CommMonitor()
                self.slc.monitor.footprint = MemFootprintProbe()
                self.slc.mem_side = self.slc.monitor.cpu_side_port
                self.slc.monitor.mem_side_port = self.membus.cpu_side_ports
            else:
                self.slc.mem_side = self.membus.cpu_side_ports

        # set the membus as PoC
        self.membus.point_of_coherency = True
        self.membus.point_of_unification = True

class SimpleSeSystem(System, ClusterSystem):
    """
    Example system class for syscall emulation mode
    """

    # Use a fixed cache line size of 64 bytes
    cache_line_size = 64

    def __init__(self, **kwargs):
        System.__init__(self, **kwargs)
        ClusterSystem.__init__(self, **kwargs)
        # Create a voltage and clock domain for system components
        self.voltage_domain = VoltageDomain(voltage="3.3V")
        self.clk_domain = SrcClockDomain(
            clock="1GHz", voltage_domain=self.voltage_domain
        )

        # Create the off-chip memory bus.
        self.membus = SystemXBar()

    def connect(self):
        self.system_port = self.membus.cpu_side_ports


class BaseSimpleSystem(ArmSystem, ClusterSystem):
    cache_line_size = 64

    def __init__(self, mem_size, platform, **kwargs):
        ArmSystem.__init__(self, **kwargs)
        ClusterSystem.__init__(self, **kwargs)

        self.voltage_domain = VoltageDomain(voltage="1.0V")
        self.clk_domain = SrcClockDomain(
            clock="1GHz", voltage_domain=Parent.voltage_domain
        )

        if platform is None:
            self.realview = VExpress_GEM5_V1()
        else:
            self.realview = platform

        if hasattr(self.realview.gic, "cpu_addr"):
            self.gic_cpu_addr = self.realview.gic.cpu_addr

        self.terminal = Terminal(port=4567)
        self.vncserver = VncServer()

        self.iobus = IOXBar()
        # Device DMA -> MEM
        self.mem_ranges = self.getMemRanges(int(Addr(mem_size)))

    def getMemRanges(self, mem_size):
        """
        Define system memory ranges. This depends on the physical
        memory map provided by the realview platform and by the memory
        size provided by the user (mem_size argument).
        The method is iterating over all platform ranges until they cover
        the entire user's memory requirements.
        """
        mem_ranges = []
        for mem_range in self.realview._mem_regions:
            size_in_range = min(mem_size, mem_range.size())

            mem_ranges.append(
                AddrRange(start=mem_range.start, size=size_in_range)
            )

            mem_size -= size_in_range
            if mem_size == 0:
                return mem_ranges

        raise ValueError("memory size too big for platform capabilities")


class SimpleSystem(BaseSimpleSystem):
    """
    Meant to be used with the classic memory model
    """

    def __init__(self, caches, mem_size, platform=None, **kwargs):
        super().__init__(mem_size, platform, **kwargs)

        self.membus = MemBus()
        # CPUs->PIO
        self.iobridge = Bridge(delay="50ns")

        self._caches = caches
        self.dmabridge = Bridge(delay="50ns", ranges=self.mem_ranges)
        if hasattr(self, "coherent_io"):
            self.coherent_io = False
        
    def connect(self):
        self.iobridge.mem_side_port = self.iobus.cpu_side_ports
        self.iobridge.cpu_side_port = self.membus.mem_side_ports

        self.dmabridge.mem_side_port = self.membus.cpu_side_ports
        self.dmabridge.cpu_side_port = self.iobus.mem_side_ports

        if hasattr(self.realview.gic, "cpu_addr"):
            self.gic_cpu_addr = self.realview.gic.cpu_addr
        self.realview.attachOnChipIO(self.membus, self.iobridge)
        self.realview.attachIO(self.iobus)
        self.system_port = self.membus.cpu_side_ports

    def attach_pci(self, dev):
        self.realview.attachPciDevice(dev, self.iobus)


class ArmRubySystem(BaseSimpleSystem):
    """
    Meant to be used with ruby
    """

    def __init__(self, mem_size, platform=None, **kwargs):
        super().__init__(mem_size, platform, **kwargs)
        self._dma_ports = []
        self._mem_ports = []

    def connect(self):
        self.realview.attachOnChipIO(
            self.iobus, dma_ports=self._dma_ports, mem_ports=self._mem_ports
        )

        self.realview.attachIO(self.iobus, dma_ports=self._dma_ports)

        for cluster in self._clusters:
            for i, cpu in enumerate(cluster.cpus):
                self.ruby._cpu_ports[i].connectCpuPorts(cpu)

    def attach_pci(self, dev):
        self.realview.attachPciDevice(
            dev, self.iobus, dma_ports=self._dma_ports
        )
