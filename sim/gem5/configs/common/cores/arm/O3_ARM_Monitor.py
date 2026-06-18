from m5.objects import *
from m5.objects.ArmMMU import ArmMMU
from m5.proxy import *
        
class ArmO3CPUWithMonitor(ArmO3CPU):
    def connectMayWithMonitor(self, owner, mem_side, cpu_side, withMonitor = False, name = "monitor"):
        if withMonitor:
            m = CommMonitor()
            m.footprint = MemFootprintProbe()

            exec(f"owner.{name} = m")
            exec(f"owner.{name}.mem_side_port = cpu_side")
            exec(f"{mem_side} = owner.{name}.cpu_side_port")
        else:
            exec(f"{mem_side} = cpu_side")

    def connectCachedPorts(self, in_ports, options = None):
        opt_l1_monitor = getattr(options, "l1_monitor", False)

        for p in self._cached_ports:
            if p == "icache.mem_side":
                self.connectMayWithMonitor(self.icache, "self.icache.mem_side", in_ports, opt_l1_monitor)
            elif p == "dcache.mem_side":
                self.connectMayWithMonitor(self.dcache, "self.dcache.mem_side", in_ports, opt_l1_monitor)
            else:
                exec(f"self.{p} = in_ports")
    
    def addPrivateSplitL1Caches(self, ic, dc, iwc=None, dwc=None, options = None):
        opt_cpu_monitor = getattr(options, "cpu_monitor", False)

        self.icache = ic
        self.dcache = dc
        
        self.connectMayWithMonitor(self, "self.icache_port", ic.cpu_side, opt_cpu_monitor, name="iportmonitor")
        self.connectMayWithMonitor(self, "self.dcache_port", dc.cpu_side, opt_cpu_monitor, name="dportmonitor")

        self._cached_ports = ["icache.mem_side", "dcache.mem_side"]

        if iwc and dwc:
            self.itb_walker_cache = iwc
            self.dtb_walker_cache = dwc
            self.mmu.connectWalkerPorts(iwc.cpu_side, dwc.cpu_side)
            self._cached_ports += [
                "itb_walker_cache.mem_side",
                "dtb_walker_cache.mem_side",
            ]
        else:
            self._cached_ports += self.ArchMMU.walkerPorts()

        # Checker doesn't need its own tlb caches because it does
        # functional accesses only
        if self.checker != NULL:
            self._cached_ports += [
                "checker." + port for port in self.ArchMMU.walkerPorts()
            ]
