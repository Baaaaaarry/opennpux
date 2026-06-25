# Copyright (c) 2025
# All rights reserved.
#
# Coral NPU stage-A model interface for gem5.
#
# This model exposes a single contiguous MMIO aperture covering the Coral NPU
# ITCM, DTCM, and external CSR windows. It intentionally does not execute real
# Coral instructions yet; instead it provides the boot/control contract and a
# fixed-latency run-to-halt state machine suitable for SoC bring-up.

from m5.objects.Device import DmaVirtDevice
from m5.params import *
from m5.util.fdthelper import *


class NPUDevice(DmaVirtDevice):
    type = "NPUDevice"
    cxx_header = "dev/npu/npu_device.hh"
    cxx_class = "gem5::NPUDevice"

    # Base address and size of the Coral NPU aperture.
    pioAddr = Param.Addr("Base physical address of the Coral NPU aperture")
    pioSize = Param.Addr(
        0x31000,
        "Size of the Coral NPU aperture covering ITCM/DTCM/CSR windows",
    )

    # Stage-A Coral memory map parameters.
    itcmSize = Param.MemorySize("8KiB", "Coral NPU ITCM size")
    dtcmSize = Param.MemorySize("32KiB", "Coral NPU DTCM size")
    pioDelay = Param.Latency("10ns", "Latency for Coral NPU MMIO accesses")
    executionLatency = Param.Latency(
        "1us", "Fixed run-to-halt latency once Coral NPU starts"
    )
    autoHalt = Param.Bool(
        True,
        "Automatically transition the stage-A Coral NPU model to halted",
    )
    backendType = Param.String(
        "stage-a",
        "Execution backend: 'stage-a' or future 'verilated-coral'",
    )
    coralRepo = Param.String(
        "",
        "Path to the external google-coral/coralnpu checkout used for RTL development",
    )
    verilatedWrapper = Param.String(
        "",
        "Path to the future Verilated Coral wrapper artifact or build directory",
    )
    rtlFirmware = Param.String(
        "",
        "Path to a Coral ELF loaded into the RTL TCM before execution",
    )
    rtlTickPeriod = Param.Latency(
        "1ns",
        "gem5 time between Verilated Coral backend events",
    )
    rtlCyclesPerEvent = Param.Unsigned(
        1,
        "Number of Coral RTL cycles evaluated per backend event",
    )

    def generateDeviceTree(self, state):
        node = self.generateBasicPioDeviceNode(
            state, "coralnpu", self.pioAddr, self.pioSize
        )
        node.appendCompatible(["google,coralnpu", "google,coralnpu-stagea"])
        yield node
