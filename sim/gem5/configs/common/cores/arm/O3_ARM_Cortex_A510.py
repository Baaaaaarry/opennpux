from m5.objects import *
from m5.objects.ArmMMU import ArmMMU
from m5.proxy import *
from .O3_ARM_Monitor import ArmO3CPUWithMonitor

# Simple ALU Instructions have a latency of 1
class O3_ARM_Cortex_A510_Simple_Int(FUDesc):
    opList = [OpDesc(opClass="IntAlu", opLat=1)]
    count = 3

# Complex ALU instructions have a variable latencies
class O3_ARM_Cortex_A510_Complex_Int(FUDesc):
    opList = [
        OpDesc(opClass="IntMult", opLat=3, pipelined=True),
        OpDesc(opClass="IntDiv", opLat=12, pipelined=False),
        OpDesc(opClass="IprAccess", opLat=3, pipelined=True),
    ]
    count = 1

# Floating point and SIMD instructions
class O3_ARM_Cortex_A510_FP(FUDesc):
    opList = [
        OpDesc(opClass="FloatAdd", opLat=5),
        OpDesc(opClass="FloatCmp", opLat=5),
        OpDesc(opClass="FloatCvt", opLat=5),
        OpDesc(opClass="FloatDiv", opLat=9, pipelined=False),
        OpDesc(opClass="FloatSqrt", opLat=33, pipelined=False),
        OpDesc(opClass="FloatMult", opLat=4),
        OpDesc(opClass="FloatMultAcc", opLat=5),
        OpDesc(opClass="FloatMisc", opLat=3),
    ]
    count = 1

class O3_ARM_Cortex_A510_SIMD(SIMD_Unit):
    count = 1

# Load/Store Units
class O3_ARM_Cortex_A510_Load(FUDesc):
    opList = [
        OpDesc(opClass="MemRead", opLat=2),
        OpDesc(opClass="FloatMemRead", opLat=2),
    ]
    count = 2

class O3_ARM_Cortex_A510_Store(FUDesc):
    opList = [
        OpDesc(opClass="MemWrite", opLat=2),
        OpDesc(opClass="FloatMemWrite", opLat=2),
    ]
    count = 1

# Functional Units for this CPU
class O3_ARM_Cortex_A510_FUP(FUPool):
    FUList = [
        O3_ARM_Cortex_A510_Simple_Int(),
        O3_ARM_Cortex_A510_Complex_Int(),
        O3_ARM_Cortex_A510_Load(),
        O3_ARM_Cortex_A510_Store(),
        O3_ARM_Cortex_A510_FP(),
        O3_ARM_Cortex_A510_SIMD(),
    ]

class O3_ARM_Cortex_A510_BTB(SimpleBTB):
    numEntries = 1024
    tagBits = 20
    associativity = 4
    instShiftAmt = 2
    btbReplPolicy = LRURP()
    btbIndexingPolicy = BTBSetAssociative(
        num_entries=Parent.numEntries,
        set_shift=Parent.instShiftAmt,
        assoc=Parent.associativity,
        tag_bits=Parent.tagBits,
    )

# Bi-Mode Branch Predictor
class O3_ARM_Cortex_A510_BP(BiModeBP):
    btb = O3_ARM_Cortex_A510_BTB()
    ras = ReturnAddrStack(numEntries=8)
    globalPredictorSize = 4096
    globalCtrBits = 2
    choicePredictorSize = 4096
    choiceCtrBits = 2
    instShiftAmt = 2
    # privatePredictorSize = 16384
    # privateCtrBits = 2

class O3_ARM_Cortex_A510(ArmO3CPUWithMonitor):
    LQEntries = 16
    SQEntries = 16
    LSQDepCheckShift = 0
    LFSTSize = 1024
    SSITSize = "1024"
    decodeToFetchDelay = 1
    renameToFetchDelay = 1
    iewToFetchDelay = 1
    commitToFetchDelay = 1
    renameToDecodeDelay = 1
    iewToDecodeDelay = 1
    commitToDecodeDelay = 1
    iewToRenameDelay = 1
    commitToRenameDelay = 1
    commitToIEWDelay = 1
    fetchWidth = 4
    fetchBufferSize = 16
    fetchToDecodeDelay = 3
    decodeWidth = 3
    decodeToRenameDelay = 1
    renameWidth = 3
    renameToIEWDelay = 1
    issueToExecuteDelay = 1
    dispatchWidth = 3
    issueWidth = 3
    wbWidth = 3
    fuPool = O3_ARM_Cortex_A510_FUP()
    iewToCommitDelay = 1
    renameToROBDelay = 1
    commitWidth = 3
    squashWidth = 3
    trapLatency = 13
    backComSize = 5
    forwardComSize = 5
    numPhysIntRegs = 64
    numPhysFloatRegs = 64
    numPhysVecRegs = 64
    numIQEntries = 64
    numROBEntries = 128

    switched_out = False
    branchPred = O3_ARM_Cortex_A510_BP()

# Instruction Cache
class O3_ARM_Cortex_A510_ICache(Cache):
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 4
    tgts_per_mshr = 8
    size = "32KiB"
    assoc = 4
    is_read_only = True
    # Writeback clean lines as well
    writeback_clean = True

# Data Cache
class O3_ARM_Cortex_A510_DCache(Cache):
    tag_latency = 2
    data_latency = 2
    response_latency = 2
    mshrs = 4
    tgts_per_mshr = 8
    size = "32KiB"
    assoc = 4
    write_buffers = 16
    # Consider the L2 a victim cache also for clean lines
    writeback_clean = True

# L2 Cache
class O3_ARM_Cortex_A510L2(Cache):
    tag_latency = 12
    data_latency = 12
    response_latency = 12
    mshrs = 32
    tgts_per_mshr = 16
    size = "128KiB"
    assoc = 8
    write_buffers = 8
    clusivity = "mostly_excl"
    writeback_clean = True
    # Simple stride prefetcher
    #prefetcher = StridePrefetcher(degree=8, latency=1, prefetch_on_access=True)
    tags = BaseSetAssoc()
    replacement_policy = LRURP()