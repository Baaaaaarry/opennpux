from m5.objects import *
from m5.objects.ArmMMU import ArmMMU
from m5.proxy import *
from .O3_ARM_Monitor import ArmO3CPUWithMonitor

# Simple ALU Instructions have a latency of 1
class O3_ARM_Cortex_x4_Simple_Int(FUDesc):
    opList = [OpDesc(opClass="IntAlu", opLat=1)]
    count = 6

# Complex ALU instructions have a variable latencies
class O3_ARM_Cortex_x4_Complex_Int(FUDesc):
    opList = [
        OpDesc(opClass="IntMult", opLat=2, pipelined=True),
        OpDesc(opClass="IntDiv", opLat=12, pipelined=False),
        OpDesc(opClass="IprAccess", opLat=2, pipelined=True),
    ]
    count = 2

# Floating point and SIMD instructions
class O3_ARM_Cortex_x4_FP(FUDesc):
    opList = [
        OpDesc(opClass="FloatAdd", opLat=2),
        OpDesc(opClass="FloatCmp", opLat=2),
        OpDesc(opClass="FloatCvt", opLat=2),
        OpDesc(opClass="FloatDiv", opLat=6, pipelined=False),
        OpDesc(opClass="FloatSqrt", opLat=13, pipelined=False),
        OpDesc(opClass="FloatMult", opLat=3),
        OpDesc(opClass="FloatMultAcc", opLat=4),
        OpDesc(opClass="FloatMisc", opLat=2),
    ]
    count = 4

class O3_ARM_Cortex_x4_SIMD(SIMD_Unit):
    count = 2

# Load/Store Units
class O3_ARM_Cortex_x4_Load(FUDesc):
    opList = [
        OpDesc(opClass="MemRead", opLat=2),
        OpDesc(opClass="FloatMemRead", opLat=2),
    ]
    count = 3

class O3_ARM_Cortex_x4_Store(FUDesc):
    opList = [
        OpDesc(opClass="MemWrite", opLat=2),
        OpDesc(opClass="FloatMemWrite", opLat=2),
    ]
    count = 3

# Functional Units for this CPU
class O3_ARM_Cortex_x4_FUP(FUPool):
    FUList = [
        O3_ARM_Cortex_x4_Simple_Int(),
        O3_ARM_Cortex_x4_Complex_Int(),
        O3_ARM_Cortex_x4_Load(),
        O3_ARM_Cortex_x4_Store(),
        O3_ARM_Cortex_x4_FP(),
        O3_ARM_Cortex_x4_SIMD(),
    ]

class O3_ARM_Cortex_x4_BTB(SimpleBTB):
    numEntries = 8192
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
class O3_ARM_Cortex_x4_BP(BiModeBP):
    btb = O3_ARM_Cortex_x4_BTB()
    ras = ReturnAddrStack(numEntries=32)
    globalPredictorSize = 32768
    globalCtrBits = 2
    choicePredictorSize = 32768
    choiceCtrBits = 2
    instShiftAmt = 2
    # privatePredictorSize = 16384
    # privateCtrBits = 2

class O3_ARM_Cortex_x4(ArmO3CPUWithMonitor):
    LQEntries = 128
    SQEntries = 64
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
    fetchWidth = 10
    fetchBufferSize = 64
    fetchToDecodeDelay = 1
    decodeWidth = 10
    decodeToRenameDelay = 1
    renameWidth = 10
    renameToIEWDelay = 1
    issueToExecuteDelay = 1
    dispatchWidth = 10
    issueWidth = 10
    wbWidth = 10
    fuPool = O3_ARM_Cortex_x4_FUP()
    iewToCommitDelay = 1
    renameToROBDelay = 1
    commitWidth = 10
    squashWidth = 10
    trapLatency = 5
    backComSize = 5
    forwardComSize = 5
    numPhysIntRegs = 256
    numPhysFloatRegs = 256
    numPhysVecRegs = 256
    numIQEntries = 256
    numROBEntries = 384

    switched_out = False
    branchPred = O3_ARM_Cortex_x4_BP()
    mmu = ArmMMU(
        l2_shared=ArmTLB(
            entry_type="unified", size=2048, assoc=4, partial_levels=["L2"]
        ),
        itb=ArmTLB(
            entry_type="instruction", size=48, next_level=Parent.l2_shared
        ),
        dtb=ArmTLB(entry_type="data", size=48, next_level=Parent.l2_shared),
    )

# Instruction Cache
class O3_ARM_Cortex_x4_ICache(Cache):
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 32
    tgts_per_mshr = 16
    size = "64KiB"
    assoc = 4
    is_read_only = True
    # Writeback clean lines as well
    prefetcher = StridePrefetcher(degree=4, latency=1, prefetch_on_access=True)
    writeback_clean = True

# Data Cache
class O3_ARM_Cortex_x4_DCache(Cache):
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 32
    tgts_per_mshr = 16
    size = "64KiB"
    assoc = 4
    write_buffers = 32
    # Consider the L2 a victim cache also for clean lines
    # prefetcher = StridePrefetcher(degree=4, latency=1, prefetch_on_access=True)
    prefetcher = StridePrefetcher(
        on_miss=False, on_read=True, on_write=False, on_data=True, on_inst=False,
        latency=1, queue_size=64, queue_filter=True, queue_squash=True,
        cache_snoop=True, tag_prefetch=True,
        use_virtual_addresses=True, prefetch_on_access=True,

        table_assoc=4, table_entries="256",
        table_indexing_policy=StridePrefetcherHashedSetAssociative(
            entry_size=1, assoc=4, size="256"),
        table_replacement_policy=RandomRP(),

        degree=32,
        # stable 中该参数若不存在，可删除；存在时建议 20~30
        confidence_threshold=25,
    )
    writeback_clean = True

# L2 Cache
class O3_ARM_Cortex_x4L2(Cache):
    tag_latency = 1
    data_latency = 1
    response_latency = 1
    mshrs = 64
    tgts_per_mshr = 16
    size = "1MiB"
    assoc = 8
    write_buffers = 32
    writeback_clean = True
    clusivity = "mostly_excl"
    # Simple stride prefetcher
    # prefetcher = StridePrefetcher(degree=8, latency=1, prefetch_on_access=True)
    prefetcher = StridePrefetcher(
        on_miss=False, on_read=True, on_write=False, on_data=True, on_inst=False,
        latency=1, queue_size=96, queue_filter=True, queue_squash=True,
        cache_snoop=True, tag_prefetch=True,

        table_assoc=8, table_entries="1024",
        table_indexing_policy=StridePrefetcherHashedSetAssociative(
            entry_size=1, assoc=8, size="1024"),
        table_replacement_policy=RandomRP(),

        degree=32,
    )
    tags = BaseSetAssoc()
    replacement_policy = LRURP()