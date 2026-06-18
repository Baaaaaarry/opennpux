GemminiDevA Linux driver (for gem5 FS)

Build (inside guest or cross-compile):
  make

Load module (MMIO base must match gem5 config):
  sudo insmod gemmini_dev_a_drv.ko mmio_base=0x40000000 mmio_size=0x1000

Run the user test:
  sudo ./gemmini_dev_a_user 2

Notes:
- The driver exposes /dev/gemmini_dev_a and uses an ioctl to run one request.
- The request uses DMA buffers allocated by the driver and copied from/to user.
- Pin the process to the CPU connected to GemminiDevA (e.g., taskset -c 0).
