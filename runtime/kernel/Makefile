obj-m += opennpux_coral.o

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)
OPENNPUX_ROOT ?= $(abspath $(PWD)/../..)
ccflags-y += -I$(OPENNPUX_ROOT)/runtime/host/include

all:
	$(MAKE) -C $(KDIR) M=$(PWD) OPENNPUX_ROOT=$(OPENNPUX_ROOT) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
