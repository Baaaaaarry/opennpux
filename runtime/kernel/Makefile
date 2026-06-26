obj-m += opennpux_coral.o

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) EXTRA_CFLAGS="-I$(PWD)/../host/include" modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
