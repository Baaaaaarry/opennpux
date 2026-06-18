obj-m += gemmini_dev_a_drv.o

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD := $(shell pwd)
CC ?= gcc
ccflags-y += -O2 -Wall -Wextra

all: module user

module:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

user: gemmini_dev_a_user

gemmini_dev_a_user: gemmini_dev_a_user.c gemmini_dev_a_ioctl.h
	$(CC) $(CFLAGS) -o $@ gemmini_dev_a_user.c

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean
	rm -f gemmini_dev_a_user
