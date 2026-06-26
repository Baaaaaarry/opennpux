// SPDX-License-Identifier: GPL-2.0
/*
 * Minimal OpenNPUX Coral platform driver.
 *
 * This is the Phase-3 kernel boundary behind the host runtime API. It exposes
 * info and run ioctls first; shared-window mmap and interrupt completion are
 * the next increments.
 */

#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/of_address.h>
#include <linux/platform_device.h>
#include <linux/uaccess.h>

#include "opennpux/coral_uapi.h"

#define DRIVER_NAME "opennpux-coral"

#define RESET_CONTROL 0x30000
#define PC_START 0x30004
#define STATUS 0x30008
#define DMA_ERRORS 0x30fe0
#define DMA_REQUESTS 0x30fe4
#define DMA_COMPLETIONS 0x30fe8
#define DMA_STATE 0x30fec
#define SHARED_BASE 0x30ff0
#define SHARED_SIZE 0x30ff4
#define FIRMWARE_ENTRY 0x30ff8
#define BACKEND_ID 0x30ffc

struct opennpux_coral_dev {
	struct device *dev;
	void __iomem *regs;
	resource_size_t regs_start;
	resource_size_t regs_size;
	dev_t devt;
	struct cdev cdev;
	struct device *class_dev;
};

static struct class *opennpux_coral_class;

static u32 coral_readl(struct opennpux_coral_dev *coral, u32 offset)
{
	return readl(coral->regs + offset);
}

static void coral_writel(struct opennpux_coral_dev *coral, u32 offset, u32 val)
{
	writel(val, coral->regs + offset);
}

static void coral_fill_info(struct opennpux_coral_dev *coral,
			    struct opennpux_coral_ioc_info *info)
{
	memset(info, 0, sizeof(*info));
	info->base = coral->regs_start;
	info->backend_id = coral_readl(coral, BACKEND_ID);
	info->firmware_entry = coral_readl(coral, FIRMWARE_ENTRY);
	info->shared_base = coral_readl(coral, SHARED_BASE);
	info->shared_size = coral_readl(coral, SHARED_SIZE);
	info->dma_requests = coral_readl(coral, DMA_REQUESTS);
	info->dma_completions = coral_readl(coral, DMA_COMPLETIONS);
	info->dma_errors = coral_readl(coral, DMA_ERRORS);
	info->dma_state = coral_readl(coral, DMA_STATE);
	info->reset_control = coral_readl(coral, RESET_CONTROL);
	info->status = coral_readl(coral, STATUS);
}

static int coral_run(struct opennpux_coral_dev *coral,
		     struct opennpux_coral_ioc_run *run)
{
	u64 i;
	u32 status = 0;

	coral_writel(coral, PC_START, run->entry);
	coral_writel(coral, RESET_CONTROL, 1);
	coral_writel(coral, RESET_CONTROL, 0);

	for (i = 0; i < run->polls; ++i) {
		status = coral_readl(coral, STATUS);
		if (status & 0x3)
			break;
		cpu_relax();
	}

	run->status = status;
	if (status & 0x2)
		return -EIO;
	if (!(status & 0x1))
		return -ETIMEDOUT;
	return 0;
}

static long coral_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
	struct opennpux_coral_dev *coral = file->private_data;

	switch (cmd) {
	case OPENNPUX_CORAL_IOC_GET_INFO: {
		struct opennpux_coral_ioc_info info;

		coral_fill_info(coral, &info);
		if (copy_to_user((void __user *)arg, &info, sizeof(info)))
			return -EFAULT;
		return 0;
	}
	case OPENNPUX_CORAL_IOC_RUN: {
		struct opennpux_coral_ioc_run run;
		int ret;

		if (copy_from_user(&run, (void __user *)arg, sizeof(run)))
			return -EFAULT;
		ret = coral_run(coral, &run);
		if (copy_to_user((void __user *)arg, &run, sizeof(run)))
			return -EFAULT;
		return ret;
	}
	default:
		return -ENOTTY;
	}
}

static int coral_open(struct inode *inode, struct file *file)
{
	struct opennpux_coral_dev *coral =
		container_of(inode->i_cdev, struct opennpux_coral_dev, cdev);

	file->private_data = coral;
	return 0;
}

static const struct file_operations coral_fops = {
	.owner = THIS_MODULE,
	.open = coral_open,
	.unlocked_ioctl = coral_ioctl,
#ifdef CONFIG_COMPAT
	.compat_ioctl = coral_ioctl,
#endif
};

static int coral_probe(struct platform_device *pdev)
{
	struct opennpux_coral_dev *coral;
	struct resource *res;
	int ret;

	coral = devm_kzalloc(&pdev->dev, sizeof(*coral), GFP_KERNEL);
	if (!coral)
		return -ENOMEM;

	coral->dev = &pdev->dev;
	res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
	if (!res)
		return -ENODEV;
	coral->regs_start = res->start;
	coral->regs_size = resource_size(res);
	coral->regs = devm_ioremap_resource(&pdev->dev, res);
	if (IS_ERR(coral->regs))
		return PTR_ERR(coral->regs);

	ret = alloc_chrdev_region(&coral->devt, 0, 1, DRIVER_NAME);
	if (ret)
		return ret;

	cdev_init(&coral->cdev, &coral_fops);
	coral->cdev.owner = THIS_MODULE;
	ret = cdev_add(&coral->cdev, coral->devt, 1);
	if (ret)
		goto unregister_chrdev;

	coral->class_dev = device_create(opennpux_coral_class, &pdev->dev,
					 coral->devt, coral,
					 "opennpux-coral");
	if (IS_ERR(coral->class_dev)) {
		ret = PTR_ERR(coral->class_dev);
		goto del_cdev;
	}

	platform_set_drvdata(pdev, coral);
	dev_info(&pdev->dev, "registered /dev/opennpux-coral at %pa size=%pa\n",
		 &coral->regs_start, &coral->regs_size);
	return 0;

del_cdev:
	cdev_del(&coral->cdev);
unregister_chrdev:
	unregister_chrdev_region(coral->devt, 1);
	return ret;
}

static int coral_remove(struct platform_device *pdev)
{
	struct opennpux_coral_dev *coral = platform_get_drvdata(pdev);

	device_destroy(opennpux_coral_class, coral->devt);
	cdev_del(&coral->cdev);
	unregister_chrdev_region(coral->devt, 1);
	return 0;
}

static const struct of_device_id coral_of_match[] = {
	{ .compatible = "google,coralnpu" },
	{ .compatible = "google,coralnpu-stagea" },
	{ }
};
MODULE_DEVICE_TABLE(of, coral_of_match);

static struct platform_driver coral_driver = {
	.probe = coral_probe,
	.remove = coral_remove,
	.driver = {
		.name = DRIVER_NAME,
		.of_match_table = coral_of_match,
	},
};

static int __init coral_init(void)
{
	int ret;

	opennpux_coral_class = class_create(THIS_MODULE, DRIVER_NAME);
	if (IS_ERR(opennpux_coral_class))
		return PTR_ERR(opennpux_coral_class);

	ret = platform_driver_register(&coral_driver);
	if (ret)
		class_destroy(opennpux_coral_class);
	return ret;
}

static void __exit coral_exit(void)
{
	platform_driver_unregister(&coral_driver);
	class_destroy(opennpux_coral_class);
}

module_init(coral_init);
module_exit(coral_exit);

MODULE_AUTHOR("OpenNPUX");
MODULE_DESCRIPTION("OpenNPUX Coral NPU platform driver");
MODULE_LICENSE("GPL");
