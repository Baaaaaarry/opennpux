// SPDX-License-Identifier: GPL-2.0
/*
 * Minimal OpenNPUX Coral platform driver.
 *
 * This is the Phase-3 kernel boundary behind the host runtime API. The shared
 * DMA window is exposed through a bounded mmap. Completion is
 * delivered through poll; delayed work is used until the platform wires a
 * real interrupt.
 */

#include <linux/atomic.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/jiffies.h>
#include <linux/mm.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/of.h>
#include <linux/of_address.h>
#include <linux/poll.h>
#include <linux/platform_device.h>
#include <linux/uaccess.h>
#include <linux/version.h>
#include <linux/workqueue.h>

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
	resource_size_t shared_start;
	resource_size_t shared_size;
	dev_t devt;
	struct cdev cdev;
	struct device *class_dev;
	wait_queue_head_t completion_wq;
	struct delayed_work completion_work;
	struct mutex state_lock;
	atomic_t running;
	atomic_t open_count;
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

static void coral_completion_work(struct work_struct *work)
{
	struct opennpux_coral_dev *coral = container_of(
		to_delayed_work(work), struct opennpux_coral_dev,
		completion_work);
	u32 status;

	if (!atomic_read(&coral->running))
		return;
	status = coral_readl(coral, STATUS);

	if (status & 0x3) {
		atomic_set(&coral->running, 0);
		wake_up_interruptible_poll(&coral->completion_wq,
					   POLLIN | POLLRDNORM | POLLERR);
		return;
	}

	if (atomic_read(&coral->running))
		schedule_delayed_work(&coral->completion_work,
				      msecs_to_jiffies(1));
}

static int coral_start(struct opennpux_coral_dev *coral, u32 entry)
{
	int ret = 0;

	mutex_lock(&coral->state_lock);
	if (atomic_read(&coral->running)) {
		ret = -EBUSY;
		goto out;
	}

	/* A poll consumer may finish before the delayed worker runs. */
	cancel_delayed_work_sync(&coral->completion_work);
	atomic_set(&coral->running, 1);

	coral_writel(coral, PC_START, entry);
	coral_writel(coral, RESET_CONTROL, 1);
	coral_writel(coral, RESET_CONTROL, 0);
	schedule_delayed_work(&coral->completion_work, 0);
out:
	mutex_unlock(&coral->state_lock);
	return ret;
}

static void coral_reset(struct opennpux_coral_dev *coral)
{
	mutex_lock(&coral->state_lock);
	atomic_set(&coral->running, 0);
	cancel_delayed_work_sync(&coral->completion_work);
	coral_writel(coral, RESET_CONTROL, 1);
	wake_up_interruptible_poll(&coral->completion_wq, POLLERR);
	mutex_unlock(&coral->state_lock);
}

static int coral_run(struct opennpux_coral_dev *coral,
		     struct opennpux_coral_ioc_run *run)
{
	u64 i;
	u32 status = 0;
	int ret = 0;

	if (atomic_cmpxchg(&coral->running, 0, 1) != 0)
		return -EBUSY;

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
		ret = -EIO;
	else if (!(status & 0x1))
		ret = -ETIMEDOUT;
	atomic_set(&coral->running, 0);
	return ret;
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
	case OPENNPUX_CORAL_IOC_GET_CAPS: {
		struct opennpux_coral_ioc_caps caps = {
			.abi_version = OPENNPUX_CORAL_ABI_VERSION,
			.features = OPENNPUX_CORAL_FEATURE_SHARED_MMAP |
				    OPENNPUX_CORAL_FEATURE_ASYNC_START |
				    OPENNPUX_CORAL_FEATURE_POLL_COMPLETION |
				    OPENNPUX_CORAL_FEATURE_RESET,
		};

		if (copy_to_user((void __user *)arg, &caps, sizeof(caps)))
			return -EFAULT;
		return 0;
	}
	case OPENNPUX_CORAL_IOC_START: {
		struct opennpux_coral_ioc_start start;

		if (copy_from_user(&start, (void __user *)arg, sizeof(start)))
			return -EFAULT;
		if (start.flags != 0)
			return -EINVAL;
		return coral_start(coral, start.entry);
	}
	case OPENNPUX_CORAL_IOC_RESET:
		coral_reset(coral);
		return 0;
	default:
		return -ENOTTY;
	}
}

static __poll_t coral_poll(struct file *file, poll_table *wait)
{
	struct opennpux_coral_dev *coral = file->private_data;
	u32 status;

	poll_wait(file, &coral->completion_wq, wait);
	status = coral_readl(coral, STATUS);
	if (status & 0x3)
		atomic_set(&coral->running, 0);
	if (status & 0x2)
		return POLLERR;
	if (status & 0x1)
		return POLLIN | POLLRDNORM;
	return 0;
}

static int coral_mmap(struct file *file, struct vm_area_struct *vma)
{
	struct opennpux_coral_dev *coral = file->private_data;
	resource_size_t offset = (resource_size_t)vma->vm_pgoff << PAGE_SHIFT;
	resource_size_t size = vma->vm_end - vma->vm_start;
	resource_size_t phys;

	if (!PAGE_ALIGNED(coral->shared_start) || !PAGE_ALIGNED(offset) ||
	    offset > coral->shared_size || size > coral->shared_size - offset)
		return -EINVAL;

	phys = coral->shared_start + offset;
	vma->vm_flags |= VM_IO | VM_DONTEXPAND | VM_DONTDUMP;
	vma->vm_page_prot = pgprot_noncached(vma->vm_page_prot);
	return remap_pfn_range(vma, vma->vm_start, phys >> PAGE_SHIFT, size,
			       vma->vm_page_prot);
}

static int coral_open(struct inode *inode, struct file *file)
{
	struct opennpux_coral_dev *coral =
		container_of(inode->i_cdev, struct opennpux_coral_dev, cdev);

	if (atomic_cmpxchg(&coral->open_count, 0, 1) != 0)
		return -EBUSY;
	file->private_data = coral;
	return 0;
}

static int coral_release(struct inode *inode, struct file *file)
{
	struct opennpux_coral_dev *coral = file->private_data;

	if (atomic_read(&coral->running))
		coral_reset(coral);
	atomic_set(&coral->open_count, 0);
	return 0;
}

static const struct file_operations coral_fops = {
	.owner = THIS_MODULE,
	.open = coral_open,
	.release = coral_release,
	.unlocked_ioctl = coral_ioctl,
	.poll = coral_poll,
	.mmap = coral_mmap,
	.llseek = no_llseek,
#ifdef CONFIG_COMPAT
	.compat_ioctl = coral_ioctl,
#endif
};

static int coral_probe(struct platform_device *pdev)
{
	struct opennpux_coral_dev *coral;
	struct resource *res;
	struct resource shared_res;
	struct device_node *memory_node;
	u32 dt_shared_base;
	u32 dt_shared_size;
	u32 shared_base;
	u32 shared_size;
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

	memory_node = of_parse_phandle(pdev->dev.of_node, "memory-region", 0);
	if (!memory_node) {
		ret = of_property_read_u32(pdev->dev.of_node,
					   "google,dma-shared-base",
					   &dt_shared_base);
		ret |= of_property_read_u32(pdev->dev.of_node,
					    "google,dma-shared-size",
					    &dt_shared_size);
		if (ret || !dt_shared_size) {
			dev_err(&pdev->dev,
				"missing or invalid memory-region and DMA window properties\n");
			return -EINVAL;
		}
		shared_res.start = dt_shared_base;
		shared_res.end = dt_shared_base + dt_shared_size - 1;
		shared_res.flags = IORESOURCE_MEM;
		dev_warn(&pdev->dev,
			 "memory-region phandle unresolved; using explicit DMA window\n");
	} else {
		ret = of_address_to_resource(memory_node, 0, &shared_res);
		of_node_put(memory_node);
		if (ret) {
			dev_err(&pdev->dev, "invalid memory-region: %d\n", ret);
			return ret;
		}
	}

	coral->shared_start = shared_res.start;
	coral->shared_size = resource_size(&shared_res);
	shared_base = coral_readl(coral, SHARED_BASE);
	shared_size = coral_readl(coral, SHARED_SIZE);
	if (shared_base != coral->shared_start ||
	    shared_size != coral->shared_size) {
		dev_err(&pdev->dev,
			"shared window CSR/DT mismatch csr=%#x/%#x dt=%pa/%pa\n",
			shared_base, shared_size, &coral->shared_start,
			&coral->shared_size);
		return -EINVAL;
	}

	init_waitqueue_head(&coral->completion_wq);
	INIT_DELAYED_WORK(&coral->completion_work, coral_completion_work);
	mutex_init(&coral->state_lock);
	atomic_set(&coral->running, 0);
	atomic_set(&coral->open_count, 0);

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
	dev_info(&pdev->dev, "shared DMA window at %pa size=%pa\n",
		 &coral->shared_start, &coral->shared_size);
	return 0;

del_cdev:
	cdev_del(&coral->cdev);
unregister_chrdev:
	unregister_chrdev_region(coral->devt, 1);
	return ret;
}

static void coral_remove_common(struct platform_device *pdev)
{
	struct opennpux_coral_dev *coral = platform_get_drvdata(pdev);

	coral_reset(coral);
	device_destroy(opennpux_coral_class, coral->devt);
	cdev_del(&coral->cdev);
	unregister_chrdev_region(coral->devt, 1);
}

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 11, 0)
static void coral_remove(struct platform_device *pdev)
{
	coral_remove_common(pdev);
}
#else
static int coral_remove(struct platform_device *pdev)
{
	coral_remove_common(pdev);
	return 0;
}
#endif

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

#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 4, 0)
	opennpux_coral_class = class_create(DRIVER_NAME);
#else
	opennpux_coral_class = class_create(THIS_MODULE, DRIVER_NAME);
#endif
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
