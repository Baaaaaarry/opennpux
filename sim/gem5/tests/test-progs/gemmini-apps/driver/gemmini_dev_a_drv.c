#include <linux/delay.h>
#include <linux/dma-mapping.h>
#include <linux/fs.h>
#include <linux/io.h>
#include <linux/jiffies.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/mutex.h>
#include <linux/of.h>
#include <linux/of_address.h>
#include <linux/overflow.h>
#include <linux/platform_device.h>
#include <linux/sched.h>
#include <linux/slab.h>
#include <linux/types.h>
#include <linux/uaccess.h>

#include "gemmini_dev_a_ioctl.h"

#define GEMMINI_REG_ADDR_M   0x00
#define GEMMINI_REG_ADDR_K   0x08
#define GEMMINI_REG_ADDR_O   0x10
#define GEMMINI_REG_SIZE_M   0x18
#define GEMMINI_REG_SIZE_K   0x20
#define GEMMINI_REG_OPCODE   0x28
#define GEMMINI_REG_START    0x30
#define GEMMINI_REG_STATUS   0x38

static unsigned long mmio_base = 0x40000000;
static unsigned long mmio_size = 0x1000;
static bool skip_claim = true;

module_param(mmio_base, ulong, 0444);
MODULE_PARM_DESC(mmio_base, "GemminiDevA MMIO base address");
module_param(mmio_size, ulong, 0444);
MODULE_PARM_DESC(mmio_size, "GemminiDevA MMIO size");
module_param(skip_claim, bool, 0444);
MODULE_PARM_DESC(skip_claim, "Skip request_mem_region when mapping MMIO");

struct gemmini_dev_a_dev {
    struct device *dev;
    void __iomem *regs;
    struct miscdevice misc;
    struct mutex lock;
};

static int gemmini_calc_sizes(
    const struct gemmini_dev_a_req *req,
    size_t *m_bytes,
    size_t *k_bytes,
    size_t *o_bytes)
{
    u64 m_elems = 0;
    u64 k_elems = 0;
    u64 o_elems = 0;
    u64 m_size = req->m_size;
    u64 k_size = req->k_size;
    u64 tmp = 0;
    u64 bytes = 0;

    if (m_size == 0)
        return -EINVAL;

    switch (req->opcode) {
    case GEMMINI_OP_CONV2D:
        if (k_size == 0)
            return -EINVAL;
        if (check_mul_overflow(m_size, m_size, &m_elems))
            return -EINVAL;
        if (check_mul_overflow(k_size, k_size, &k_elems))
            return -EINVAL;
        o_elems = m_elems;
        break;
    case GEMMINI_OP_CONV2D_GEMM:
        if (k_size == 0)
            return -EINVAL;
        if (check_mul_overflow(m_size, m_size, &tmp))
            return -EINVAL;
        if (check_mul_overflow(tmp, k_size, &tmp))
            return -EINVAL;
        if (check_mul_overflow(tmp, k_size, &m_elems))
            return -EINVAL;
        if (check_mul_overflow(k_size, k_size, &k_elems))
            return -EINVAL;
        if (check_mul_overflow(m_size, m_size, &o_elems))
            return -EINVAL;
        break;
    case GEMMINI_OP_CONV3D:
        if (k_size == 0)
            return -EINVAL;
        if (check_mul_overflow(m_size, m_size, &tmp))
            return -EINVAL;
        if (check_mul_overflow(tmp, m_size, &m_elems))
            return -EINVAL;
        if (check_mul_overflow(k_size, k_size, &tmp))
            return -EINVAL;
        if (check_mul_overflow(tmp, k_size, &k_elems))
            return -EINVAL;
        o_elems = m_elems;
        break;
    case GEMMINI_OP_CONV3D_GEMM:
        if (k_size == 0)
            return -EINVAL;
        if (check_mul_overflow(m_size, m_size, &tmp))
            return -EINVAL;
        if (check_mul_overflow(tmp, m_size, &tmp))
            return -EINVAL;
        if (check_mul_overflow(k_size, k_size, &k_elems))
            return -EINVAL;
        if (check_mul_overflow(k_elems, k_size, &k_elems))
            return -EINVAL;
        if (check_mul_overflow(tmp, k_elems, &m_elems))
            return -EINVAL;
        o_elems = tmp;
        break;
    case GEMMINI_OP_MAXPOOL:
    case GEMMINI_OP_MAXPOOL_GEMM:
        if (k_size == 0)
            return -EINVAL;
        if (m_size % k_size != 0)
            return -EINVAL;
        if (check_mul_overflow(m_size, m_size, &m_elems))
            return -EINVAL;
        o_elems = (m_size / k_size) * (m_size / k_size);
        k_elems = 0;
        break;
    case GEMMINI_OP_RELU:
        if (check_mul_overflow(m_size, m_size, &m_elems))
            return -EINVAL;
        o_elems = m_elems;
        k_elems = 0;
        break;
    case GEMMINI_OP_MM:
    case GEMMINI_OP_MM_GEMM:
        if (check_mul_overflow(m_size, m_size, &m_elems))
            return -EINVAL;
        k_elems = m_elems;
        o_elems = m_elems;
        break;
    default:
        return -EINVAL;
    }

    if (check_mul_overflow(m_elems, (u64)sizeof(float), &bytes))
        return -EINVAL;
    if (bytes > SIZE_MAX)
        return -EINVAL;
    *m_bytes = (size_t)bytes;

    if (k_elems) {
        if (check_mul_overflow(k_elems, (u64)sizeof(float), &bytes))
            return -EINVAL;
        if (bytes > SIZE_MAX)
            return -EINVAL;
        *k_bytes = (size_t)bytes;
    } else {
        *k_bytes = 0;
    }

    if (check_mul_overflow(o_elems, (u64)sizeof(float), &bytes))
        return -EINVAL;
    if (bytes > SIZE_MAX)
        return -EINVAL;
    *o_bytes = (size_t)bytes;

    return 0;
}

static int gemmini_wait_done(void __iomem *regs)
{
    unsigned long timeout = jiffies + msecs_to_jiffies(5000);

    while (!readq(regs + GEMMINI_REG_STATUS)) {
        if (time_after(jiffies, timeout))
            return -ETIMEDOUT;
        cpu_relax();
        cond_resched();
    }

    return 0;
}

static long gemmini_dev_a_ioctl(
    struct file *file, unsigned int cmd, unsigned long arg)
{
    struct gemmini_dev_a_dev *gdev = file->private_data;
    struct gemmini_dev_a_req req;
    size_t m_bytes = 0, k_bytes = 0, o_bytes = 0;
    void *m_buf = NULL;
    void *k_buf = NULL;
    void *o_buf = NULL;
    dma_addr_t m_dma = 0, k_dma = 0, o_dma = 0;
    int ret = 0;

    if (cmd != GEMMINI_DEV_A_IOC_RUN)
        return -ENOTTY;

    if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
        return -EFAULT;

    ret = gemmini_calc_sizes(&req, &m_bytes, &k_bytes, &o_bytes);
    if (ret)
        return ret;

    if (!req.m_user || !req.o_user)
        return -EINVAL;
    if (k_bytes && !req.k_user)
        return -EINVAL;

    mutex_lock(&gdev->lock);

    m_buf = dma_alloc_coherent(gdev->dev, m_bytes, &m_dma, GFP_KERNEL);
    if (!m_buf) {
        ret = -ENOMEM;
        goto out;
    }

    if (k_bytes) {
        k_buf = dma_alloc_coherent(gdev->dev, k_bytes, &k_dma, GFP_KERNEL);
        if (!k_buf) {
            ret = -ENOMEM;
            goto out;
        }
    }

    o_buf = dma_alloc_coherent(gdev->dev, o_bytes, &o_dma, GFP_KERNEL);
    if (!o_buf) {
        ret = -ENOMEM;
        goto out;
    }

    if (copy_from_user(m_buf, (void __user *)(uintptr_t)req.m_user, m_bytes)) {
        ret = -EFAULT;
        goto out;
    }

    if (k_bytes) {
        if (copy_from_user(k_buf, (void __user *)(uintptr_t)req.k_user, k_bytes)) {
            ret = -EFAULT;
            goto out;
        }
    }

    memset(o_buf, 0, o_bytes);

    writeq(m_dma, gdev->regs + GEMMINI_REG_ADDR_M);
    writeq(k_bytes ? k_dma : 0, gdev->regs + GEMMINI_REG_ADDR_K);
    writeq(o_dma, gdev->regs + GEMMINI_REG_ADDR_O);
    writeq(req.m_size, gdev->regs + GEMMINI_REG_SIZE_M);
    writeq(req.k_size, gdev->regs + GEMMINI_REG_SIZE_K);
    writeq(req.opcode, gdev->regs + GEMMINI_REG_OPCODE);
    writeq(1, gdev->regs + GEMMINI_REG_START);

    ret = gemmini_wait_done(gdev->regs);
    if (ret)
        goto out;

    if (copy_to_user((void __user *)(uintptr_t)req.o_user, o_buf, o_bytes))
        ret = -EFAULT;

out:
    if (o_buf)
        dma_free_coherent(gdev->dev, o_bytes, o_buf, o_dma);
    if (k_buf)
        dma_free_coherent(gdev->dev, k_bytes, k_buf, k_dma);
    if (m_buf)
        dma_free_coherent(gdev->dev, m_bytes, m_buf, m_dma);

    mutex_unlock(&gdev->lock);
    return ret;
}

static int gemmini_dev_a_open(struct inode *inode, struct file *file)
{
    struct miscdevice *misc = file->private_data;
    struct gemmini_dev_a_dev *gdev = container_of(misc, struct gemmini_dev_a_dev, misc);

    file->private_data = gdev;
    return 0;
}

static const struct file_operations gemmini_dev_a_fops = {
    .owner = THIS_MODULE,
    .open = gemmini_dev_a_open,
    .unlocked_ioctl = gemmini_dev_a_ioctl,
    .compat_ioctl = gemmini_dev_a_ioctl,
};

static int gemmini_dev_a_probe(struct platform_device *pdev)
{
    struct gemmini_dev_a_dev *gdev;
    struct resource *res;
    int ret;

    gdev = devm_kzalloc(&pdev->dev, sizeof(*gdev), GFP_KERNEL);
    if (!gdev)
        return -ENOMEM;

    if (skip_claim) {
        if (pdev->dev.of_node) {
            size_t res_size;

            res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
            if (!res)
                return -EINVAL;
            res_size = res->end - res->start + 1;
            gdev->regs = devm_ioremap(&pdev->dev, res->start, res_size);
        } else {
            if (!mmio_size)
                return -EINVAL;
            gdev->regs = devm_ioremap(&pdev->dev, mmio_base, mmio_size);
        }
        if (!gdev->regs)
            return -ENOMEM;
    } else {
        res = platform_get_resource(pdev, IORESOURCE_MEM, 0);
        gdev->regs = devm_ioremap_resource(&pdev->dev, res);
        if (IS_ERR(gdev->regs))
            return PTR_ERR(gdev->regs);
    }

    gdev->dev = &pdev->dev;
    mutex_init(&gdev->lock);

    ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(48));
    if (ret) {
        ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
        if (ret)
            return ret;
    }

    gdev->misc.minor = MISC_DYNAMIC_MINOR;
    gdev->misc.name = "gemmini_dev_a";
    gdev->misc.fops = &gemmini_dev_a_fops;
    gdev->misc.parent = &pdev->dev;

    ret = misc_register(&gdev->misc);
    if (ret)
        return ret;

    platform_set_drvdata(pdev, gdev);
    return 0;
}

static int gemmini_dev_a_remove(struct platform_device *pdev)
{
    struct gemmini_dev_a_dev *gdev = platform_get_drvdata(pdev);

    misc_deregister(&gdev->misc);
    return 0;
}

static const struct of_device_id gemmini_dev_a_of_match[] = {
    { .compatible = "gem5,gemmini-dev-a" },
    { },
};
MODULE_DEVICE_TABLE(of, gemmini_dev_a_of_match);

static struct platform_driver gemmini_dev_a_driver = {
    .probe = gemmini_dev_a_probe,
    .remove = gemmini_dev_a_remove,
    .driver = {
        .name = "gemmini-dev-a",
        .of_match_table = of_match_ptr(gemmini_dev_a_of_match),
    },
};

static struct platform_device *gemmini_pdev;
static bool using_dt;

static int __init gemmini_dev_a_init(void)
{
    struct resource res;
    int ret;

    if (of_find_compatible_node(NULL, NULL, "gem5,gemmini-dev-a")) {
        using_dt = true;
        return platform_driver_register(&gemmini_dev_a_driver);
    }

    if (!mmio_size)
        return -EINVAL;

    res.start = mmio_base;
    res.end = mmio_base + mmio_size - 1;
    res.flags = IORESOURCE_MEM;

    gemmini_pdev = platform_device_register_simple(
        "gemmini-dev-a", -1, &res, 1);
    if (IS_ERR(gemmini_pdev))
        return PTR_ERR(gemmini_pdev);

    ret = platform_driver_register(&gemmini_dev_a_driver);
    if (ret) {
        platform_device_unregister(gemmini_pdev);
        return ret;
    }

    return 0;
}

static void __exit gemmini_dev_a_exit(void)
{
    platform_driver_unregister(&gemmini_dev_a_driver);
    if (!using_dt && gemmini_pdev)
        platform_device_unregister(gemmini_pdev);
}

module_init(gemmini_dev_a_init);
module_exit(gemmini_dev_a_exit);

MODULE_DESCRIPTION("GemminiDevA NDP driver");
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Joao Vieira, adapted");
