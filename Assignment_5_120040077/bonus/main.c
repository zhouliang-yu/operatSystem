#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <linux/slab.h>
#include <linux/cdev.h>
#include <linux/delay.h>
#include <asm/uaccess.h>
#include "ioc_hw5.h"

MODULE_LICENSE("GPL");

#define PREFIX_TITLE "OS_AS5"

// DMA
#define DMA_BUFSIZE 64
#define DMASTUIDADDR 0x0	 // Student ID
#define DMARWOKADDR 0x4		 // RW function complete
#define DMAIOCOKADDR 0x8	 // ioctl function complete
#define DMAIRQOKADDR 0xc	 // ISR function complete
#define DMACOUNTADDR 0x10	 // interrupt count function complete
#define DMAANSADDR 0x14		 // Computation answer
#define DMAREADABLEADDR 0x18 // READABLE variable for synchronize
#define DMABLOCKADDR 0x1c	 // Blocking or non-blocking IO
#define DMAOPCODEADDR 0x20	 // data.a opcode
#define DMAOPERANDBADDR 0x21 // data.b operand1
#define DMAOPERANDCADDR 0x25 // data.c operand2
void *dma_buf;

#define DEV_NAME "mydev" // name for alloc_chrdev_region
#define DEV_BASEMINOR 0	 // baseminor for alloc_chrdev_region
#define DEV_COUNT 1		 // count for alloc_chrdev_region
#define INTERRUPT_DEV_NAME "myinterrupt"
#define IRQ_NUM 1

static int dev_major;
static int dev_minor;
static struct cdev *dev_cdev;
static int interrupt_t = 0;

// Declaration for file operations
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t, loff_t *);
static int drv_open(struct inode *, struct file *);
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t, loff_t *);
static int drv_release(struct inode *, struct file *);
static long drv_ioctl(struct file *, unsigned int, unsigned long);
static int prime(int, short);

// cdev file_operations
static struct file_operations fops = {
	owner : THIS_MODULE,
	read : drv_read,
	write : drv_write,
	unlocked_ioctl : drv_ioctl,
	open : drv_open,
	release : drv_release,
};

// in and out function
void myoutc(unsigned char data, unsigned short int port);
void myouts(unsigned short data, unsigned short int port);
void myouti(unsigned int data, unsigned short int port);
unsigned char myinc(unsigned short int port);
unsigned short myins(unsigned short int port);
unsigned int myini(unsigned short int port);

// Work routine
static struct work_struct *work_routine;

// For input data structure
struct DataIn
{
	char a;
	int b;
	short c;
} * dataIn;

// Arithmetic funciton
static void drv_arithmetic_routine(struct work_struct *ws);

// Input and output data from/to DMA
void myoutc(unsigned char data, unsigned short int port)
{
	*(volatile unsigned char *)(dma_buf + port) = data;
}
void myouts(unsigned short data, unsigned short int port)
{
	*(volatile unsigned short *)(dma_buf + port) = data;
}
void myouti(unsigned int data, unsigned short int port)
{
	*(volatile unsigned int *)(dma_buf + port) = data;
}
unsigned char myinc(unsigned short int port)
{
	return *(volatile unsigned char *)(dma_buf + port);
}
unsigned short myins(unsigned short int port)
{
	return *(volatile unsigned short *)(dma_buf + port);
}
unsigned int myini(unsigned short int port)
{
	return *(volatile unsigned int *)(dma_buf + port);
}

static int drv_open(struct inode *ii, struct file *ff)
{
	try_module_get(THIS_MODULE);
	printk("%s:%s(): device open\n", PREFIX_TITLE, __func__);
	return 0;
}
static int drv_release(struct inode *ii, struct file *ff)
{
	module_put(THIS_MODULE);
	printk("%s:%s(): device close\n", PREFIX_TITLE, __func__);
	return 0;
}

static ssize_t drv_read(struct file *filp, char __user *buffer, size_t ss, loff_t *lo)
{

	/* Implement read operation for your device */
	int IOMode = myini(DMAREADABLEADDR);
	if (IOMode == 1)
	{ // readable
		printk("%s:%s(): the answer is %i\n", PREFIX_TITLE, __func__, myini(DMAANSADDR));
		// put the computation result to user
		put_user(myini(DMAANSADDR), (int *)buffer);
		// clean the result
		myouti(0, DMASTUIDADDR);
		myouti(0, DMARWOKADDR);
		myouti(0, DMAIOCOKADDR);
		myouti(0, DMAIRQOKADDR);
		myouti(0, DMACOUNTADDR);
		myouti(0, DMAANSADDR);
		myouti(0, DMABLOCKADDR);
		myoutc(NULL, DMAOPCODEADDR);
		myouti(0, DMAOPERANDBADDR);
		myouts(0, DMAOPERANDCADDR);

		// set the readable as false
		myouti(0, DMAREADABLEADDR);
	}
	return 0;
}
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t ss, loff_t *lo)
{
	/* Implement write operation for your device */
	struct DataIn data;
	int IOMode = myini(DMAREADABLEADDR);

	get_user(data.a, (char *)buffer);
	get_user(data.b, (int *)buffer + 1);
	get_user(data.c, (short *)buffer + 2);

	// write data into DMA buffer with specific port
	myoutc(data.a, DMAOPCODEADDR);
	myouti(data.b, DMAOPERANDBADDR);
	myouts(data.c, DMAOPERANDCADDR);
	printk("%s:%s():queue work\n", PREFIX_TITLE, __func__);

	INIT_WORK(work_routine, drv_arithmetic_routine);

	if (IOMode)
	{
		// Blocking IO
		printk("%s:%s():block\n", PREFIX_TITLE, __func__);
		schedule_work(work_routine);
		flush_scheduled_work();
	}
	else
	{
		// non-blocking IO
		printk("%s:%s():block\n", PREFIX_TITLE, __func__);
		schedule_work(work_routine);
	}
	return 0;
}

static long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg)
{
	/* Implement ioctl setting for your device */
	int IOMode = myini(DMAREADABLEADDR);

	int signal; // store the result of ret to the signal
	get_user(signal, (int *)arg);

	if (cmd == HW5_IOCSETSTUID)
	{
		// STUID
		myouti(signal, DMASTUIDADDR);
		printk("%s:%s(): My student id = <%i>\n", PREFIX_TITLE, __func__, signal);
	}

	if (cmd == HW5_IOCSETRWOK)
	{
		if (signal == 0 || signal == 1)
		{
			// RW complete
			printk("%s:%s()RW OK\n", PREFIX_TITLE, __func__);
			myouti(signal, DMARWOKADDR);
		}
		else
		{
			printk("%s:%s(): RW Not Complete\n", PREFIX_TITLE, __func__);
			return -1;
		}
	}

	if (cmd == HW5_IOCSETBLOCK)
	{
		// complete set block or non-block
		if (signal == 0)
		{
			// non-blocking IO
			printk("%s,%s():Non-Blocking IO\n", PREFIX_TITLE, __func__);
			myouti(signal, HW5_IOCSETBLOCK);
		}

		if (signal == 1)
		{
			// blocking IO
			printk("%s,%s():Blocking IO\n", PREFIX_TITLE, __func__);
			myouti(signal, HW5_IOCSETBLOCK);
		}
	}

	if (cmd == HW5_IOCSETIRQOK)
	{
		// Complete the bonus task
		if (signal == 0 || signal == 1)
		{
			myouti(signal, DMAIRQOKADDR);
			printk("%s,%s(): IRQ OK\n", PREFIX_TITLE, __func__);
		}
		else
		{
			printk("%s,%s(): IRQ not complete \n", PREFIX_TITLE, __func__);
			return -1;
		}
	}

	if (cmd == HW5_IOCSETIOCOK)
	{
		if (signal == 0 || signal == 1)
		{
			myouti(signal, DMAIOCOKADDR);
			printk("%s,%s(): IOC OK\n", PREFIX_TITLE, __func__);
		}
		else
		{
			printk("%s,%s(): IOC not complete\n", PREFIX_TITLE, __func__);
			return -1;
		}
	}

	if (cmd == HW5_IOCWAITREADABLE)
	{
		while (IOMode == 0)
		{
			// blocking IO
			msleep(5000);
			IOMode = myini(DMAREADABLEADDR);
		}

		put_user(IOMode, (int *)arg);
		printk("%s,%s(): wait readable 1\n", PREFIX_TITLE, __func__);
	}

	return 0;
}

static void drv_arithmetic_routine(struct work_struct *ws)
{
	/* Implement arthemetic routine */
	struct DataIn data;
	int result;

	data.a = myinc(DMAOPCODEADDR);
	data.b = myini(DMAOPERANDBADDR);
	data.c = myini(DMAOPERANDCADDR);

	if (data.a == '+')
	{
		result = data.b + data.c;
	}
	else if (data.a == '-')
	{
		result = data.b - data.c;
	}
	else if (data.a == '*')
	{
		result = data.b * data.c;
	}
	else if (data.a == '/')
	{
		result = data.b / data.c;
	}
	else if (data.a == 'p')
	{
		result = prime(data.b, data.c);
	}
	else
	{
		result = -1;
	}

	myouti(result, DMAANSADDR);

	printk("%s:%s(): %d %c %d = %d\n", PREFIX_TITLE, __func__, data.b, data.a, data.c, result);

	myouti(1, DMAREADABLEADDR);
}

int prime(int base, short nth)
{
	int fnd = 0;
	int i, num, isPrime;

	num = base;
	while (fnd != nth)
	{
		isPrime = 1;
		num++;
		for (i = 2; i <= num / 2; i++)
		{
			if (num % i == 0)
			{
				isPrime = 0;
				break;
			}
		}

		if (isPrime)
		{
			fnd++;
		}
	}
	return num;
}

static irqreturn_t handler(int irq, void *dev_id)
{
	interrupt_t += 1;
	return IRQ_HANDLED;
}

static int __init init_modules(void)
{
	dev_t dev;
	int ret = 0;

	free_irq(IRQ_NUM, NULL);
	int irq = request_irq(IRQ_NUM, handler, IRQF_SHARED, INTERRUPT_DEV_NAME, (void *)handler);

	printk("%s:%s():...............Start...............\n", PREFIX_TITLE, __func__);

	printk("%s:%s(): request irq %d return %d\n", PREFIX_TITLE, __FUNCTION__, IRQ_NUM, irq);

	dev_cdev = cdev_alloc();

	/* Register chrdev */
	ret = alloc_chrdev_region(&dev, DEV_BASEMINOR, DEV_COUNT, DEV_NAME);
	if (ret)
	{
		printk("cannot alloc chrdev\n");
		return ret;
	}
	dev_major = MAJOR(dev);
	dev_minor = MINOR(dev);
	printk("%s : %s(): register chrdev(%d, %d)\n", PREFIX_TITLE, __func__, dev_major, dev_minor);

	/* Init cdev and make it alive */
	cdev_init(dev_cdev, &fops);
	dev_cdev->owner = THIS_MODULE;

	ret = cdev_add(dev_cdev, MKDEV(dev_major, dev_minor), 1);
	if (ret < 0)
	{
		printk("add chrdev failed\n");
		return -1;
	}

	/* Allocate DMA buffer */
	dma_buf = kzalloc(DMA_BUFSIZE, GFP_KERNEL);
	printk("%s:%s():allocate dma buffer\n", PREFIX_TITLE, __func__);

	/* Allocate work routine */
	work_routine = kmalloc(sizeof(typeof(*work_routine)), GFP_KERNEL);

	return 0;
}

static void __exit exit_modules(void)
{

	dev_t dev;
	free_irq(IRQ_NUM, (void *)(handler));
	printk("%s:%s(): interrupt count = %d\n", PREFIX_TITLE, __FUNCTION__, interrupt_t);

	dev = MKDEV(dev_major, dev_minor);
	cdev_del(dev_cdev);

	/* Free DMA buffer when exit modules */
	kfree(dma_buf);
	printk("%s:%s():free dma buffer\n", PREFIX_TITLE, __func__);

	/* Delete character device */
	printk("%s:%s():unregister chrdev\n", PREFIX_TITLE, __func__);
	unregister_chrdev_region(MKDEV(dev_major, dev_minor), DEV_COUNT);

	/* Free work routine */
	kfree(work_routine);

	printk("%s:%s():..............End..............\n", PREFIX_TITLE, __func__);
}

module_init(init_modules);
module_exit(exit_modules);