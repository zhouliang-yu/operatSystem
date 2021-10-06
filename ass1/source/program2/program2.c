#include <module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>

MODULE_LICENSE("GPL");

EXPORT_SYMBOL(do_fork);
EXPORT_SYMBOL(do_wait);
EXPORT_SYMBOL(do_execve);

extern long do_wait(struct wait_opts *wo);
extern long do_fork(unsigned long clone_flags, unsigned long stack_start, unsigned long stack_size, int __user *parent_tidptr, int __user *child_tidptr, unsigned long tls);
extern long do_execve(struct filename *filename, const char__user *const__user *__argv, const char __user *const__user *__envp);


extern

void my_wait(pid_t pid) {

    int status;
    struct wait_opts wo;
    struct pid *wo_pid = NULL;
    type = PIDTYPE_PID;
    wo_pid = find_get_pid(pid);

    wo.wo_type = type;
    wo.wo_pid = wo_pid;
    wo.wo_flags = WEXITED;
    wo.woinfo = NULL;
    wo.wo_stat = (int __user*)&status;
    wo.wo_rusage = NULL;

    int a;
    a = do_wait(&wo);
    printk("do_wait return value is %d\n", &a);

    //output child process exit status
    printk("[Do_Fork]: The return signal is %d\n", *wo.wo_stat);

    put_pid(wo_pid);

    return;
}


int my_exec(void){
    int result;
    const char path[] = "/home/zhouliang1/OS/operatSystem/Assigenment_1_120040077/sorce/program2/test.c"
    const char *const argv[] = {path, NULL, NULL};
    const char *const envp[] = {"HOME=/", "PATH =/sbin:/user/sbin:/bin:/usr/bin", NULL};

    struct filename * my_filename = getname(path);

    result = do_execve(my_filename, argv, envp);

    if (!result)
        return 0;
    do_exit(result);
}

//implement fork function
int my_fork(void *argc){


    //set default sigaction for current process
    int i;
    struct k_sigaction *k_action = &current->sighand->action[0];
    for(i=0;i<_NSIG;i++){
        k_action->sa.sa_handler = SIG_DFL;
        k_action->sa.sa_flags = 0;
        k_action->sa.sa_restorer = NULL;
        sigemptyset(&k_action->sa.sa_mask);
        k_action++;
    }

    /* fork a process using do_fork */
    pid = do_fork(SIGCHILD, (unsigned long)& my_exec, 0, NULL, NULL, 0);

    printk("[Do_Fork]: The child process has pid = %ld\n", pid);
    printk("[Do_Fork]: This is the parent process, pid = %d \n", int(current)->pid);
    /* execute a test program in child process */
    my_exec();
    /* wait until child process terminates */
    my_wait(pid);
    return 0;
}

static int __init program2_init(void){

    printk("[program2] : Module_init\n");

    /* write your code here */

    /* create a kernel thread to run my_fork */
    task = kthread_create(&my_fork, NULL, "MyThread"); //here it requires a parameter

    // wake up the new thread if ok
    if(!IS_ERR(task)) {
        printk("kthread starts\n");
        wake_up_process(task);
    }
    return 0;
}

static void __exit program2_exit(void){
    printk("[program2] : Module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);