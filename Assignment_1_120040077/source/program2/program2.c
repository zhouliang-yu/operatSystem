#include <linux/module.h>
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
#include <linux/signal.h>
MODULE_LICENSE("GPL");



static struct task_struct *task;


struct wait_opts { enum pid_type wo_type; 
					int wo_flags; 
					struct pid *wo_pid; 
					struct siginfo __user *wo_info;
					int __user *wo_stat; 
					struct rusage __user *wo_rusage;
					wait_queue_t child_wait;
					int notask_error;
					};



extern long do_wait(struct wait_opts *wo);
extern long _do_fork(unsigned long clone_flags, unsigned long stack_start, unsigned long stack_size, int __user *parent_tidptr, int __user *child_tidptr, unsigned long tls);
extern int do_execve(struct filename *filename,
	const char __user *const __user *__argv,
	const char __user *const __user *__envp);

extern struct filename * getname(const char __user * filename);




void my_wait(pid_t pid) {

	int status;
	struct wait_opts wo;
	struct pid *wo_pid = NULL;
	enum pid_type type;
	type = PIDTYPE_PID;
	wo_pid = find_get_pid(pid);

	wo.wo_type = type;
	wo.wo_pid = wo_pid;
	wo.wo_flags = WUNTRACED | WEXITED;
	wo.wo_info = NULL;
	wo.wo_stat = (int __user*)&status;
	wo.wo_rusage = NULL;

	int a;
	a = do_wait(&wo);
	
	if(*wo.wo_stat == 0) {
		printk("[program2] : get NORMAL signal \n");
		printk("[program2] : child process exit normally\n");
	}else if(*wo.wo_stat == 1) {
		printk("[program2] : get SIGHUP signal \n");
		printk("[program2] : child process is hung up\n");
	}else if(*wo.wo_stat == 2) {
		printk("[program2] : get SIGINT signal \n");
		printk("[program2] : child process interrupt\n");
	}else if(*wo.wo_stat == 131) {
		printk("[program2] : get SIGQUIT signal \n");
		printk("[program2] : child process quit\n");
	}else if(*wo.wo_stat == 132) {
		printk("[program2] : get SIGILL signal \n");
		printk("[program2] : child process illegal instruction \n");
	}else if(*wo.wo_stat == 133) {
		printk("[program2] : get SIGTRAP signal \n");
		printk("[program2] : child process trapped \n");
	}else if(*wo.wo_stat == 134) {
		printk("[program2] : get SIGABRT signal \n");
		printk("[program2] : child process abort error \n");
	}else if(*wo.wo_stat == 135) {
		printk("[program2] : get SIGBUS signal \n");
		printk("[program2] : child process bus error \n");
	}else if(*wo.wo_stat == 136) {
		printk("[program2] : get SIGFPE signal \n");
		printk("[program2] : child process float error \n");
	}else if(*wo.wo_stat == 9) {
		printk("[program2] : get SIGKILL signal \n");
		printk("[program2] : child process killed\n");
	}else if(*wo.wo_stat == 139) {
		printk("[program2] : get SIGSEGV signal \n");
		printk("[program2] : child process segmentation fault error\n");
	}else if(*wo.wo_stat == 13) {
		printk("[program2] : get SIGPIPE signal \n");
		printk("[program2] : child process has pipe error\n");
	}else if(*wo.wo_stat == 14) {
		printk("[program2] : get SIGALARM signal \n");
		printk("[program2] : child process has alarm error\n");
	}else if(*wo.wo_stat == 15) {
		printk("[program2] : get SIGTERM signal \n");
		printk("[program2] : child process terminated\n");
	}else if (*wo.wo_stat == 4991) {
		printk("[program2] : get SIGSTOP signal \n");
		printk("[program2] : child process stop\n");
	}


	printk("[program2] : child process terminated\n");
	//output child process exit status
	if (*wo.wo_stat < 128) {
                printk("[program2]: The return signal is %d\n", *wo.wo_stat);
        }else if(*wo.wo_stat > 128 & *wo.wo_stat != 4991){
                printk("[program2]: The return signal is %d\n", *wo.wo_stat - 128);
        }else{
                printk("[program2]: The return signal is %d\n", 19);
        }
        put_pid(wo_pid);

        return;

}


int my_exec(void){
	int result;
	const char path[] = "/opt/test";
	const char *const argv[] = {path, NULL, NULL};
	const char *const envp[] = {"HOME=/", "PATH =/sbin:/user/sbin:/bin:/usr/bin", NULL};

	struct filename * my_filename = getname(path);

	printk("[program2] : child process");

	result = do_execve(my_filename, argv, envp);

	if (!result){
		return 0;
	}

	do_exit(result);
}

//implement fork function
int my_fork(void *argc){
	
	
	//set default sigaction for current process
	int i;
	long pid = 0;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for(i=0;i<_NSIG;i++){
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}
	
	/* fork a process using do_fork */
	pid = _do_fork(SIGCHLD, (unsigned long)&my_exec, 0, NULL, NULL, 0);

	printk("[program2]: The child process has pid = %ld\n", pid);
	printk("[program2]: This is the parent process, pid = %d \n", (int)current->pid);
	/* execute a test program in child process */
	// my_exec(); 
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
		printk("[program2] : module_init kthread start\n");
		wake_up_process(task);
	}
	return 0;
}

static void __exit program2_exit(void){
	printk("[program2] : Module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);
