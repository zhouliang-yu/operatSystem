#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[]) {

    int status;
    /* fork a child process */
    pid_t pid;

    printf("process start to fork\n");
    pid = fork();

    /* execute test program */
    if (pid == -1) {
        perror("fork error");
        exit(1);
    } else {
        // child procss
        if (pid == 0) {
            int i;
            char *arg[argc];

            printf("I'm the Child process, my pid = %d \n", getpid());

            for (i = 0; i < argc - 1; i++) {
                arg[i] = argv[i + 1];
            }
            arg[argc - 1] = NULL;
            printf("child process start to execute test program \n");

            execve(arg[0], arg, NULL);


            printf("continue to run original child process!\n");
            perror("execve");
            exit(EXIT_FAILURE);

        } else {
            printf("I'm the Parent process, my pid = %d \n", getpid());

            /* wait for child process terminates */
            waitpid(pid, &status, WUNTRACED);
            printf("Parent process receiving the SIGCHILD signal\n");

            if (WIFEXITED(status)) {
                printf("normal termination with Exit STATUS = %d \n", WEXITSTATUS(status));
            }
            else if(WIFSIGNALED(status)) {
                printf("CHILD EXECUTION FAILED %d\n", WTERMSIG(status));
                if (WTERMSIG(status) == 1) {
					printf("child process get SIGHUP signal\n");
					printf("child process is terminated by hangup signal\n");
					printf("child process exection failed\n");
				}
				else if (WTERMSIG(status) == 2) {
					printf("child process get SIGINT signal\n");
					printf("child process is terminated by interrupt signal\n");
					printf("child process exection failed\n");
				}
				else if (WTERMSIG(status) == 3) {
					printf("child process get SIGQUIT signal\n");
					printf("child process is terminated by quit signal\n");
					printf("child process exection failed\n");
				}
				else if (WTERMSIG(status) == 4) {
					printf("child process get SIGILL signal\n");
					printf("child process is terminated by illegal instruction signal\n");
					printf("child process exection failed\n");
				}
				else if (WTERMSIG(status) == 5) {
					printf("child process get SIGTRAP signal\n");
					printf("child process is terminated by trap signal\n");
					printf("child process exection failed\n");
				}
				else if (WTERMSIG(status) == 6) {
					printf("child process get SIGABRT signal\n");
					printf("child process is terminated by abort signal\n");
					printf("child process exection failed\n");
				}
				else if (WTERMSIG(status) == 7) {
					printf("child process get SIGBUS signal\n");
					printf("child process is terminated by bus signal\n");
					printf("child process exection failed\n");
				}
				else if (WTERMSIG(status) == 8) {
					printf("child process get SIGFPE signal\n");
					printf("child process is terminated by floating point exception signal\n");
					printf("child process exection failed\n");
				}
				else if (WTERMSIG(status) == 9) {
					printf("child process get SIGKILL signal\n");
					printf("child process is terminated by kill signal\n");
					printf("child process exection failed\n");
				}
				else if (WTERMSIG(status) == 11) {
					printf("child process get SIGSEGV signal\n");
					printf("child process is terminated by segmentation violation signal\n");
					printf("child process exection failed\n");
				}
				else if (WTERMSIG(status) == 13) {
					printf("child process get SIGPIPE signal\n");
					printf("child process is terminated by pipe signal\n");
					printf("child process exection failed\n");
				}
				else if (WTERMSIG(status) == 14) {
					printf("child process get SIGAlRM signal\n");
					printf("child process is terminated by alarm signal\n");
					printf("child process exection failed\n");
				}
				else if (WTERMSIG(status) == 15) {
					printf("child process get SIGTERM signal\n");
					printf("child process is terminated by termination signal\n");
					printf("child process exection failed\n");
				}
            }
            else if(WIFSTOPPED(status)) {
				printf("CHILD PROCESS get the SIGSTOP signal \n");
                printf("CHILD PROCESS STOPPED\n");
            }
			else{
                printf("CHILD PROCESS CONTINUED\n");
            }
            exit(0);
        }
    }
    return 0;

    /* check child process'  termination status */
}
