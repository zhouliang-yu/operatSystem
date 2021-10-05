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

    printf("I'm the Parent process, my pid = %d \n", getpid());
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
            /* wait for child process terminates */
            waitpid(pid, &status, WUNTRACED);
            printf("Parent process receiving the SIGCHILD signal");

            if (WIFEXITED(status)) {
                printf("normal termination with Exit STATUS = %d \n", WEXITSTATUS(status));
            }
            else if(WIFSIGNALED(status)) {
                printf("CHILD EXECUTION FAILED %d\n", WTERMSIG(status));
            }
            else if(WIFSTOPPED(status)) {
                printf("CHILD PROCESS STOPPED %d\n", WSTOPSIG(status))
            }else{
                printf("CHILD PROCESS CONTINUED\n");
            }
            exit(0);
        }
    }
    return 0;

    /* check child process'  termination status */
}

