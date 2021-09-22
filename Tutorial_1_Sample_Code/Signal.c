#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/wait.h>

/** Process signals
 *  1. send a signal to caller
 *          int raise(int sig)
 *  2. evaluate child process's status (zero or non-zero)
 *           int WIFEXITED (int status) return true if the program exited under control
 *           int WIFSIGNALED (int status)return true if the program exited because of signal
 *           int WIFSTOPPED (int status) if the child stopped by a signal
 *
 *  3. evaluate child process's returned value of status argument(exact values)
 *           int WEXITSTATUS(int status) return the exit status (0..255)
 *           int WTERMSIG(int status) return the terminating signal
 *           int WSTOPSIG(int status) return the signal that  stopped the child
 *
 *  4. Signals:
 *          SIGQUIT 3
 *          SIGKILL 9
 *          SIGTERM 15
 *          SIGSTOP 19
 *          SIGCHILD 0 CHECK IF CHILD PROCESS QUITS NORMALLY
 */

int main(int argc, char *argv[]){

    pid_t pid;
    int status;

    printf("Process start to fork\n");
    pid=fork();

    if(pid==-1){
        perror("fork");
        exit(1);
    }
    else{
    
        //Child process
        if(pid==0){
            printf("I'm the Child Process:\n");
            printf("I'm raising SIGCHLD signal!\n\n");
//            raise(SIGKILL);
            raise(SIGCHLD);
        }
    
        //Parent process
        else{
            wait(&status);
            printf("Parent process receives the signal\n");
            
            if(WIFEXITED(status)){
                printf("Normal termination with EXIT STATUS = %d\n",WEXITSTATUS(status));
            }
            else if(WIFSIGNALED(status)){
                printf("CHILD EXECUTION FAILED: %d\n", WTERMSIG(status));
            }
            else if(WIFSTOPPED(status)){
                printf("CHILD PROCESS STOPPED: %d\n", WSTOPSIG(status));
            }
            else{
                printf("CHILD PROCESS CONTINUED\n");
            }
            exit(0);
        }
    }

    return 0;
}




