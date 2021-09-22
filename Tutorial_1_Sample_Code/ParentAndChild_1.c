#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>


int main(int argc, char *argv[]){
    
    char buf[50] = "Original test strings";    
    pid_t pid;
    
    printf("Process start to fork\n");
    pid=fork(); // fork is system call to create process, assign to pid
    
    if(pid==-1){
        perror("fork");
        exit(1);
    }
    else{
        
        //Child process
        if(pid==0){
            strcpy(buf, "Test strings are updated by child.");
            printf("I'm the Child Process: %s\n", buf);
            exit(0); // 0 reapreants exit without fault
        }
        
        //Parent process
        else{
            sleep(3);
            printf("I'm the Parent Process: %s\n", buf);
            exit(0);
        }
    }
    
    return 0;
}
