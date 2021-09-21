#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>


int main(int argc, char* argv[]) {
    int id = fork();
    if (id == 0) {
        sleep(1);
    }
    printf("current id: %d, parent id: %d\n ",getpid(), getppid());
    int res = wait(NULL);
    if (res == -1) {
        printf("no children to wait for \n");
    }else{
        printf("%d finished execution\n", res);
    }
    return 0;
}