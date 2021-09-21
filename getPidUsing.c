#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>


int main(int argc, char* argv[]) {
    int id = fork();
    if (id == 0) {
        sleep(1);
    }
    printf("current id: %d, parent id: %d\n ",getpid(), getppid());
    return 0;
}