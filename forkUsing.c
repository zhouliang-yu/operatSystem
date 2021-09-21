#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
//    int id = fork();
//    printf("hello world from id : %d\n", id);
//    if (id == 0) {
//        printf("hello from child process \n");
//    }else {
//        printf("hello from main process \n");
//    }
//    fork();
//    fork();
//    fork();
//    fork();
    int id = fork();
    if (id != 0) {
        fork();
    }

    printf("hello \n");
    return 0;
}
