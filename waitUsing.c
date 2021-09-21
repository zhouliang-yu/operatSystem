#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>


int main(int argc, char* argv[]) {
    int id = fork();
    int n;
    if (id == 0) {
        n = 1;
    }else {
        n = 6;
    }
    if (id != 0) {
        wait(&id);
    }

    int i;
//    for (i = n; i < n + 5; i ++) {
//        printf("%d", i);// OS decides the order without wait()
//        fflush(stdout); //clear the output buffer and move the buffered data to console (in case of stdout) or disk (in case of the file output stream)
//        printf("\n");
//    }

    for (i = n; i < n + 5; i ++) {
        printf("%d", i);// OS decides the order without wait()
        fflush(stdout); //clear the output buffer and move the buffered data to console (in case of stdout) or disk (in case of the file output stream)

    }


}