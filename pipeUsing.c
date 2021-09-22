#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <fcntl.h>

// fifo is sth you can read or write between processes
int main (int argc, char * argv[]) {
    if (mkfifo("myfifo", 0777) == -1) {
        if (errno != EEXIST) {
            printf("could not create fifo file \n");
            return 1;
        }
    }

    open("myfifo", O_WRONLY); // return file descripter

    return 0;
}