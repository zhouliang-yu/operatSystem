/** The create_shared_memory is cited from
 * website : http://www.thetopsites.net/article/54638460.shtml
*/
void* create_shared_memory(size_t size) {
  // Our memory buffer will be readable and writable:
  int protection = PROT_READ | PROT_WRITE;

  // The buffer will be shared (meaning other processes can access it), but
  // anonymous (meaning third-party processes cannot obtain an address for it),
  // so only this process and its children will be able to use it:
  int visibility = MAP_SHARED | MAP_ANONYMOUS;

  // The remaining parameters to `mmap()` are not important for this use case,
  // but the manpage for `mmap` explains their purpose.
  return mmap(NULL, size, protection, visibility, -1, 0);
}

void *convert2 (void* v2s) {
    char *result;
    result = (char *) v2s;
    return (void*)result; //cast to void*
}



void signal_manipulation(int status, char* pro_mesg, void* shared_memory) {
    int signal;
    if (WIFEXITED(status)) { //normally exited
        sprintf(pro_mesg, "normally with exit code %d\n", 0);
        // memcpy(convert2(shared_memory), pro_mesg, sizeof(pro_mesg));
    }else if(WIFSIGNALED(status)) { //abnormal exit
        signal = WTERMSIG(status);
        if (signal == 1) {
            sprintf(pro_mesg, "by signal %d (SIGHUP) \n", signal);
        }
        else if(signal == 2) {
            sprintf(pro_mesg, "by signal %d (SIGINT) \n", signal);
        }else if(signal == 3) {
           sprintf(pro_mesg, "by signal %d (SIGQUIT) \n", signal);
        }else if(signal == 4) {
            sprintf(pro_mesg, "by signal %d (SIGILL) \n", signal);
        }else if(signal == 5) {
            sprintf(pro_mesg, "by signal %d (SIFTRAP) \n", signal);
        }else if(signal == 6) {
            sprintf(pro_mesg, "by signal %d (SIGABRT) \n", signal);
        }else if(signal == 7) {
            sprintf(pro_mesg, "by signal %d (SIGBUS) \n", signal);
        }else if(signal == 8) {
            sprintf(pro_mesg, "by signal %d (SIGFPE) \n", signal);
        }else if(signal == 9) {
            sprintf(pro_mesg, "by signal %d (SIGKILL) \n", signal);
        }else if(signal == 11) {
            sprintf(pro_mesg, "by signal %d (SIGSEGV) \n", signal);
        }else if(signal == 13) {
            sprintf(pro_mesg, "by signal %d (SIGPIPE) \n", signal);
        }else if(signal == 14) {
            sprintf(pro_mesg, "by signal %d (SIGALARM) \n", signal);
        }else if(signal == 15) {
            sprintf(pro_mesg, "by signal %d (SIGTERM) \n", signal);
        }
    }else if (WIFSTOPPED(signal)){
        sprintf(pro_mesg, "by signal %d (SIGSTOP) \n", signal);
    }
    memcpy(convert2(shared_memory), pro_mesg, sizeof(pro_mesg));
}

void my_execute(pid_t pid, char **pro_tree_var, char* pro_mesg, char* ext, void* shared_memory, int status, int argc,char *argv[]) {

   while((*pro_tree_var) != NULL) {

       if ((pid = fork()) == 0) { // child process
           sprintf(pro_mesg, "->%d", getpid());
           memcpy(convert2(shared_memory), pro_mesg, sizeof(pro_mesg));
           fflush(stdout);
           strcpy(ext, *(pro_tree_var++)); //ext is string that can be excuted in child processes
       }else{
           break;
       }
   }
   if (pid > 0) {
       pid = waitpid(pid, &status, WUNTRACED);
       sprintf(pro_mesg, "Child process %d of Parent process %d is terminated", pid, getpid());
       memcpy(convert2(shared_memory), pro_mesg, sizeof(pro_mesg));
       signal_manipulation(status, pro_mesg, shared_memory);
    }
    if (sizeof(ext) > 0) {

        if (pro_tree_var - argv == argc) {
            sprintf(pro_mesg, "\n");
            memcpy(convert2(shared_memory), pro_mesg, sizeof(pro_mesg));
        }
        char * par[2];
        par[0] = ext;
        par[1] = NULL;
        execve(par[0], par, NULL);
    }else {
        sprintf(pro_mesg, "Myfork process (%d) terminated normally \n ", getpid());
        memcpy(convert2(shared_memory), pro_mesg, sizeof(pro_mesg));
        printf("\n%s", (char*)shared_memory);
    }


}


int main(int argc,char *argv[]){
	/* Implement the functions here */
    int signal;
	pid_t pid = getpid();
    int status;
    char ext[20] = "";
    char **pro_tree_var = argv + 1;

    char pro_mesg[128];
    void* shared_memory = create_shared_memory(2000);

    sprintf(pro_mesg, "---\n Processt tree: %d", pid);
    memcpy(shared_memory, pro_mesg, sizeof(pro_mesg));

    my_execute(pid, pro_tree_var, pro_mesg, ext, shared_memory, status, argc, argv);

}