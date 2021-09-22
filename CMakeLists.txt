# cmake_minimum_required(VERSION <specify CMake version here>)
project(operatSystem C)

set(CMAKE_C_STANDARD 11)

add_executable(operatSystem
        a.c
        ctu.c
        lab0.c
        printBytes.c)

add_executable(execve Tutorial_1_Sample_Code/execve.c)
add_executable(forkUsing forkUsing.c)
add_executable(waitUsing waitUsing.c)
add_executable(getPidUsing getPidUsing.c)
add_executable(Multiplefork Multiplefork.c)
add_executable(pipeUsing pipeUsing.c)
add_executable(ParentAndChild_2 Tutorial_1_Sample_Code/ParentAndChild_2.c)
add_executable(ProcessTermination_1 Tutorial_1_Sample_Code/ProcessTermination_1.c)
add_executable(Signal Tutorial_1_Sample_Code/Signal.c)