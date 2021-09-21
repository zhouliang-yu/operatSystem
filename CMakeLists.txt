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