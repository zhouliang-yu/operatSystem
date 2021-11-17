vm: bonus_main.o virtual_memory.o user_program.o
	nvcc -arch=sm_30 bonus_main.o virtual_memory.o user_program.o -rdc=true -o vm

bonus_main.o: bonus_main.cu
	nvcc -arch=sm_30 bonus_main.cu -rdc=true -c

user_program.o: user_program.cu
	nvcc -arch=sm_30 user_program.cu -rdc=true -c

virtual_memory.o: virtual_memory.cu
	nvcc -arch=sm_30 virtual_memory.cu -rdc=true -c

clean:
	rm vm virtual_memory.o user_program.o bonus_main.o
