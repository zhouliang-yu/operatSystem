vm: main.o virtual_memory.o user_program.o
	nvcc -arch=sm_30 main.o virtual_memory.o user_program.o -rdc=true -o vm

main.o: main.cu
	nvcc -arch=sm_30 main.cu -rdc=true -c

user_program.o: user_program.cu
	nvcc -arch=sm_30 user_program.cu -rdc=true -c

virtual_memory.o: virtual_memory.cu
	nvcc -arch=sm_30 virtual_memory.cu -rdc=true -c

clean:
	rm vm virtual_memory.o user_program.o main.o


