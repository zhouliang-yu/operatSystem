fs: main.o user_program.o file_system.o
	nvcc -arch=sm_30 main.o file_system.o user_program.o -rdc=true -o fs


main.o: main.cu
	nvcc -arch=sm_30 main.cu -rdc=true -c


user_program.o: user_program.cu
	nvcc -arch=sm_30 user_program.cu -rdc=true -c


file_system.o: file_system.cu
	nvcc -arch=sm_30 file_system.cu -rdc=true -c

clean: 
	rm fs file_system.o user_program.o main.o
