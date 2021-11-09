#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <cuda.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>

//page size is 32 bytes
#define PAGESIZE (32)
//32 KB in the shared memory
#define PHYSICAL_MEM_SIZE (32768)
//128 KB of secondary storage
#define STORAGE_SIZE (131072)
//memory segment size for each thread
#define MEMORY_SEGMENT (32768)

//number of pages in shared memory
#define PHYSICAL_PAGE_NUM (PHYSICAL_MEM_SIZE/PAGESIZE)
//number of pages in global memory
#define STORAGE_PAGE_NUM (STORAGE_MEM_SIZE/PAGESIZE)

#define DATAFILE "./data.bin"
#define OUTFILE "./snapshot.bin"
typedef unsigned char uchar;
typedef uint32_t u32;

/*----------------------------------The macro below are for page table--------------------------------*/

#define INVALID_VALUE (0xffffffff)

#define THREAD_PID_0					(0)
#define THREAD_PID_1					(1)
#define THREAD_PID_2					(2)
#define THREAD_PID_3					(3)

#define PID_BIT_START					(0)
#define PID_BIT_LEN 					(2)
#define VIRPAGE_BIT_START				(PID_BIT_START+PID_BIT_LEN)
#define VIRPAGE_BIT_LEN					(12)
#define COUNTER_BIT_START				(VIRPAGE_BIT_START+VIRPAGE_BIT_LEN)
#define COUNTER_BIT_LEN					(18)

#define FULL_MASK						(0xFFFFFFFF)
#define PID_MASK						(0x3)
#define VIRPAGE_MASK					(0x3FFC)
#define COUNTER_MASK					(0xFFFFC000)

#define GET_PID(x)						((x&PID_MASK)>>PID_BIT_START)
#define GET_VIRPAGE(x)					((x&VIRPAGE_MASK)>>VIRPAGE_BIT_START)
#define GET_COUNTER(x)					((x&COUNTER_MASK)>>COUNTER_BIT_START)

#define CLEAR_PID(x)					(x&(~PID_MASK))
#define CLEAR_VIRPAGE(x)				(x&(~VIRPAGE_MASK))
#define CLEAR_COUNTER(x)				(x&(~COUNTER_MASK))

#define SET_PID(src,value)				(CLEAR_PID(src)|(value<<PID_BIT_START))
#define SET_VIRPAGE(src,value)			(CLEAR_VIRPAGE(src)|(value<<VIRPAGE_BIT_START))
#define SET_COUNTER(src,value)			(CLEAR_COUNTER(src)|(value<<COUNTER_BIT_START))

/*----------------------------------The macro below are for storage table-------------------------------------*/

#define FIRST_PID_START					(0)
#define FIRST_PID_LEN					(2)
#define FIRST_VIRPAGE_START				(FIRST_PID_START+FIRST_PID_LEN)
#define FIRST_VIRPAGE_LEN				(12)
#define SECOND_PID_START				(FIRST_VIRPAGE_START+FIRST_VIRPAGE_LEN)
#define SECOND_PID_LEN					(2)
#define SECOND_VIRPAGE_START			(SECOND_PID_START+SECOND_PID_LEN)
#define SECOND_VIRPAGE_LEN				(12)

#define FIRST_PID_MASK					(0x3)
#define FIRST_VIRPAGE_MASK				(0x3FFC)
#define SECOND_PID_MASK					(0xC000)
#define SECOND_VIRPAGE_MASK				(0xFFF0000)

#define GET_FIRST_PID(x)				((x&FIRST_PID_MASK)>>FIRST_PID_START)
#define GET_FIRST_VIRPAGE(x)			((x&FIRST_VIRPAGE_MASK)>>FIRST_VIRPAGE_START)
#define GET_SECOND_PID(x)				((x&SECOND_PID_MASK)>>SECOND_PID_START)
#define GET_SECOND_VIRPAGE(x)			((x&SECOND_VIRPAGE_MASK)>>SECOND_VIRPAGE_START)

#define CLEAR_FIRST_PID(x)				(x&(~FIRST_PID_MASK))
#define CLEAR_FIRST_VIRPAGE(x)			(x&(~FIRST_VIRPAGE_MASK))
#define CLEAR_SECOND_PID(x)				(x&(~SECOND_PID_MASK))
#define CLEAR_SECOND_VIRPAGE(x)			(x&(~SECOND_VIRPAGE_MASK))

#define SET_FIRST_PID(src,value)		(CLEAR_FIRST_PID(src)|(value<<FIRST_PID_START))
#define SET_FIRST_VIRPAGE(src,value)	(CLEAR_FIRST_VIRPAGE(src)|(value<<FIRST_VIRPAGE_START))
#define SET_SECOND_PID(src,value)		(CLEAR_SECOND_PID(src)|(value<<SECOND_PID_START))
#define SET_SECOND_VIRPAGE(src,value)	(CLEAR_SECOND_VIRPAGE(src)|(value<<SECOND_VIRPAGE_START))

#define GET_STORAGE_PID(num)	 		((num%2==0)?GET_FIRST_PID(STORAGE_TABLE[num/2]):GET_SECOND_PID(STORAGE_TABLE[num/2]))
#define GET_STORAGE_VIRPAGE(num)	 	((num%2==0)?GET_FIRST_VIRPAGE(STORAGE_TABLE[num/2]):GET_SECOND_VIRPAGE(STORAGE_TABLE[num/2]))
#define SET_STORAGE_PID(num,value)		STORAGE_TABLE[num/2]=((num%2==0)?SET_FIRST_PID(STORAGE_TABLE[num/2],value):SET_SECOND_PID(STORAGE_TABLE[num/2],value))
#define SET_STORAGE_VIRPAGE(num,value)	STORAGE_TABLE[num/2]=((num%2==0)?SET_FIRST_VIRPAGE(STORAGE_TABLE[num/2],value):SET_SECOND_VIRPAGE(STORAGE_TABLE[num/2],value))

#define __LOCK(); for(int p=0;p<4;p++){if(threadIdx.x==p){
#define __UNLOCK(); }__syncthreads();}

//#define __GET_BASE() ((threadIdx.x)*MEMORY_SEGMENT)
#define __GET_BASE() (p*MEMORY_SEGMENT)

//Storage table

__device__ __managed__ u32 *STORAGE_TABLE;
__device__ __managed__ u32 STORAGE_COUNT;

//Page-fault times
__device__ __managed__ u32 PAGEFAULT = 0;

//secondary memory
__device__ __managed__ uchar storage[STORAGE_SIZE];

//data input and output
__device__ __managed__ uchar results[STORAGE_SIZE];
__device__ __managed__ uchar input[STORAGE_SIZE];

//shared memory for page table and memory-occupied table
extern __shared__ u32 pt[];


__device__ u32 paging(uchar *buffer, u32 page_num, u32 offset)
{
	u32 free_page_num=INVALID_VALUE;
	u32 lru_time=1;
	u32 lru_page_num=INVALID_VALUE;
	u32 valid_page_num=INVALID_VALUE;
	for(int i=0;i<PHYSICAL_PAGE_NUM;i++){
		if((GET_PID(pt[i])==threadIdx.x) && (GET_VIRPAGE(pt[i])==page_num) && (GET_COUNTER(pt[i])>0)){
			pt[i]=SET_COUNTER(pt[i],1);
			valid_page_num=i;
		}
		else if(GET_COUNTER(pt[i])>0)
			pt[i]=SET_COUNTER(pt[i],GET_COUNTER(pt[i])+1);

		//get free page
		if(GET_COUNTER(pt[i])==0)
			free_page_num=i;
		//get LRU page
		if(GET_COUNTER(pt[i])>lru_time){
			lru_time=GET_COUNTER(pt[i]);
			lru_page_num=i;
		}
	}
	if(valid_page_num!=INVALID_VALUE){
		return valid_page_num*PAGESIZE+offset;
	}
	else if(free_page_num!=INVALID_VALUE){
		pt[free_page_num]=SET_PID(pt[free_page_num],threadIdx.x);
		pt[free_page_num]=SET_VIRPAGE(pt[free_page_num],page_num);
		pt[free_page_num]=SET_COUNTER(pt[free_page_num],1);
/*		for(int i=0;i<PHYSICAL_PAGE_NUM;i++){
			if((GET_COUNTER(pt[i])>0)||(i==free_page_num))
				pt[i]=SET_COUNTER(pt[i],GET_COUNTER(pt[i])+1);
		}*/
		PAGEFAULT++;
		return free_page_num*PAGESIZE+offset;
	}
	else{
		u32 swap_out_start=INVALID_VALUE;
		u32 swap_in_start=INVALID_VALUE;
		u32 phy_start=lru_page_num*PAGESIZE;
		for(int i=0;i<STORAGE_COUNT;i++){
			if((GET_STORAGE_PID(i)==GET_PID(pt[lru_page_num])) && (GET_STORAGE_VIRPAGE(i)==GET_VIRPAGE(pt[lru_page_num])))
				swap_out_start=i*PAGESIZE;

		
			if((GET_STORAGE_PID(i)==threadIdx.x) && (GET_STORAGE_VIRPAGE(i)==page_num))
				swap_in_start=i*PAGESIZE;
		}
		//if the storage does not contain the page to be swapped out
		if(swap_out_start==INVALID_VALUE){
			swap_out_start=STORAGE_COUNT*PAGESIZE;
			SET_STORAGE_PID(STORAGE_COUNT,GET_PID(pt[lru_page_num]));
			SET_STORAGE_VIRPAGE(STORAGE_COUNT,GET_VIRPAGE(pt[lru_page_num]));
			STORAGE_COUNT++;
		}
		//if the page to be swapped in is not in the storage
		if(swap_in_start==INVALID_VALUE){
			for(int i=0;i<PAGESIZE;i++){
				storage[swap_out_start+i]=buffer[phy_start+i];
				buffer[phy_start+i]=0;
			}
		}
		//if the page to be swapped in us in the storage
		else{
			for(int i=0;i<PAGESIZE;i++){
				storage[swap_out_start+i]=buffer[phy_start+i];
				buffer[phy_start+i]=storage[swap_in_start+i];
			}
		}
		pt[lru_page_num]=SET_PID(pt[lru_page_num],threadIdx.x);
		pt[lru_page_num]=SET_VIRPAGE(pt[lru_page_num],page_num);
		pt[lru_page_num]=SET_COUNTER(pt[lru_page_num],1);
/*		for(int i=0;i<PHYSICAL_PAGE_NUM;i++){
			if((GET_COUNTER(pt[i])>0)||(i==lru_page_num))
				pt[i]=SET_COUNTER(pt[i],GET_COUNTER(pt[i])+1);
		}*/
		PAGEFAULT++;
		return phy_start+offset; 
	}

}

__device__ uchar Gread(uchar *buffer,u32 addr)
{
	u32 page_num  = addr/PAGESIZE;
	u32 offset 	  = addr%PAGESIZE;

	//addr means the addr in shared memory
	addr = paging(buffer, page_num, offset);
	return buffer[addr];
}
__device__ void Gwrite(uchar *buffer, u32 addr, uchar value)
{
	u32 page_num  = addr/PAGESIZE;
	u32 offset    = addr%PAGESIZE;

	//addr means the addr in shared memory
	addr = paging(buffer, page_num, offset);
	buffer[addr] = value;
}

__device__ void snapshot(uchar *results, uchar *buffer, int offset, int input_size)
{
	for(int i=0;i<input_size;i++)
		results[i] = Gread(buffer, i+offset);
}

__device__ void init_pageTable(int pt_entries)
{
	PAGEFAULT=0;
	STORAGE_TABLE = pt+PHYSICAL_PAGE_NUM;
	STORAGE_COUNT=0;
	for(int i=0;i<PHYSICAL_PAGE_NUM+2048;i++){
		pt[i]=0;
	}
	for(int i=0;i<STORAGE_SIZE;i++){
		results[i]=0;
		storage[i]=0;
	}
}

int load_binaryFile(const char *filename, uchar *input, int size)
{
	int fd=0;
	int sizeread=0;
	int sizehasread=0;
	fd=open(filename,O_RDONLY);
	if(fd==-1){
		perror("open data.bin error");
		return -1;
	}
	while((sizeread=read(fd,input,size))!=-1){
		sizehasread+=sizeread;
		size-=sizeread;
		input+=sizeread;
		if(sizeread==0){
			close(fd);
			return sizehasread;
		}
	}
	close(fd);
	perror("read data.bin error");
	return -1;
}

int write_binaryFile(const char *filename, uchar *output, int size)
{
	int fd=0;
	int sizewritten=0;
	int sizehaswritten=0;
	fd=open(filename,O_WRONLY|O_CREAT,S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH);
	if(fd==-1){
		perror("open snapshot.bin error");
		return -1;
	}
	while((sizewritten=write(fd,output,size))!=-1){
		sizehaswritten+=sizewritten;
		size-=sizewritten;
		output+=sizewritten;
		if(sizewritten==0){
			close(fd);
			return sizehaswritten;
		}
	}
	close(fd);
	perror("write snapshot.bin error");
	return -1;
}

__global__ void mykernel(int input_size)
{
	//Regard shared memory as physical memory
	__shared__ uchar data[PHYSICAL_MEM_SIZE];

	//get page table entries
	int pt_entries=PHYSICAL_MEM_SIZE/PAGESIZE;
	//We should initialize the page table
	if(threadIdx.x==0)
		init_pageTable(pt_entries);

	
	//####GWrite/Gread code section start####
	__LOCK();
	for(int i=0;i<input_size;i++)
		Gwrite(data,i+__GET_BASE(),input[i+__GET_BASE()]);
	__UNLOCK();


	for(int i=input_size-1;i>=input_size-10;i--){
	__LOCK();
		int value = Gread(data,i+__GET_BASE());
	__UNLOCK();
	}

	__LOCK();
	snapshot(results+__GET_BASE(),data,__GET_BASE(),input_size);
	__UNLOCK();
	//####GWrite/Gread code section end####
	printf("this thread pid = %d, total pagefault times=%u\n",threadIdx.x,PAGEFAULT);
	return;
}

int main()
{
	clock_t t;
	t=clock();

	//Load data.bin into input buffer
	int input_size = load_binaryFile(DATAFILE, input, STORAGE_SIZE);

	printf("The read size is %d\n", input_size);

	//main procedure
	cudaSetDevice(4);
	mykernel<<<1,4,16384>>>(input_size/4);
	cudaDeviceSynchronize();
	cudaDeviceReset();

	//write binary file from results buffer
	write_binaryFile(OUTFILE, results, input_size);

	t=clock()-t;
	printf("total elapsed time = %f\n",((float)t)/CLOCKS_PER_SEC);

	return 0;
}