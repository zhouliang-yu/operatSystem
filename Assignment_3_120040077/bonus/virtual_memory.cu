#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
  }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ bool page_fault(VirtualMemory *vm, u32 page_num) {
  for (int i = 0; i < vm -> PAGE_ENTRIES; i ++) {
    if (vm -> invert_page_table[i] == page_num) {
      return false;
    }
  }
  *vm->pagefault_num_ptr = *vm->pagefault_num_ptr + 1;
  return true;
}



__device__ int frame_number_page(VirtualMemory *vm, u32 page_num) {
 
 int frame_number;
// if there is a page number equals to page_num, return the frame number(index ) as frame number*/
 for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    if (vm -> invert_page_table[i] == page_num) {
	frame_number = i;
      return i;
    }
  }
  return -1; //out of index
}

__device__ int find_frame_num_frame_table(VirtualMemory *vm, u32 frame_num)
{
//given frame number find the corresponding block  in the frame list
  for (int i = 0; i < vm -> PAGE_ENTRIES; i++){
    if (vm -> invert_page_table[i + vm ->PAGE_ENTRIES] == frame_num){
      return i;
    }
  }
  return -1;
}

__device__ void paging_for_read(VirtualMemory *vm, u32 page_num, u32 page_offset, u32 frame_num){
/** check page fault*/	
if (page_fault(vm, page_num)) {
    frame_num = vm -> invert_page_table[vm->PAGE_ENTRIES];

    /** move to memory*/
    u32 original_page_num = vm->invert_page_table[frame_num];
    for (int i = 0; i < 32; i ++) {
    vm->storage[original_page_num * 32 + i] = vm -> buffer[frame_num * 32 + i];
    vm->buffer[frame_num * 32 + i] = vm -> storage[page_num * 32 + i];
  }
  vm -> invert_page_table[frame_num] = page_num;
  }else {
    frame_num = frame_number_page(vm, page_num);
  }
  
int frame_id = find_frame_num_frame_table(vm, frame_num);
int tempt = vm->invert_page_table[vm->PAGE_ENTRIES + frame_id];
  for (int i = frame_id; i < vm -> PAGE_ENTRIES - 1; i ++) {
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = vm->invert_page_table[i + vm->PAGE_ENTRIES + 1];
  }
  vm -> invert_page_table[2 * vm->PAGE_ENTRIES - 1] = tempt;

}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr)
{
  /* Complate vm_read function to read single element from data buffer */
  u32 page_num = addr / 32;
  u32 page_offset = addr % 32;
  u32 frame_num; 
 
  paging_for_read(vm,  page_num,  page_offset,  frame_num);


}



__device__ void move_to_storage(VirtualMemory *vm, u32 frame_num){
  u32 page_num = vm -> invert_page_table[frame_num];
  for (int i = 0; i < 32; i ++) {
    vm->storage[page_num * 32 + i] = vm->buffer[frame_num * 32 + i];
  }
}

__device__ void paging_for_write(VirtualMemory *vm, u32 page_num, u32 page_offset, u32 frame_num, u32 value){
/** check page fault*/	
  if(page_fault(vm, page_num)) {
    frame_num = vm->invert_page_table[vm->PAGE_ENTRIES];
    
/** check if frame is full*/
if (vm->invert_page_table[frame_num] != 0x80000000) 
  {
    /** if is full move to the storage*/
	move_to_storage(vm, frame_num);
  }


    vm->invert_page_table[frame_num] = page_num;
  }
  else{
    frame_num = frame_number_page(vm, page_num);
  }
  vm->buffer[frame_num * 32 + page_offset] = value;

/** change from invalid to valid*/
int frame_id = find_frame_num_frame_table(vm, frame_num);
  int tempt = vm->invert_page_table[vm->PAGE_ENTRIES + frame_id];
  for (int i = frame_id; i < vm -> PAGE_ENTRIES - 1; i ++) {
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = vm->invert_page_table[i + vm->PAGE_ENTRIES + 1];
  }
  vm -> invert_page_table[2 * vm->PAGE_ENTRIES - 1] = tempt;
}



__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  u32 page_num = addr / 32;
  u32 page_offset = addr % 32;
  u32 frame_num;

paging_for_write(vm,  page_num,  page_offset,  frame_num, value);


}



__device__ void paging_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size, int page_num, int frame_num, int frame_offset){
if (page_fault(vm, page_num)) {
      frame_num = vm -> invert_page_table[vm -> PAGE_ENTRIES];
      u32 original_page_num = vm->invert_page_table[frame_num];
    for (int i = 0; i < 32; i ++) {
    vm->storage[original_page_num * 32 + i] = vm -> buffer[frame_num * 32 + i];
    vm->buffer[frame_num * 32 + i] = vm -> storage[page_num * 32 + i];
  }
  vm -> invert_page_table[frame_num] = page_num;
    }else{
      frame_num = frame_number_page(vm, page_num);
    }
 
/** move to the results buffer*/
 for (int i = 0; i < 32; i++){
    results[page_num * 32 + i] = vm -> buffer[frame_num * 32 + i]; // load element from vm buffer to result buffer in global memory
  }    

/** change from invalid to valid*/
int frame_id = find_frame_num_frame_table(vm, frame_num);
    int tempt = vm->invert_page_table[vm->PAGE_ENTRIES + frame_id];
  for (int i = frame_id; i < vm -> PAGE_ENTRIES - 1; i ++) {
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = vm->invert_page_table[i + vm->PAGE_ENTRIES + 1];
  }
  vm -> invert_page_table[2 * vm->PAGE_ENTRIES - 1] = tempt;


}





__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
  for (int i = 0; i < input_size; i++){
    u32 page_num = i /32;
    u32 frame_offset = i % 32;
    u32 frame_num;

paging_snapshot(vm, results,  offset, input_size,  page_num,  frame_num,  frame_offset);
  
  }
}


