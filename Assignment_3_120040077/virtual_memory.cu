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

__device__ bool check_page_fault(VirtualMemory *vm, u32 addr, u32 page_num) {
  for (int i = 0; i < vm -> PAGE_ENTRIES; i ++) {
    if (vm -> invert_page_table[i] == page_num) {
      return true;
    }
  }
  *vm->pagefault_num_ptr = *vm->pagefault_num_ptr + 1;
  return false;
}

__device__ void move_to_memory(VirtualMemory *vm, u32 frame_num, u32 page_num) {
  u32 original_page_num = vm->invert_page_table[frame_num];
  for (int i = 0; i < 32; i ++) {
    vm->storage[original_page_num * 32 + i] = vm -> buffer[frame_num * 32 + i];
    vm->buffer[frame_num * 32 + i] = vm -> storage[page_num * 32 + i];
  }
  vm -> invert_page_table[frame_num] = page_num;
}

__device__ int find_frame_number(VirtualMemory *vm, u32 page_num) {
  for (int i = 0; i < vm -> PAGE_ENGTRIES; i++) {
    if (vm -> invert_page_table[i] == page_num) {
      return i;
    }
  }
  printf("fault: get out of the index range");
  return -1;
}

__device__ int find_frame_num_in_frame_table(VirtualMemory *vm, u32 frame_num)
{
  for (int i = 0; i < vm -> PAGE_ENTRIES; i++){
    if (vm -> invert_page_table[i + vm ->PAGE_ENTRIES] == frame_num){
      return i;
    }
  }
  printf("fault: get out of the index range");
  return -1;
}

__device__ void change_frame_table_valid_to_invalid(VirtualMemory *vm, u32 frame_num) {
  int tempt = vm->invert_page_table[vm->PAGE_ENTRIES + find_frame_num_in_frame_table(vm, frame_num);
  for (int i = find_frame_num_in_frame_table(vm, frame_num); i < vm -> PAGE_ENTRIES - 1; i ++) {
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

  if (!check_page_fault(vm, page_num)) {
    frame_num = vm -> invert_page_table[vm->PAGE_ENTRIES];
    move_to_memory(vm, frame_num, page_num);
  }else {
    frame_num = find_frame_number(vm, page_num);
  }
  change_frame_table_valid_to_invalid(vm, frame_num);
}






__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
  u32 page_num = addr / 32;
  u32 page_offset = addr % 32;
  u32 frame_num;
  if(!check_page_fault(vm, page_num)) {
    frame_num = vm->invert_page_table[vm->Page_ENTRIES];
    check_frame_full(vm,page_num,frame_num);
    vm->invert_page_table[frame_num] = page_num;
  }
  else{
    frame_num = find_frame_number(vm, page_num);
  }
  vm->buffer[frame_num * 32 + page_offset] = value;
  change_frame_table_valid_to_invalid(vm, frame_num);
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
}


