#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;
__device__ __managed__ u32 gtime_create = 0;
__device__ __managed__ u32 block_position = 0;
__device__ __managed__ u32 FCB_position = 4096;
__device__ __managed__ u32 current_FCB_position = 4096;

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

}



/*
 * my FCB structure
 * |0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|   20|21   | 22|23 | 24 | 25|26  | 27|28  |29 | 30|  31|
 * |                    file name                    |       location    |  size           |create_t|modify_t|
 */
struct My_FCB
{
  char file_name[20];
  u32 location;
  int file_size;
  int mode;
  int create_time;
  int modified_time;
}Current_FCB;


__device__ u32 search_FCB(FileSystem *fs, char *s)
{
  int flag_find = 1;//1 indicates that we find the s, 0 means we can't find
  for (int i = fs->SUPERBLOCK_SIZE; i < fs->FILE_BASE_ADDRESS - 1; i += 32) {
    if (fs->volume[i] == 0 && fs->volume[i + 1] == 0 && fs->volume[i + 2] == 0 && fs->volume[i+3] == 0)
    { // cannot find
      flag_find = 0
      break;
    }
    else
    {
      for (int j = 0; j < 20; j++) { // 0-20 stores the file name
        if(fs->volume[i+j] != s[j]){
          flag_find = 0;
          break;
        }
      }
    }

    if (flag_find == 1) {
      return i;
    }else{
      continue;
    }
    
  }

  return -1;

}

__device__ void file_info_store(FileSystem *fs, char *s){
	current_FCB_position = FCB_position;
	for (int j = 0; j < 20; j++)
	{ // 0-20 stores the file name
		fs->volume[FCB_position + i] = s[i];
	}

	//store the create time
	fs->volume[FCB_position + 28] = gtime_create >> 8;
	fs->volume[FCB_position + 29] = gtime_create & 0x000000FF;

	//store the modified time
	fs->volume[FCB_position + 30] = gtime >> 8;
	fs->volume[FCB_position + 31] = gtime & 0x000000FF;

	//store the start block
	fs->volume[FCB_position + 20] = block_position >> 24;
	fs->volume[FCB_position + 21] = block_position >> 16;
	fs->volume[FCB_position + 22] = block_position >> 8;
	fs->volume[FCB_position + 23] = block_position;

	//update the time
	gtime++;
	gtime_create++;

	//update FCB position
	FCB_position = FCB_position + 32;
	return block_position;
}


	__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
  
  u32 file_exist = search_FCB(fs, s);
	/* Implement open operation here */
  if (op == G_READ) { // in the read mode
    if (file_exist == -1) {
      print("cannot find file in the read mode");
    }else{ //we find match s
		current_FCB_position = file_exist;
		u32 start_location = (fs->volume[current_FCB_position + 20] << 24) + (fs->volume[current_FCB_position + 21] << 16) + (fs->volume[current_FCB_position + 22] << 8) + (fs->volume[current_FCB_position + 23]);
		// fs->volume[current_FCB_position + 26] = uchar('r');
		return start_location;
    }
  }

  if(op == G_WRITE) {
	  if (file_exist == -1) // if the file doesn't exist create a file in FCB
	  {
		file_info_store(fs, s);
	  }else{
		  current_FCB_position = file_exist;
		  u32 start_location = (fs->volume[current_FCB_position + 20] << 24) + (fs->volume[current_FCB_position + 21] << 16) + (fs->volume[current_FCB_position + 22] << 8) + (fs->volume[current_FCB_position + 23]);
		 
		 
		  //clear the old file content in storage
		  u32 size = (fs->volume[current_FCB_position+24] << 24) + (fs->volume[current_FCB_position + 25] << 16) + (fs->volume[current_FCB_position + 26] << 8) + (fs->volume[current_FCB_position + 27]);
		  for (int i = 0; i < size; i++)
		  {
			  fs->volume[start_location * 32 + i + fs->FILE_BASE_ADDRESS] = 0; // fill with 0
		  }

		  //clear the old file in the superblock
		  for (int i = 0; i < (size - 1) / 32 + 1; i++)
		  {
			  u32 super_block_position = start_location + i;
			  int shift_number = super_block_position % 8;
			  fs->volume[super_block_position / 8] = fs->volume[super_block_position / 8] - (1 << shift_number);
		  }

		  //update the modified time
		  fs->volume[FCB_position + 30] = gtime >> 8;
		  fs->volume[FCB_position + 31] = gtime & 0x000000FF;

		  //update gtime
		  gtime++;

		//   fs->volume[current_FCB_position + 26] = uchar('w');
		  return start_location; //a pointer at super block
	  }
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */




	
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
}
