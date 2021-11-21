#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;
__device__ __managed__ u32 gtime_create = 0;
__device__ __managed__ u32 file_start_location = 0;
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
 * |0|1|2|3|4|5|6|7|8|9|10|11|12|13|14|15|16|17|18|19|   20|21   | 22|23 | 24 | 25|26  | 27 |28  |29 | 30|  31|
 * |                    file name                    |       location    |  size       |mode|create_t|modify_t|
 */
struct My_FCB
{
  char file_name[20];
  u32 location;
  u32 size;
  int create_time;
  int modified_time;
}Current_FCB;


__device__ u32 search_FCB(FileSystem *fs, char *s)
{
  int flag_find = 1;//1 indicates that we find the s, 0 means we can't find
  for (int i = fs->SUPERBLOCK_SIZE; i < fs->FILE_BASE_ADDRESS - 1; i += fs->STORAGE_BLOCK_SIZE)
  {
	  if (fs->volume[i + 24] == 0 && fs->volume[i + 25] == 0 && fs->volume[i + 26] == 0) // nothing has been stored
	  {						  // cannot find
		  flag_find = 0;
		  break;
	  }
	  else
	  {
		  
		  for(int j = 0; j < 20; j++)
		  {
			  if (fs->volume[i + j] != s[j])
			  {

				  flag_find = 0;
				  break;
			  }
		  }
	  }

	  if (flag_find == 1)
	  {
		  return i;
	  }
	  else
	  {
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

	//store the start location of block
	fs->volume[FCB_position + 20] = file_start_location >> 24;
	fs->volume[FCB_position + 21] = file_start_location >> 16;
	fs->volume[FCB_position + 22] = file_start_location >> 8;
	fs->volume[FCB_position + 23] = file_start_location;

	/** set the size to be 0*/
	fs->volume[FCB_position + 24] = 0;
	fs->volume[FCB_position + 25] = 0;
	fs->volume[FCB_position + 26] = 0;

	//update the time
	gtime++;
	gtime_create++;

	//update FCB position
	FCB_position = FCB_position + 32;
	return file_start_location;
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
		u32 file_start_location = (fs->volume[current_FCB_position + 20] << 24) + (fs->volume[current_FCB_position + 21] << 16) + (fs->volume[current_FCB_position + 22] << 8) + (fs->volume[current_FCB_position + 23]);
		//set the mode to be read
		fs->volume[current_FCB_position + 26] = G_READ;
		return file_start_location;
    }
  }

  if(op == G_WRITE) {
	  if (file_exist == -1) // if the file doesn't exist create a file in FCB
	  {
		file_info_store(fs, s);
	  }else{
		  current_FCB_position = file_exist;
		  file_start_location = (fs->volume[current_FCB_position + 20] << 24) + (fs->volume[current_FCB_position + 21] << 16) + (fs->volume[current_FCB_position + 22] << 8) + (fs->volume[current_FCB_position + 23]);
		 
		 
		  
		  //get the size
		  u32 size = ((int)fs->volume[current_FCB_position + 24]) +
					 ((int)fs->volume[current_FCB_position + 25]) * (1 << 8) +
					 ((int)fs->volume[current_FCB_position + 26]) * (1 << 16);
		  //clear the old file content in storage
		  for (int i = 0; i < size; i++)
		  {
			  fs->volume[file_start_location * fs->STORAGE_BLOCK_SIZE + i + fs->FILE_BASE_ADDRESS] = 0;
		  }

		  //clear the old file in the superblock because each bit in superblock represent a block in storage
		  for (int i = 0; i < (size - 1) / 32 + 1; i++)
		  {
			  fs->volume[(file_start_location + i) / 8] = fs->volume[(file_start_location + i) / 8] - (1 << (super_location % 8));
		  }

		  //update the modified time
		  fs->volume[FCB_position + 30] = gtime >> 8;
		  fs->volume[FCB_position + 31] = gtime & 0x000000FF;

		  //update gtime
		  gtime++;

          //set the mode to write
		   fs->volume[current_FCB_position + 26] = G_WRITE;
		  return file_start_location; //a pointer at super block
	  }
}



__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
	if(fp == -1){
		print("error\n");
	}
	
	for (int i = 0; i < size; i++)
	{
		output[i] = fs->volume[fp * fs->STORAGE_BLOCK_SIZE + i + fs->FILE_BASE_ADDRESS];
	}
}

__device__ void manage_segmentation(FileSystem * fs, u32 fp, u32 original_size, u32 size)
{
	u32 block_position = fp * 32 + fs->FILE_BASE_ADDRESS;
	u32 new_size = ((original_size - size - 1) / 32 + 1) * 32;
	while ((fs->volume[block_position + new_size] != 0 || (block_position + new_size) % 32 != 0) && block_position + (original_size - size) < fs->STORAGE_SIZE){
		fs->volume[block_position] = fs-> volume[block_position + new_size];
		fs->volume[block_position + new_size] = 0;
		block_position++;
	}

	/** manage the superblock*/
	for (int i =0 ; i < file_start_location /8 + 1; i++) {
		fs->volume[i] = 0;
	}

	file_start_location = file_start_location - ((original_size - size) -1) / 32 - 1;
	u32 file_start_location_q = file_start_location / 8;
	u32 file_start_location_r = block_position % 8;

	for (int i = 0; i < file_start_location_q && i < fs->SUPERBLOCK_SIZE; i ++) {
		fs->volume[i] = 511;
	}
	for (int i = 0; i < file_start_location_r; i++)
	{
		fs->volume[file_start_location_q] = fs->volume[file_start_location_q] + (1 << i);
	}

	//change FCB
	u32 FCB_block_position;
	for (int i = 4096; i < 36863; i = i + 32)
	{
		if (fs->volume[i + 24] == 0 && fs->volume[i + 25] == 0 && fs->volume[i + 26] == 0){
			break;
		}
		FCB_block_position = (fs->volume[i + 20] << 24) + (fs->volume[i + 21] << 16) + (fs->volume[i + 22] << 8) + (fs->volume[i + 23]);
		if (FCB_block_position > fp)
		{
			FCB_block_position = FCB_block_position - ((original_size - size) - 1) / 32 - 1;
			fs->volume[i + 20] = FCB_block_position >> 24;
			fs->volume[i + 21] = FCB_block_position >> 16;
			fs->volume[i + 22] = FCB_block_position >> 8;
			fs->volume[i + 23] = FCB_block_position;
		}
	}
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
	
	//check if the file is in the write mood
	if (((fp & 0xf0000000) >> 30) != G_WRITE)
	{
		printf("no writing allowed \n");
	}

	if(size > fs->MAX_FILE_NUM) {
		print("incorrect error\n")
		return -1;
	}

	fp &= 0x0fffffff;
	if (fp == -1){
		printf(" error\n");
	}

	int enough_space = (fs->volume[(fp + (size - 1) / 32)/8] >> (fp + (size - 1) / 32) % 8) % 2;

	/**get the original file size*/
	// check if there is enough space for it
	u32 original_size = ((int)fs->volume[current_FCB_position + 24]) +
			   ((int)fs->volume[current_FCB_position + 25]) * (1 << 8) +
			   ((int)fs->volume[current_FCB_position + 26]) * (1 << 16);
	
	file_start_location = (fs->volume[current_FCB_position + 20] << 24) + (fs->volume[current_FCB_position + 21] << 16) + (fs->volume[current_FCB_position + 22] << 8) + (fs->volume[current_FCB_position + 23]);

	// clear all the contents in storage
	for (int i = 0; i < original_size; i ++) {
		fs->volume[fp * fs->FCB_SIZE + i + fs->FILE_BASE_ADDRESS] = 0;
	}

	//enough space to write
	if(enough_space == 0) {
		//write in the storage
		for (int i = 0; i < size; i++)
		{
			fs->volume[fp * fs->STORAGE_BLOCK_SIZE + i + fs->FILE_BASE_ADDRESS] = input[i];
		}
		//update the superblock
		for (int i = 0; i < size; i++)
		{
			if (i % 32 == 0) {
				fs->volume[(fp + i / 32) / 8] = fs->volume[(fp + i / 32) / 8] + (1 << (fp + i / 32));
			}
		}

		if (int(original_size - size) < 0)
			file_start_location = file_start_location + (-(original_size - size) - 1) / 32 + 1;

		//update the size in FCB
		fs->volume[current_FCB_position + 24] = size >> 16;
		fs->volume[current_FCB_position + 25] = size >> 8;
		fs->volume[current_FCB_position + 26] = size;

		if (int(original_size - size) > 0 && original_size != 0 && fp != file_start_location - 1)
		{
			manage_segmentation(fs, fp, original_size, size);
		}

	}
	else{ //DONT have enough space
		if (file_start_location * 32 - 1 + size < fs->SUPERBLOCK_SIZE){
			for (int i = 0; i < size; i ++) {
				fs->volume[file_start_location * 32 + i + fs->FILE_BASE_ADDRESS] = input[i];
			//update the superblock
			if(i % 32 == 0){
				fs->volume[(file_start_location + i / 32) / 8] = fs->volume[(file_start_location + i / 32) / 8] + (1 << (file_start_location % 8));
			}
			
			//update the FCB
			fs->volume[current_FCB_position + 24] = size >> 16;
			fs->volume[current_FCB_position + 25] = size >> 8;
			fs->volume[current_FCB_position + 26] = size;

			//update block position
			fs->volume[i + 20] = file_start_location >> 24;
			fs->volume[i + 21] = file_start_location >> 16;
			fs->volume[i + 22] = file_start_location >> 8;
			fs->volume[i + 23] = file_start_location;
			}
		manage_segmentation(fs, fp, original_size, size);
		}
	}
	


}

__device__ u32 endpoint(FileSystem *fs){
	u32 end_point
	for (U32 i = 4096; I < 36863 + 32; i += 32){
		u32 size = u32 size = ((int)fs->volume[i + 24]) +
							  ((int)fs->volume[i + 25]) * (1 << 8) +
							  ((int)fs->volume[i + 26]) * (1 << 16);
		
		if (size == 0) {
			size = (fs->volume[4096] << 24) + (fs->volume[4096 + 1] << 16) + (fs->volume[4096 + 2] << 8);
			end_point = i - 32;
			break;
		}
	}
	end_point = i - 32;
	return end_point;
}

__device__ void bubblesort(FileSystem *fs, u32 begin, u32 end, int op) {
	
	if (op == 1) { //by size
		for (int i = begin; i < end; i = i + 32){
			for (int j = begin; j < end + begin - i ; j = j+ 32){
				u32 j_size_pre = (fs->volume[j+24] << 16) + (fs->volume[j + 25] << 8 )  + (fs->volume[j + 26]);
				u32 j_size_after = (fs->volume[j + 24 + 32] << 16) + (fs->volume[j + 25 + 32] << 8) + (fs->volume[j + 26 + 32]);
				u32 j_time_pre = (fs->volume[j + 30] << 8) + (fs->volume[j + 31]);
				u32 j_time_after = (fs->volume[j + 30 + 32] << 8) + (fs->volume[j + 31 + 32]);
				if (j_size_pre < j_size_after){
					// swap
					for (int k = 0; i < 32; i++)
					{
						uchar tempt = fs->volume[j + k];
						fs->volume[j + k] = fs->volume[j + k + 32];
						fs->volume[j + k + 32] = tempt;
					}
				}
				if (j_size_after == j_size_pre && j_time_pre > j_time_after){
					// swap
					for (int k = 0; i < 32; i++)
					{
						uchar tempt = fs->volume[j + k];
						fs->volume[j + k] = fs->volume[j + k + 32];
						fs->volume[j + k + 32] = tempt;
					}
				}
			}
		}
	}else{ // by time
		for (int i = begin; i < end; i = i + 32)
		{
			for (int j = begin; j < end + begin - i; j = j + 32)
			{
				u32 j_time_prev = (fs->volume[j + 28] << 8) + (fs->volume[j + 29]);
				u32 j_time_after = (fs->volume[j + 28 + 32] << 8) + (fs->volume[j + 29 + 32]);
				if (j_time_prev < j_time_after){
					// swap
					for (int k = 0; i < 32; i++)
					{
						uchar tempt = fs->volume[j + k];
						fs->volume[j + k] = fs->volume[j + k + 32];
						fs->volume[j + k + 32] = tempt;
					}
				}
			}
		}
	}

}

__device__ void display(FileSystem*fs, u32 end_point, int op){
	char file_name[20];
	if (op != 0) { // sort by file size
		for (u32 i = 4096; i <= end_point; i = i + 32)
		{
			for (int j = 0; j < 20; j++)
			{
				name[j] = fs->volume[i + j];
			}
			u32 size = (fs->volume[j + 24] << 16) + (fs->volume[j + 25] << 8) + (fs->volume[j + 26]);
			printf("%s %d\n", file_name, size);
		}
	}
	else{ //sort by time
		for (u32 i = 4096; i <= end_point; i = i + 32)
		{
			for (int j = 0; j < 20; j++)
			{
				name[j] = fs->volume[i + j];
			}
			printf("%s\n", name);
		}
	}



}




__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
	/** sort by date*/
	u32 end_point = end_point(fs);

	if (end_point < 4096)
	{
		print("error: no file in FCB");
		return;
	}
	
	if(op != 0) { //sort by size
		print("---sort by file size---\n");
		bubblesort(fs, 4096, end_point, 1);
		display(fs, end_point, 1);
	}else{
		print("---sort by time---\n")
		bubblesort(fs, 4096, end_point, 0);
		display(fs, end_point, 0);
	}

}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
	u32 file_exist = search_FCB(fs, s);
	if (file_exist == -1)
		printf("error : the file is not exist\n");
	else
	{
		current_FCB_position = file_exist;
		//find where the file start from FCB
		u32 FCB_start_block = (fs->volume[current_FCB_position + 20] << 24) + (fs->volume[current_FCB_position + 21] << 16) + (fs->volume[current_FCB_position + 22] << 8) + (fs->volume[current_FCB_position + 23]);

		//find the size of file
		u32 size = ((int)fs->volume[current_FCB_position + 24]) +
				   ((int)fs->volume[current_FCB_position + 25]) * (1 << 8) +
				   ((int)fs->volume[current_FCB_position + 26]) * (1 << 16);
		
		//clear content in storage
		for (int i = 0; i < size; i++)
		{
			fs->volume[FCB_start_block * 32 + i + fs->FILE_BASE_ADDRESS] = 0;
		}

		//clean corresponding superblock
		for (int i = 0; i < (size - 1) / 32 + 1; i++)
		{
			fs->volume[FCB_start_block + i] = 0;
		}

		//clean the FCB
		for (int i = 0; i < 32; i++)
		{
			fs->volume[current_FCB_position + i] = 0;
		}

		segment_management(fs, FCB_start_block, size);
		
		for (int i = current_FCB_position;i < 36863; i = i + 32){
			if (fs->volume[i + 32 + 24] == 0 && fs->volume[i + 32+25] == 0 && fs->volume[i +32+ 26] == 0){
				for (int j = 0; j < 32; j ++){
					fs->volume[i + j + 32] = 0;
				}
			}
		}


		FCB_position = FCB_position - 32;
	}
}
