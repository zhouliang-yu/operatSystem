#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;

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
 * |                    file name                    |       location    |      size        |create_t|modify_t|
 */
struct My_FCB
{
  char file_name[20];
  u32 location;
  u32 size;
  int create_time;
  int modified_time;
}Current_FCB;

__device__ void segment_management_storage(FileSystem *fs, u32 fp, u32 original_size) {
	u32 position = fs->FILE_BASE_ADDRESS + fp * 32;
	u32 size = ((original_size - 1) / 32 + 1) * 32;
	while ((fs->volume[position + size] != 0 || (position + size) %32 != 0)&& position + original_size < fs->STORAGE_SIZE) {
		fs->volume[position] = fs->volume[position + size];
		fs->volume[position + size] = 0;
		position++;
	}

}

__device__ void segment_management_sup(FileSystem *fs, u32 fp, u32 original_size) {

	//manage the segmentation in superblock
	for (int i = 0; i < file_start_location / 8 + 1; i++) {
		fs->volume[i] = 0;
	}
	file_start_location = file_start_location - (original_size - 1) / 32 - 1;
	u32 file_start_location_q = file_start_location / 8;
	u32 remainder = file_start_location % 8;

	for (int i = 0; i < file_start_location_q && i < fs->SUPERBLOCK_SIZE ; i++) {
		fs->volume[i] = 512 - 1;
	}
	for (int i = 0; i < remainder; i++) {
		fs->volume[file_start_location_q] = fs->volume[file_start_location_q] + (1 << i);
	}


}

__device__ void segment_management_FCB(FileSystem *fs, u32 fp, u32 original_size) {
	//manage the segmentation in FCB
	u32 FCB_file_start_location;
	
	for (int i = 4096; i < 36863; i = i + 32) {
		if (fs->volume[i] == 0 && fs->volume[i + 1] == 0 && fs->volume[i + 2] == 0 && fs->volume[i + 3] == 0) break;
		FCB_file_start_location = (fs->volume[i + 28] << 24) + (fs->volume[i + 29] << 16) + (fs->volume[i + 30] << 8) + (fs->volume[i + 31]);
		if (FCB_file_start_location > fp) {
			FCB_file_start_location = FCB_file_start_location - (original_size - 1) / 32 - 1;
			fs->volume[i + 28] = FCB_file_start_location >> 24;
			fs->volume[i + 29] = FCB_file_start_location >> 16;
			fs->volume[i + 30] = FCB_file_start_location >> 8;
			fs->volume[i + 31] = FCB_file_start_location;
		}
	}


}

__device__ void segment_management(FileSystem *fs, u32 fp, u32 original_size) {

	segment_management_storage(fs, fp, original_size);
	segment_management_sup(fs, fp, original_size);
	segment_management_FCB(fs, fp, original_size);
	
}


__device__ u32 search_FCB(FileSystem *fs, char *s) {
	int flag_find;
  for (int i = fs->SUPERBLOCK_SIZE; i < fs->FILE_BASE_ADDRESS - 1; i += fs->FCB_SIZE)
  {
  	  flag_find = 0;
	  if (fs->volume[i + 24] == 0 && fs->volume[i + 25] == 0 && fs->volume[i + 26] == 0 && fs->volume[i + 27] == 0) // nothing has been stored
	  {						  // cannot find
		  break;
	  }
	
		  
	for(int j = 0; j < 20; j++)
	{
	  if (fs->volume[i + j] != s[j])
	  {

		  flag_find = 1;
		  break;
	  }
	}
	  

	  if (flag_find == 0)
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

__device__ u32 file_info_store(FileSystem *fs, char *s){
	gtime++;
	
	current_FCB_position = FCB_position;
	for (int i = 0; i < 20; i++)
	{ // 0-20 stores the file name
		fs->volume[FCB_position + i] = s[i];
	}

	//store the create time
	fs->volume[FCB_position + 28] = gtime >> 8;
	fs->volume[FCB_position + 29] = gtime & 0x000000FF;

	//store the modified time
	fs->volume[FCB_position + 30] = gtime >> 8;
	fs->volume[FCB_position + 31] = gtime & 0x000000FF;

	//store the start location of block
	fs->volume[FCB_position + 20] = file_start_location >> 24;
	fs->volume[FCB_position + 21] = file_start_location >> 16;
	fs->volume[FCB_position + 22] = file_start_location >> 8;
	fs->volume[FCB_position + 23] = file_start_location;


	//uptime the time
	//gtime++;
	//gtime++;

	//uptime FCB position
	FCB_position = FCB_position + 32;
	
}

__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
	//if not exist
	u32 file_exist = search_FCB(fs, s);
	if (op == G_READ) {
		if (file_exist == -1) {
			printf("error: no such file to read\n");
			return -1;
		}else{
			current_FCB_position = search_FCB(fs, s);
			u32 start_block = (fs->volume[current_FCB_position + 20] << 24) + (fs->volume[current_FCB_position + 21] << 16) + (fs->volume[current_FCB_position + 22] << 8) + (fs->volume[current_FCB_position + 23]);
			return start_block;
		}
    }
	
	if 	(op == G_WRITE){
		if(file_exist == -1){
			file_info_store(fs, s);
			
		}
		else{
			gtime++;
			current_FCB_position = search_FCB(fs, s);
			u32 start_block = (fs->volume[current_FCB_position + 20] << 24) + (fs->volume[current_FCB_position + 21] << 16) + (fs->volume[current_FCB_position + 22] << 8) + (fs->volume[current_FCB_position + 23]);
			u32 size = (fs->volume[current_FCB_position + 24] << 24) + (fs->volume[current_FCB_position + 25] << 16) + (fs->volume[current_FCB_position + 26] << 8) + (fs->volume[current_FCB_position + 27]);
			for (int i = 0; i < size; i++) {
				fs->volume[start_block * 32 + i + fs->FILE_BASE_ADDRESS] = 0;
			}

			//clean the old file in block
			for (int i = 0; i < (size - 1) / 32 + 1; i++) {
				u32 super_file_start_location = start_block + i;
				int shift_number = super_file_start_location % 8;
				fs->volume[super_file_start_location / 8] = fs->volume[super_file_start_location / 8] - (1 << shift_number);
			}

			//uptime FCB time
			fs->volume[current_FCB_position + 30] = gtime >> 8;
			fs->volume[current_FCB_position + 31] = gtime;

			//uptime the time
			
			return start_block;
		}
		
		


	}
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{

	if(fp == -1){
		printf("%d", size);
		printf("error\n");
	}
	
	for (int i = 0; i < size; i++) {
		output[i] = fs->volume[fp * 32 + i + fs->FILE_BASE_ADDRESS];
	}
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */

	if(size > fs->MAX_FILE_NUM) {
		printf("incorrect error\n");
		return -1;
	}

	fp &= 0x0fffffff;
	if (fp == -1){
		printf(" error\n");
	}

    
	int enough_space = (fs->volume[(fp + (size - 1) / 32)/8] >> (fp + (size - 1) / 32) % 8) % 2;
	if (enough_space == 1) {
		u32 original_size = (fs->volume[current_FCB_position + 24] << 24) + (fs->volume[current_FCB_position + 25] << 16) + (fs->volume[current_FCB_position + 26] << 8) + (fs->volume[current_FCB_position + 27]);
		if (file_start_location * 32 - 1 + size >= fs->SUPERBLOCK_SIZE) {
			return -1;
		}

	
		else {
			for (int i = 0; i < size; i++) { 
				fs->volume[file_start_location * 32 + i + fs->FILE_BASE_ADDRESS] = input[i];

				
				if (i % 32 == 0) {
					
					fs->volume[(file_start_location + i / 32) / 8] = fs->volume[(file_start_location + i / 32) / 8] + (1 << ((file_start_location + i / 32) % 8));
				}
			}

			
			fs->volume[current_FCB_position + 24] = size >> 24;
			fs->volume[current_FCB_position + 25] = size >> 16;
			fs->volume[current_FCB_position + 26] = size >> 8;
			fs->volume[current_FCB_position + 27] = size;

			
			fs->volume[current_FCB_position + 20] = file_start_location >> 16;
			fs->volume[current_FCB_position + 21] = file_start_location >> 16;
			fs->volume[current_FCB_position + 22] = file_start_location >> 8;
			fs->volume[current_FCB_position + 23] = file_start_location;
		}
		segment_management(fs, fp, original_size);
	}	
	if(enough_space == 0){
		u32 old_file_size = (fs->volume[current_FCB_position + 24] << 24) + (fs->volume[current_FCB_position + 25] << 16) + (fs->volume[current_FCB_position + 26] << 8) + (fs->volume[current_FCB_position + 27]);
		u32 original_size = old_file_size - size;

		
		for (int i = 0; i < size; i++) {
			fs->volume[fp * 32 + i + fs->FILE_BASE_ADDRESS] = input[i];

		
			if (i % 32 == 0) { 
				fs->volume[(fp + i /32) / 8] = fs->volume[(fp + i / 32) / 8] + (1 << ((fp + i / 32) % 8));
			}
		}
		if (int (original_size) < 0) file_start_location = file_start_location + (-original_size - 1) / 32 + 1;

		
		fs->volume[current_FCB_position + 24] = size >> 24;
		fs->volume[current_FCB_position + 25] = size >> 16;
		fs->volume[current_FCB_position + 26] = size >> 8;
		fs->volume[current_FCB_position + 27] = size;
		if (original_size > 0 && old_file_size != 0 && fp != file_start_location - 1) segment_management(fs, fp + (size - 1) / 32 + 1, original_size);
	}

	
}

__device__ void fs_gsys(FileSystem *fs, int op)
{
	u32 end_point;

	/* Implement LS_D and LS_S operation here */
	for (u32 i = 4096; i - 32 < 36863; i = i + 32) {
		u32 size = (fs->volume[i + 24] << 24) + (fs->volume[i + 25] << 16) + (fs->volume[i + 26] << 8) + (fs->volume[i + 27]);
		if (size == 0) {
			size = (fs->volume[4096 + 24] << 24) + (fs->volume[4096 + 25] << 16) + (fs->volume[4096 + 26] << 8) + (fs->volume[4096 + 27]);
			end_point = i - 32;
			break;
		}
		end_point = i - 32;
	}

	
	if (end_point < 4096) printf("error: no file in FCB \n");

	
	if (op == 0) {
		for (int i = 4096; i < end_point; i = i + 32) {
			for (int j = 4096; j < end_point + 4096 - i; j += 32) {
				u32 j_time_previous =  (fs->volume[j + 30] << 8) + (fs->volume[j + 31]);
				u32 j_time_after = (fs->volume[j + 30 + 32] << 8) + (fs->volume[j + 31+ 32]);
				if (j_time_previous < j_time_after){ 
					for (int k = 0; k < 32; k++) {
						uchar tempt = fs->volume[j + k];
						fs->volume[j + k] = fs->volume[j + k +32];
						fs->volume[j + k +32] = tempt;
					}
				}
			}
		}
	}

	
	else {
		for (int i = 4096; i < end_point; i = i + 32) {
			for (int j = 4096; j < end_point - i + 4096; j = j + 32) {
				u32 j_size_previous = (fs->volume[j + 24] << 24) + (fs->volume[j + 25] << 16) + (fs->volume[j + 26] << 8) + (fs->volume[j + 27]);
				u32 j_size_after = (fs->volume[j + 32 + 24] << 24) + (fs->volume[j + 25 + 32] << 16) + (fs->volume[j + 26 + 32] << 8) + (fs->volume[j + 27 + 32]);
				u32 j_time_previous = (fs->volume[j + 30] << 8) + (fs->volume[j + 31]);
				u32 j_time_after = (fs->volume[j + 30 + 32] << 8) + (fs->volume[j + 31 + 32]);
				if (j_size_previous < j_size_after){
					for (int k = 0; k < 32; k++) {
						uchar tempt = fs->volume[j + k];
						fs->volume[j + k] = fs->volume[j + k +32];
						fs->volume[j + k +32] = tempt;
					}
				}
				if (j_size_after == j_size_previous && j_time_previous > j_time_after){
					for (int k = 0; k < 32; k++) {
						uchar tempt = fs->volume[j + k];
						fs->volume[j + k] = fs->volume[j + k +32];
						fs->volume[j + k +32] = tempt;
					}
				}
			}
		}
	}

	char name[20];
	if (op == 1) {
		u32 size;
		printf("---sort by file size---\n");
		for (u32 i = 4096; i <= end_point; i = i + 32) {
			for (int j = 0; j < 20 ; j++) {
				name[j] = fs->volume[i + j];
			}
			size = (fs->volume[i+ 24] << 24) + (fs->volume[i + 25] << 16) + (fs->volume[i + 26] << 8) + (fs->volume[i + 27]);
			printf("%s %d\n", name, size);
		}
	}
	else {
		printf("---sort by modified time---\n");
		for (u32 i = 4096; i <= end_point; i = i + 32) {
			for (int j = 0; j < 20 ; j++) {
				name[j] = fs->volume[i + j];
			}
			printf("%s\n",name);
		}
	}
}

__device__ u32 cuda_strlen(char* s) {
	u32 result = 0;
	for (u32 i = 0; i < 50; i++)
	{
		if (s[i] == '\0')
		{
			break;
		}
		result++;
	}
	return result;
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
	if (cuda_strlen(s) > 20)
	{
			printf("File Name Size Exceed The Largest Filename Limit \n");
			return;
	}


	if (op == RM){
		u32 file_exist = search_FCB(fs, s);
		if (file_exist == -1){
			printf("error: no such file to remove\n");}
		else {
			current_FCB_position = search_FCB(fs, s);

			// find the start block in FCB
			u32 start_block = (fs->volume[current_FCB_position + 20] << 24) + (fs->volume[current_FCB_position + 21] << 16) + (fs->volume[current_FCB_position + 22] << 8) + (fs->volume[current_FCB_position + 23]);

			//clean the old file in storage
			u32 size = (fs->volume[current_FCB_position+24] << 24) + (fs->volume[current_FCB_position + 25] << 16) + (fs->volume[current_FCB_position + 26] << 8) + (fs->volume[current_FCB_position + 27]);
			for (int i = 0; i < size; i++) {
				fs->volume[start_block * 32 + i + fs->FILE_BASE_ADDRESS] = 0;
			}

			//clean the file in superblock
			for (int i = 0; i < (size - 1) / 32 + 1; i++) {
				fs->volume[start_block + i] = 0;
			}

			//clear content in FCB
			for (int i = 0; i < 32; i++) {
				fs->volume[current_FCB_position + i] = 0;
			}
			segment_management(fs, start_block, size);

			for (int i = current_FCB_position; i < 36863; i = i + 32) {
			if (fs->volume[i + 32 + 24] == 0 && fs->volume[i + 32 + 25] == 0 && fs->volume[i + 32 + 26] == 0 && fs->volume[i + 32 + 27] == 0) break;
			for (int j = 0; j < 32; j++) {
				fs->volume[i + j] = fs->volume[i + j + 32];
				fs->volume[i + j + 32] = 0;
			}
		}
			FCB_position = FCB_position - 32;
		}
	}
}



