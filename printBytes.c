/* Zhouliang Yu
 * 10 Septermber 2021
 *
 * This program displays the bytes (one at a time) from an int 
 */

# include <stdio.h>
# include <stdlib.h>

int my_new_func(int a) {
	printf("hi\n");
	return 0;
}

int main (int argc, char *argv[]) {
	int num = 4058;
	my_new_func (num);
	printf("here are the bytes of %d in hex : %08x\n, num, num");

	int i;
	unsigned int mask = 0xFF; // One byte
	for (i = 0; i < sizeof(int); i++) {
		int printme = (num >> 8*i)&mask;
		printf("\t bytes%d = %02x\n", i, printme);
	}

	return 0;
}
