/** zhouliang YU
 * macro
 * sep 9th 
 */

# include <stdio.h>
# include <stdlib.h>

/** preprocess macro*/
# define hundred 100
# define twice(x) 2 * x
# define twice_as_good(x) (2 * (x)) // importance of wrap things in parenthess when defining the macro

int main () {
	printf("hundred: %d\n", hundred);

	int x = twice(5);
	int y = twice(5 + 1);
	int z = twice_as_good(5 + 1);
	printf("x: %d \n y: %d \n z: %d \n", x, y, z);
	exit(0);
}
