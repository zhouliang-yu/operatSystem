#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50 
#define THREADNUM 10
bool isWin = 0;
bool isLose = 0;
bool isExit = 0;

pthread_mutex_t mutex;
pthread_cond_t cond;
int logs_speed[ROW - 1];
int logs_size[ROW + 1];

struct Node{
	int x , y;
	Node( int _x , int _y ) : x( _x ) , y( _y ) {};
	Node(){} ;
} frog ;


char map[ROW+10][COLUMN] ;

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0.
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}



void *logs_move( void *t ){
	/* initilize the setting of logs*/
	int *log_id = (int*) t;

	logs_speed[*log_id] = rand()%20 + 10;
	logs_size[*log_id] = rand() % 3 + 13;

	while (!isExit) {
		pthread_mutex_lock(&mutex);

		if (*log_id % 2){
			/* odd row move from left to right*/

		}else {

		}
	}


	/*  Check keyboard hits, to change frog's position or quit the game. */
	if (kbhit()) {

	}

	/*  Check game's status  */


	/*  Print the map on the screen  */

	// }
}

int main( int argc, char *argv[] ){

	pthread_t threads[THREADNUM];
	pthread_mutex_init(&mutex, NULL);
	pthread_cond_init(&cond, NULL);

	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	printf("\e[?25l");
	int i , j ;
	for( i = 1; i < ROW; ++i ){
		for( j = 0; j < COLUMN - 1; ++j )
			map[i][j] = ' ' ;
	}

	for( j = 0; j < COLUMN - 1; ++j )
		map[ROW][j] = map[0][j] = '|' ;

	for( j = 0; j < COLUMN - 1; ++j )
		map[0][j] = map[0][j] = '|' ;

	frog = Node( ROW, (COLUMN-1) / 2 ) ;
	map[frog.x][frog.y] = '0' ;

	//Print the map into screen
	for( i = 0; i <= ROW; ++i)
		puts( map[i] );


	/*  Create pthreads for wood move and frog control.  */
	for (int i = 1; i < THREADNUM; ++ i) {
		pthread_create(&threads[i], NULL, logs_move, (void*) i);
		usleep(200);
	}

	for (int i = 1; i < THREADNUM; ++i) {
		pthread_join(threads[i], NULL);
	}
	printf("\033[0;0H\033[2J\033[?25h");
	usleep(1000) ;




	/*  Display the output for user: win, lose or quit.  */
	if (isExit){
		printf("\033[H\033[2J");
		puts("You exit the game");
	}else if (isWin) {
		printf("\033[H\033[2J");
		puts("You win the game \n");
	}else if (isLose) {
		printf("\033[H\033[2J");
		puts("You lose the game \n");
	}


	pthread_mutex_destroy(&mutex);
	pthread_cond_destroy(&cond);
	pthread_exit(NULL);

	return 0;

}
