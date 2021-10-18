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
#define THREADNUM 11
bool isWin = 0;
bool isLose = 0;
bool isExit = 0;

pthread_mutex_t mutex;
pthread_cond_t cond;
int logs_speed[ROW - 1];
int logs_size[ROW -1];
int left_end[ROW - 1];
int right_end[ROW - 1];

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

	while (!isExit) {
		pthread_mutex_lock(&mutex);

		/* draw the river*/
		for (int i = 0; i < COLUMN - 1; i++) {
			map[*log_id][i] = ' ';
		}

		if (*log_id % 2){
			/* odd row move from left to right*/
			right_end[*log_id] = (right_end[*log_id] + 1) %  (COLUMN - 1);
			left_end[*log_id] = (left_end[*log_id] + 1) %  (COLUMN - 1);
		}
		else {
			/* even row move from right to left*/
			left_end[*log_id] = (left_end[*log_id] - 1) %  (COLUMN - 1);
			right_end[*log_id] = (right_end[*log_id] - 1) %  (COLUMN - 1);
		}

		if (right_end[*log_id] > left_end[*log_id]) { // the log is crossing the right boundary
			for (int j = left_end[*log_id]; j < logs_size[*log_id]; j ++) {
				map[*log_id][j] = '=';
			}
		}else { // the log is crossing the boundary
			for (int j = left_end[*log_id]; j < COLUMN - 1; j ++) {
				map[*log_id][j] = '=';
			}
			for (int k = 0; k < right_end[*log_id]; k++) {
				map[*log_id][k] = '=';
			}
		}

		/* draw the two sides*/
		for (int j = 0; j < COLUMN - 1; j ++) {
			map[ROW][j] = '|';
			map[0][j] = '|';
		}

		/*  Check keyboard hits, to change frog's position or quit the game. */
		if (kbhit()) {
			char dir = getchar();
			if (  dir == 'w' || dir == 'W') {
				frog.x = frog.x + 1;
			}
			if ( dir == 'a' || dir == 'A') {
				frog.y = frog.y - 1;
			}
			if ( (dir == 's' || dir == 'S') && frog.x != ROW) {
				frog.x = frog.x - 1;
			}
			if (dir == 'd' || dir == 'D') {
				frog.y = frog.y + 1;
			}
			if (dir == 'q' || dir == 'Q') {
				isExit = 1;
			}
		}

		map[frog.x][frog.y] = '0';

		/*  Check game's status  */
		if (map[frog.x][frog.y] == ' ') { // the flog falls to the river
			isExit = 1;
		}

		if (frog.y <= 0) { //touches the right boundary
			isExit = 1;
			isLose = 1;
		}

		if (frog.y >= COLUMN - 1) { //touches the left boundary
			isExit = 1;
			isLose = 1;
		}

		if (frog.x == 0) {
			isExit = 1;
			isWin = 1;
		}
		/*  Print the map on the screen  */

		printf("\033c");
		//Print the map into screen
		for( int i = 0; i <= ROW; ++i)	{
			puts( map[i] );
		}
		usleep(1000);
	}
	pthread_mutex_unlock(&mutex);
	usleep(logs_speed[*log_id] * 6000);
	pthread_exit(NULL);
}

void *log_ini() {
	/* initilize the setting of logs*/
	srand((unsigned int)time(NULL));

	for (int i = 1; i < ROW; i ++) {
		logs_speed[i] = rand()%20 + 10;
		logs_size[i] = rand()%5 + 13;
		left_end[i] = rand() % (COLUMN - 1);
		right_end[i] = (left_end[i] + logs_size[i]) % (COLUMN - 1);

		if (right_end[i] > left_end[i]) {
			for (int k = left_end[i]; k < right_end[i]; k++) {
				map[i][k] = '=';
			}
		}else {
			for (int k = left_end[i]; k < COLUMN - 1; k ++) {
				map[i][k] = '=';
			}
			for (int j = 0; j < right_end[i]; j ++) {
				map[i][j] = '=';
			}
		}
	}


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

	/* initialize the logs*/
	log_ini();

	//Print the map into screen
	for( i = 0; i <= ROW; ++i)	{
		puts( map[i] );
	}



	/*  Create pthreads for logs move and frog control.  */
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
