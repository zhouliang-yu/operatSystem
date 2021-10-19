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
int logs_speed[ROW + 10];
int logs_size[ROW + 10];
int left_end[ROW + 10];
int right_end[ROW + 10];

struct Node{
	int x , y;
	Node( int _x , int _y ) : x( _x ) , y( _y ) {};
	Node(){} ;
} frog, speed;


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
	long log_id;
	log_id = (long) t;

	while (isExit == 0 && isWin == 0 && isLose == 0) {




		if (log_id % 2){
			/* odd row move from left to right*/
			left_end[log_id] = (left_end[log_id] + 1) % (COLUMN - 1);
		}
		else {
			/* even row move from right to left*/
			left_end[log_id] = left_end[log_id] - 1;
			if (left_end[log_id] < 0) left_end[log_id] += COLUMN - 1;
		}

		pthread_mutex_lock(&mutex);

		/* draw the river*/
		for (int j = 0; j < COLUMN - 1; ++j) {
			map[log_id][j] = ' ';
		}

		for (int i = 0, j = left_end[log_id]; i < logs_size[log_id]; ++j, ++i) {
			map[log_id][j%(COLUMN - 1)] = '=';
		}

		 /* draw the two sides*/
		 for (int j = 0; j < COLUMN - 1; j ++) {
		 	map[ROW][j] = map[0][j] = '|' ;
		 }

        for( int j = 0; j < COLUMN - 1; ++j )
		    map[ROW + 1][j] = '-';



		/*  Check keyboard hits, to change frog's position or quit the game. */
		 if (kbhit()) {
		 	char dir = getchar();
		 	if (  dir == 'w' || dir == 'W') {
		 		frog.x = frog.x - 1;
		 	}
		 	if ( (dir == 'a' || dir == 'A') && frog.y != 0) {
		 		frog.y = frog.y - 1;
		 	}
		 	if ( (dir == 's' || dir == 'S') && frog.x != ROW) {
		 		frog.x = frog.x + 1;
		 	}
		 	if ((dir == 'd' || dir == 'D') && frog.y!= COLUMN - 2) {
		 		frog.y = frog.y + 1;
		 	}
		 	if (dir == 'q' || dir == 'Q') {
		 		isExit = 1;
		 	}

            /* switch the speed*/
            if ((dir == 'j' || dir == 'J') && speed.y != 0) {
                speed.y = speed.y - 1;
            }
            if ((dir == 'k' || dir == 'K') && speed.y != COLUMN - 2) {
                speed.y = speed.y + 1;
            }
		 }

		 /*  Check game's status  */
		 if (map[frog.x][frog.y] == ' ') { // the flog falls to the river
			isLose = 1;
		 }

		 if (frog.y <= 0) { //touches the left boundary
		 	isLose = 1;
		 }

		 if (frog.y >= COLUMN - 1) { //touches the right boundary
		 	isLose = 1;
		 }

		 if (frog.x == 0) {
		 	isWin = 1;
		 }

		 if (isExit == 0 && isWin == 0 && isLose == 0) {
		 /*  Print the map on the screen  */
			if (frog.x == log_id && map[frog.x][frog.y] == '=') {
				if (frog.x % 2) frog.y ++;
				else frog.y --;
			}

			//  printf("\033[0;0h\033[2J");
			printf("\033[H\033[2J");
			 //Print the map into screen
			 usleep(1000);
			 map[frog.x][frog.y] = '0';
             map[speed.x][speed.y] = '>';
			 for( int i = 0; i <= ROW + 1; ++i)	{
			 	puts( map[i] );
			 }
		 }

		pthread_mutex_unlock(&mutex);
		usleep(logs_speed[log_id] * 6000 * (COLUMN - 1 - speed.y)/ (COLUMN - 1));
	}

	pthread_exit(NULL);
}





int main( int argc, char *argv[] ){

	pthread_t threads[THREADNUM];
	pthread_mutex_init(&mutex, NULL);
	srand((unsigned int)time(NULL));

	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	printf("\e[?25l");

	int i , j ;
	for( i = 1; i < ROW; ++i ){
		for( j = 0; j < COLUMN - 1; ++j )
			map[i][j] = ' ' ;
		logs_speed[i] = rand()%20 + 10;
		logs_size[i] = rand()%5 + 13;
		left_end[i] = rand() % (COLUMN - 1);
	}


	/* initilize the setting of logs*/


	for( j = 0; j < COLUMN - 1; ++j )
		map[ROW][j] = map[0][j] = '|' ;


    /* set the speed changing slide bar*/
	for( j = 0; j < COLUMN - 1; ++j )
		map[ROW + 1][j] = '-';

    speed = Node(ROW + 1, 1);
    map[speed.x][speed.y] = '>';

	frog = Node( ROW, (COLUMN-1) / 2 ) ;
	map[frog.x][frog.y] = '0' ;

	/* set the speed changing slide bar*/



	//Print the map into screen
	for( i = 0; i <= ROW + 1; ++i)	{
		puts( map[i] );
	}



	/*  Create pthreads for logs move and frog control.  */
	for (int i = 1; i < THREADNUM; ++ i) {
		int rc;
		rc = pthread_create(&threads[i], NULL, logs_move, (void*)i);
		if (rc) {
			printf("ERROR: return code from pthread_create() is %d",rc);
			exit(1);
		}
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
	pthread_exit(NULL);

	return 0;

}
