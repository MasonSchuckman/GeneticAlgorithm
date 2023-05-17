#include "AirHockeySimulation.h"
#include <random>

#define maxSpeed 1
#define maxAccel .5f
#define maxRotSpeed 30

#define degrees 90.0f
#define toRads 3.141592654f / 180.0f
#define ROTATION_ANGLE degrees *toRads
#define friction 0.95f

#define OUT_OF_BOUNDS_DIST = 5
#define actor_state_len 5
#define actor_size 2
#define goal_height 5
#define goal_dist 20
enum actor_state_offset
{
	x_offset,
	y_offset,
	xvel_offset,
	yvel_offset,
	score_offset
};
#define gen_num 15

void AirHockeySimulation::getStartingParams(float *startingParams)
{
	static int iterationsCompleted = 0;
	// printf("iters completed = %d\n", iterationsCompleted);

	// get random target coordinates
	float minPos = -10;
	float maxPos = 10;
	std::random_device rd;										 // obtain a random seed from hardware
	std::mt19937 eng(rd());										 // seed the generator
	std::uniform_real_distribution<float> distr(minPos, maxPos); // define the range

	float targetX = distr(eng) / 3;
	float targetY = distr(eng);

	startingParams[0] = targetX;
	startingParams[1] = targetY;
	startingParams[2] = iterationsCompleted;
	startingParams[3] = (((double)rand() / RAND_MAX) - 0) * 10 + 10;
	startingParams[4] = (((double)rand() / RAND_MAX) - 0.5) * 20;
	startingParams[5] = (((double)rand() / RAND_MAX) - 1) * 10 - 10;
	startingParams[6] = (((double)rand() / RAND_MAX) - 0.5) * 20;
	// printf("Starting params:\n");
	// for(int i = 0; i < 7; i++){
	// 	printf("%f, ", startingParams[i]);
	// }
	// printf("\n\n");
	iterationsCompleted++;
}

/*
Two bots in this simulation. Calling them A and B.
The two bots control themselves via acceleration & direction. They have 2 output neurons with (likely) linear activation.
This simulation is mainly to verify a single block can handle multiple bots.
Ax refers to A's x coordinate.
Avx is A's x velocity

Gamestate description
0-4: BotA data
5-9: BotB data
10-14: Ball data (ball 'score' is Game Tick)
15: generation number
20,21 = TOUCHES
*/

// Bots start at ~ -10 and 10.  More starting positions may be used in the future
void AirHockeySimulation::setupSimulation(int tid, int block, const float *startingParams, float *gamestate)
{

	// Bot A State
	// 5 Units away (to the left)
	gamestate[0 + x_offset] = startingParams[5];
	gamestate[0 + y_offset] = startingParams[6];
	gamestate[0 + xvel_offset] = 0;
	gamestate[0 + yvel_offset] = 0;
	gamestate[0 + score_offset] = 0;

	// Bot B State
	// 5 Units away (up and to the right)
	gamestate[actor_state_len + x_offset] = startingParams[3];
	gamestate[actor_state_len + y_offset] = startingParams[4];
	gamestate[actor_state_len + xvel_offset] = 0;
	gamestate[actor_state_len + yvel_offset] = 0;
	gamestate[actor_state_len + score_offset] = 0;

	// ball state
	gamestate[actor_state_len * 2 + x_offset] = startingParams[0];
	gamestate[actor_state_len * 2 + y_offset] = startingParams[1];
	gamestate[actor_state_len * 2 + xvel_offset] = 0;
	gamestate[actor_state_len * 2 + yvel_offset] = 0;
	gamestate[actor_state_len * 2 + score_offset] = 0; // Iteration number

	gamestate[gen_num] = startingParams[2]; // what generation we're on
	gamestate[16] = 0;
	gamestate[20] = 0;
	gamestate[21] = 0;
}

float AirLimits[5] = {goal_dist, goal_dist, maxSpeed, maxSpeed, 1};

void AirHockeySimulation::setActivations(int tid, int block, float *gamestate, float **activs, int iter)
{

	// Loop over the two bots
	for (int i = 0; i < 2; i++)
	{

		// Enter position info for current bot
		for (int j = 0; j < 2; j++)
		{
			activs[i][j] = gamestate[i * actor_state_len + j] / AirLimits[j];
		}

		// Enter position info for other bot
		for (int j = 0; j < 2; j++)
		{
			//activs[i][j + 2] = gamestate[((i + 1) % 2) * actor_state_len + j] / AirLimits[j];
			activs[i][j + 2] = 0;
		}

		// Enter ball position and velocity
		for (int j = 0; j < 4; j++)
		{
			activs[i][j + 4] = gamestate[2 * actor_state_len + j] / AirLimits[j];
		}

		// Flip x components for bot_b
		if (i == 1)
		{
			for (int j = 0; j < 8; j += 2)
			{
				activs[i][j] *= -1;
			}
		}
	}

	// int bot = -1;

	// // Bot 0
	// // Iterates through bot A, B, and Ball
	// for (int i = 0; i < 3 * actor_state_len; i++)
	// {
	// 	if (i % 5 != 4)
	// 		activs[0][i] = gamestate[i] / AirLimits[i % 5];
	// 	else
	// 		activs[0][i] = 0;
	// }

	// // Bot 1
	// for (int i = 0; i < actor_state_len - 1; i++)
	// {
	// 	// Bot 1 info
	// 	activs[1][i] = gamestate[1 * actor_state_len + i] / AirLimits[i % 5];
	// 	// Bot 0 info
	// 	activs[1][1 * actor_state_len + i] = gamestate[i] / AirLimits[i % 5];

	// 	// This makes it so the bots don't see each other's info.
	// 	activs[1][1 * actor_state_len + i] = 0;
	// 	activs[0][1 * actor_state_len + i] = 0;

	// 	// Ball info
	// 	activs[1][2 * actor_state_len + i] = gamestate[2 * actor_state_len + i] / AirLimits[i % 5];
	// }

	// // Bot 1
	// for (int i = actor_state_len - 1; i < actor_state_len; i++)
	// {
	// 	// Bot 1 info
	// 	activs[1][i] = 0;
	// 	// Bot 0 info
	// 	activs[1][1 * actor_state_len + i] = 0;
	// 	// Ball info
	// 	activs[1][2 * actor_state_len + i] = 0;
	// }

	// // 3 actors (Bot 0 & 1, and Ball)
	// for (int i = 0; i < 3; i++)
	// {
	// 	activs[1][i * actor_state_len + x_offset] *= -1;
	// 	activs[1][i * actor_state_len + xvel_offset] *= -1;
	// }
}

void AirHockeySimulation::eval(int tid, int block, float **actions, float *gamestate)
{
	// 1 point for being closest to the ball
	// 100 points for ball touch
	// 10000 points for goal
	float distToGoalMultiplier = 2;
	int bot = -1;
	// if (tid == 0 && blockIdx.x == 0 && gamestate[14] > 95) {
	// 	printf("Gamestate:\n");
	// 	for (int i = 0; i < 15; i++) {
	// 		printf("%f, ", gamestate[i]);
	// 	}
	// 	printf("\n");
	// }

	// Reduce the ball's velocity due to friction

	gamestate[2 * actor_state_len + xvel_offset] *= friction;
	gamestate[2 * actor_state_len + yvel_offset] *= friction;

	// update the bots' position
	for (tid = 0; tid < 1; tid++)
	{
		bot = tid;

		// Testing letting the bots control velocity directly instead of acceleration
		if(bot == 0){
			gamestate[bot * actor_state_len + xvel_offset] = actions[bot][0] * maxSpeed;
			gamestate[bot * actor_state_len + yvel_offset] = actions[bot][1] * maxSpeed;
		}else{
			gamestate[bot * actor_state_len + xvel_offset] = -actions[bot][0] * maxSpeed;
			gamestate[bot * actor_state_len + yvel_offset] = actions[bot][1] * maxSpeed;
		}

		float speed = hypotf(
			gamestate[bot * actor_state_len + xvel_offset],
			gamestate[bot * actor_state_len + yvel_offset]);
		if (speed > maxSpeed)
		{
			float f = maxSpeed / speed;
			gamestate[bot * actor_state_len + xvel_offset] *= f;
			gamestate[bot * actor_state_len + yvel_offset] *= f;
		}

		// Update the bot's position
		gamestate[bot * actor_state_len + x_offset] += gamestate[bot * actor_state_len + xvel_offset];
		gamestate[bot * actor_state_len + y_offset] += gamestate[bot * actor_state_len + yvel_offset];

		// Check if the player is within bounds, if not, move the player closer to the center
		if(hypotf(gamestate[bot * actor_state_len + x_offset], gamestate[bot * actor_state_len + y_offset]) > goal_dist*2){
			gamestate[bot * actor_state_len + x_offset] /= 2;
			gamestate[bot * actor_state_len + y_offset] /= 2;
		}
	}

	// Incentivize the bots to be close to the ball
	float ballx = gamestate[2 * actor_state_len + x_offset];
	float bally = gamestate[2 * actor_state_len + y_offset];
	// float botDist[2];
	for (int i = 0; i < 2; i++)
	{
		gamestate[i * actor_state_len + score_offset] -= hypotf(
			ballx - gamestate[i * actor_state_len + x_offset],
			bally - gamestate[i * actor_state_len + y_offset]);
	}
	// Bot 0 has a slight disadvantage
	// int closestBot = botDist[1] < botDist[0];
	// gamestate[closestBot * actor_state_len + score_offset] += 1;

	// Reward the bots based on how close the ball is to the goal
	float distToRightGoal = hypotf(ballx - goal_dist, bally - 0);
	float distToLeftGoal = hypotf(ballx - (-goal_dist), bally - 0);

	// Reward left bot based on dist to right goal
	gamestate[0 * actor_state_len + score_offset] -= distToRightGoal * distToGoalMultiplier;

	// Reward right bot based on dist to left goal
	gamestate[1 * actor_state_len + score_offset] -= distToLeftGoal * distToGoalMultiplier;

	// Kick ball
	for (tid = 0; tid < 2; tid++)
	{
		bot = tid;

		ballx = gamestate[2 * actor_state_len + x_offset];
		bally = gamestate[2 * actor_state_len + y_offset];

		if (hypotf(
				ballx - gamestate[bot * actor_state_len + x_offset],
				bally - gamestate[bot * actor_state_len + y_offset]) < actor_size)
		{
			float xDif = ballx - gamestate[bot * actor_state_len + x_offset];
			float yDif = bally - gamestate[bot * actor_state_len + y_offset];
			gamestate[2 * actor_state_len + xvel_offset] = xDif / 4;
			gamestate[2 * actor_state_len + yvel_offset] = yDif / 4;

			//Update ball pos
			gamestate[2 * actor_state_len + x_offset] += gamestate[2 * actor_state_len + xvel_offset];
			gamestate[2 * actor_state_len + y_offset] += gamestate[2 * actor_state_len + yvel_offset];

			gamestate[20 + bot] += 1.0f;
			//printf("HIT!\n");
			// gamestate[2 * actor_state_len + xvel_offset] = gamestate[bot * actor_state_len + xvel_offset] * 1.2;
			// gamestate[2 * actor_state_len + yvel_offset] = gamestate[bot * actor_state_len + yvel_offset] * 1.2;
			// gamestate[bot * actor_state_len + score_offset] += 100;
		}
	}

	ballx = gamestate[2 * actor_state_len + x_offset];
	bally = gamestate[2 * actor_state_len + y_offset];

	// Either bounce or score
	if (abs(ballx) > goal_dist)
	{
		// Goal
		if (abs(bally) < goal_height)
		{
			// Bot 0 wants to score to the right
			int scorer = ballx > 0;
			gamestate[scorer * actor_state_len + score_offset] += 100000;
		}
		else
		{
			gamestate[2 * actor_state_len + xvel_offset] *= -1;
		}

		if (abs(bally) > goal_dist)
		{
			gamestate[2 * actor_state_len + yvel_offset] *= -1;
		}

		ballx += gamestate[2 * actor_state_len + xvel_offset];
		bally += gamestate[2 * actor_state_len + yvel_offset];
		gamestate[2 * actor_state_len + x_offset] = ballx;
		gamestate[2 * actor_state_len + y_offset] = bally;
	}
}

// Game doesn't end on its own
int AirHockeySimulation::checkFinished(int tid, int block, float *gamestate)
{
	gamestate[16]++;
	float ballx = gamestate[2 * actor_state_len + x_offset];
	float bally = gamestate[2 * actor_state_len + y_offset];

	// Scored
	if (abs(ballx) > goal_dist && abs(bally) < goal_height)
	{
		return true;
	}
	return false;
}

void AirHockeySimulation::setOutput(int tid, int block, float *output, float *gamestate, const float *startingParams_d)
{
	// output[block * 2] = (startingParams[2] / gamestate[11]); // Uses efficiency as a metric
	// output[block * 2 + 1] = (startingParams[2] / gamestate[12]); // Uses efficiency as a metric

	output[block * 2] = gamestate[0 * actor_state_len + score_offset];
	output[block * 2 + 1] = gamestate[1 * actor_state_len + score_offset];

	if (block == 0 && (int)startingParams_d[2] % 25 == 0)
		printf("Block %d AScore = %f, BScore = %f, counter = %d, touches = %f, %f\n", block, gamestate[score_offset], gamestate[actor_state_len + score_offset], (int)startingParams_d[2], gamestate[20], gamestate[21]);
}

int AirHockeySimulation::getID()
{
	return 5;
}