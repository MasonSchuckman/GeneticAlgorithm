#include "AirHockeySimulation.cuh"
#include <curand_kernel.h>
#include <random>

// NOTE: this must be present in every derived simulation!
extern __constant__ SimConfig config_d;

extern __device__ unsigned int xorshift(unsigned int x);

// Used for cheap (fast) random numbers in setActivations. Random numbers help the model fit to more general information.
extern __device__ float rng(float a, float b, unsigned int seed);

#define maxSpeed 5
#define maxAccel .5f
#define maxRotSpeed 30

#define degrees 90.0f
#define toRads 3.141592654f / 180.0f
#define ROTATION_ANGLE degrees * toRads
#define friction 0.95f

#define OUT_OF_BOUNDS_DIST = 5
#define actor_state_len 5
#define actor_size 2
#define goal_height 5
#define goal_dist 20
enum actor_state_offset { x_offset, y_offset, xvel_offset, yvel_offset, score_offset };
#define gen_num 15

__host__ void AirHockeySimulation::getStartingParams(float* startingParams)
{
	static int iterationsCompleted = 0;
	//printf("iters completed = %d\n", iterationsCompleted);


	// get random target coordinates
	float minPos = -10;
	float maxPos = 10;
	std::random_device rd;                                 // obtain a random seed from hardware
	std::mt19937 eng(rd());                                // seed the generator
	std::uniform_real_distribution<float> distr(minPos, maxPos); // define the range

	float targetX = 0;
	float targetY = distr(eng);

	

	startingParams[0] = targetX;
	startingParams[1] = targetY;	
	startingParams[2] = iterationsCompleted;
	startingParams[3] =  (((double) rand() / RAND_MAX) - 0) * 20 + 20;
	startingParams[4] =  (((double) rand() / RAND_MAX) - 0.5) * 20;
	startingParams[5] =  (((double) rand() / RAND_MAX) - 1) * 20 - 20;
	startingParams[6] =  (((double) rand() / RAND_MAX) - 0.5) * 20;
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

*/

// Bots start at ~ -10 and 10.  More starting positions may be used in the future
__device__ void AirHockeySimulation::setupSimulation(const float* startingParams, float* gamestate)
{
	if (threadIdx.x == 0)
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

		gamestate[gen_num] = startingParams[2]; //what generation we're on
	}
	__syncthreads();
}


__constant__ float AirLimits[5] = {goal_dist, goal_dist, maxSpeed, maxSpeed, 1};

__device__ void AirHockeySimulation::setActivations(float* gamestate, float** activs, int iter)
{
	int bot = -1;
	int tid = threadIdx.x;

	if (tid == 0) {
		// Bot 0
		// Iterates through bot A, B, and Ball
		for (int i = 0; i < 3 * actor_state_len; i++) {
			if(i % 5 != 4)
				activs[0][i] = gamestate[i] / AirLimits[i % 5];
			else
				activs[0][i] = 0;
		}

		// Bot 1
		for (int i = 0; i < actor_state_len - 1; i++) {
			// Bot 1 info
			activs[1][i] = gamestate[1 * actor_state_len + i] / AirLimits[i % 5];
			// Bot 0 info
			activs[1][1 * actor_state_len + i] = gamestate[i] / AirLimits[i % 5];
			
			
			// This makes it so the bots don't see each other's info.
			activs[1][1 * actor_state_len + i] = 0;
			activs[0][1 * actor_state_len + i] = 0;


			// Ball info
			activs[1][2 * actor_state_len + i] = gamestate[2 * actor_state_len + i] / AirLimits[i % 5];
		}

		// Bot 1
		for (int i = actor_state_len - 1; i < actor_state_len; i++) {
			// Bot 1 info
			activs[1][i] = 0;
			// Bot 0 info
			activs[1][1 * actor_state_len + i] = 0;
			// Ball info
			activs[1][2 * actor_state_len + i] = 0;
		}

		// 3 actors (Bot 0 & 1, and Ball)
		for (int i = 0; i < 3; i++) {
			activs[1][i * actor_state_len + x_offset] *= -1;
			activs[1][i * actor_state_len + xvel_offset] *= -1;
		}
	}

	// Multi-threaded version
	// 
	// if (tid < actor_state_len)
	// {
	// 	bot = 0;
	// }
	// else if (tid < actor_state_len * 2)
	// {
	// 	bot = 1;
	// }
	// int otherBot = !bot;

	// if (bot == 0 || bot == 1)
	// {
	// 	activs[bot][tid - actor_state_len * bot] = gamestate[tid];

    //     if(bot == 0)
	// 	    activs[bot][tid + actor_state_len] = gamestate[tid + actor_state_len];
    //     else
    //         activs[bot][tid - actor_state_len] = gamestate[tid - actor_state_len];
	// }
	// __syncthreads();

	// if (tid < 2)
	// {
	// 	bot = tid;

	// 	// Input the ball data
	// 	int ball_offset = actor_state_len * 2;
	// 	activs[bot][ball_offset + x_offset] = gamestate[ball_offset + x_offset];
	// 	activs[bot][ball_offset + y_offset] = gamestate[ball_offset + y_offset];
	// 	activs[bot][ball_offset + xvel_offset] = gamestate[ball_offset + xvel_offset];
	// 	activs[bot][ball_offset + yvel_offset] = gamestate[ball_offset + yvel_offset];
	// 	// Iteration number
	// 	activs[bot][ball_offset + score_offset] = gamestate[ball_offset + score_offset];
	// }

	
	
	__syncthreads();
}

__device__ void AirHockeySimulation::eval(float** actions, float* gamestate)
{
	// 1 point for being closest to the ball
	// 100 points for ball touch
	// 10000 points for goal

	int tid = threadIdx.x;
	int bot = -1;
	if (tid == 0 && blockIdx.x == 0 && gamestate[14] > 95) {
		printf("Gamestate:\n");
		for (int i = 0; i < 15; i++) {
			printf("%f, ", gamestate[i]);
		}
		printf("\n");
	}


	// Reduce the ball's velocity due to friction
	if(tid == 0){
		gamestate[2 * actor_state_len + xvel_offset] *= friction;
		gamestate[2 * actor_state_len + yvel_offset] *= friction;
	}
	__syncthreads();

	// update the bots' position
	if (tid < 2)
	{
		bot = tid;

		// float xaccel = actions[bot][0] * maxAccel;
		// float yaccel = actions[bot][1] * maxAccel;

		// float accel = hypotf(xaccel, yaccel);
		// if (accel > maxAccel) {
		// 	float f = maxAccel / accel;
		// 	xaccel *= f;
		// 	yaccel *= f;
		// }

		// gamestate[bot * actor_state_len + xvel_offset] += xaccel;
		// gamestate[bot * actor_state_len + yvel_offset] += yaccel;

		// Testing letting the bots control velocity directly instead of acceleration
		gamestate[bot * actor_state_len + xvel_offset] = actions[bot][0] * maxSpeed;
		gamestate[bot * actor_state_len + yvel_offset] = actions[bot][1] * maxSpeed;


		float speed = hypotf(
			gamestate[bot * actor_state_len + xvel_offset], 
			gamestate[bot * actor_state_len + yvel_offset]);
		if (speed > maxSpeed) {
			float f = maxSpeed / speed;
			gamestate[bot * actor_state_len + xvel_offset] *= f;
			gamestate[bot * actor_state_len + yvel_offset] *= f;
		}

		// Update the bot's position
		gamestate[bot * actor_state_len + x_offset] += gamestate[bot * actor_state_len + xvel_offset];
		gamestate[bot * actor_state_len + y_offset] += gamestate[bot * actor_state_len + yvel_offset];
	}

	__syncthreads();

	if (tid == 0) {
		float ballx = gamestate[2 * actor_state_len + x_offset];
		float bally = gamestate[2 * actor_state_len + y_offset];
		float botDist[2];
		for (int i = 0; i < 2; i++) {
			botDist[i] = hypotf(
				ballx - gamestate[i * actor_state_len + x_offset],
				bally - gamestate[i * actor_state_len + y_offset]);
		}
		// Bot 0 has a slight disadvantage
		int closestBot = botDist[1] < botDist[0];
		gamestate[closestBot * actor_state_len + score_offset] += 1;
	}
	__syncthreads();
	// Kick ball
	if (tid < 2) {
		bot = tid;

		float ballx = gamestate[2 * actor_state_len + x_offset];
		float bally = gamestate[2 * actor_state_len + y_offset];

		if (hypotf(
			ballx - gamestate[bot * actor_state_len + x_offset],
			bally - gamestate[bot * actor_state_len + y_offset]
		) < actor_size) {
			gamestate[2 * actor_state_len + xvel_offset] = gamestate[bot * actor_state_len + xvel_offset];
			gamestate[2 * actor_state_len + yvel_offset] = gamestate[bot * actor_state_len + yvel_offset];
			gamestate[bot * actor_state_len + score_offset] += 100;
		}
	}

	__syncthreads();

	if (tid == 0) {
		float ballx = gamestate[2 * actor_state_len + x_offset];
		float bally = gamestate[2 * actor_state_len + y_offset];

		// Either bounce or score
		if (abs(ballx) > goal_dist) {
			// Goal
			if (abs(bally) < goal_height) {
				// Bot 0 wants to score to the right
				int scorer = bally > 0;
				gamestate[scorer * actor_state_len + score_offset] += 10000;
			}
			else
			{
				gamestate[2 * actor_state_len + xvel_offset] *= -1;
			}
		}
		if (abs(bally) > goal_dist) {
			gamestate[2 * actor_state_len + yvel_offset] *= -1;
		}

		ballx += gamestate[2 * actor_state_len + xvel_offset];
		bally += gamestate[2 * actor_state_len + yvel_offset];
		gamestate[2 * actor_state_len + x_offset] = ballx;
		gamestate[2 * actor_state_len + y_offset] = bally;
	}

	__syncthreads();
}

// Game doesn't end on its own
__device__ int AirHockeySimulation::checkFinished(float* gamestate)
{
	__syncthreads();
	float ballx = gamestate[2 * actor_state_len + x_offset];
	float bally = gamestate[2 * actor_state_len + y_offset];

	// Scored
	if (abs(ballx) > goal_dist && abs(bally) < goal_height) {
		return true;
	}
	return false;
}


__device__ void AirHockeySimulation::setOutput(float* output, float* gamestate, const float* startingParams_d)
{
	// output[block * 2] = (startingParams[2] / gamestate[11]); // Uses efficiency as a metric
	// output[block * 2 + 1] = (startingParams[2] / gamestate[12]); // Uses efficiency as a metric


	if (threadIdx.x == 0)
	{
		output[blockIdx.x * 2] = gamestate[0 * actor_state_len + score_offset];
		output[blockIdx.x * 2 + 1] = gamestate[1 * actor_state_len + score_offset];

		if (blockIdx.x == 0)
		{
			if ((int)startingParams_d[2] % 25 == 0)
				printf("Block %d AScore = %f, BScore = %f, counter = %d\n", blockIdx.x, gamestate[score_offset], gamestate[actor_state_len + score_offset], (int)startingParams_d[2]);
			
		}
	}
}

__host__ int AirHockeySimulation::getID()
{
	return 5;
}