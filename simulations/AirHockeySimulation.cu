#include "AirHockeySimulation.cuh"
#include <curand_kernel.h>
#include <random>

// NOTE: this must be present in every derived simulation!
extern __constant__ SimConfig config_d;

extern __device__ unsigned int xorshift(unsigned int x);

// Used for cheap (fast) random numbers in setActivations. Random numbers help the model fit to more general information.
extern __device__ float rng(float a, float b, unsigned int seed);

#define maxSpeed 1
#define maxAccel .5f
#define maxRotSpeed 30

#define degrees 90.0f
#define toRads 3.141592654f / 180.0f
#define ROTATION_ANGLE degrees * toRads

#define OUT_OF_BOUNDS_DIST = 5
#define actor_state_len 5
#define actor_size .5f
#define goal_height 5
#define goal_dist 20
enum actor_state_offset { x_offset, y_offset, vel_offset, dir_offset, score_offset };
#define gen_num 15

__host__ void AirHockeySimulation::getStartingParams(float* startingParams)
{
	static int iterationsCompleted = 0;
	//printf("iters completed = %d\n", iterationsCompleted);


	// get random target coordinates
	int minPos = -2;
	int maxPos = 2;
	std::random_device rd;                                 // obtain a random seed from hardware
	std::mt19937 eng(rd());                                // seed the generator
	std::uniform_int_distribution<> distr(minPos, maxPos); // define the range
	float targetX = distr(eng);
	float targetY = distr(eng);

	// random starting pos
	float startingX = distr(eng);
	float startingY = distr(eng);

	double r = 5.0 + iterationsCompleted / 10;                // radius of circle
	double angle = ((double)rand() / RAND_MAX) * 2 * 3.14159; // generate random angle between 0 and 2*pi
	targetX = r * cosf(angle);                                 // compute x coordinate
	targetY = r * sinf(angle);                                 // compute y coordinate
	// targetX = 10;
	// targetY = 0;
	startingX = 0;
	startingY = 0;
	if (targetX == 0 && targetY == 0)
		targetX = 2;

	float optimal = hypotf(targetX, targetY) / 2.0 * hypotf(targetX, targetY);

	// transfer target coordinates to GPU

	startingParams[0] = targetX;
	startingParams[1] = targetY;
	startingParams[2] = optimal;
	startingParams[3] = startingX;
	startingParams[4] = startingY;
	startingParams[5] = iterationsCompleted;

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
		gamestate[0 + x_offset] = -5;
		gamestate[0 + y_offset] = 0;
		gamestate[0 + vel_offset] = 0;
		gamestate[0 + dir_offset] = 0;
		gamestate[0 + score_offset] = 0;


		// Bot B State
		// 5 Units away (up and to the right)
		gamestate[actor_state_len + x_offset] = 4;
		gamestate[actor_state_len + y_offset] = 3;
		gamestate[actor_state_len + vel_offset] = 0;
		gamestate[actor_state_len + dir_offset] = 0;
		gamestate[actor_state_len + score_offset] = 0;

		// ball state
		gamestate[actor_state_len * 2 + x_offset] = 0;
		gamestate[actor_state_len * 2 + y_offset] = 0;
		gamestate[actor_state_len * 2 + vel_offset] = 0;
		gamestate[actor_state_len * 2 + dir_offset] = 0;
		gamestate[actor_state_len * 2 + score_offset] = 0; // Iteration number

		gamestate[gen_num] = 0; //what generation we're on
	}
	__syncthreads();
}




__device__ void AirHockeySimulation::setActivations(float* gamestate, float** activs, int iter)
{
	int bot = -1;
	int tid = threadIdx.x;

	if (tid < actor_state_len)
	{
		bot = 0;
	}
	else if (tid < actor_state_len * 2)
	{
		bot = 1;
	}
	int otherBot = !bot;

	if (bot == 0 || bot == 1)
	{
		activs[bot][tid - actor_state_len * bot] = gamestate[tid];

        if(bot == 0)
		    activs[bot][tid + actor_state_len] = gamestate[tid + actor_state_len];
        else
            activs[bot][tid - actor_state_len] = gamestate[tid - actor_state_len];
	}
	__syncthreads();

	if (tid < 2)
	{
		bot = tid;

		// Input the ball data
		int ball_offset = actor_state_len * 2;
		activs[bot][ball_offset + x_offset] = gamestate[ball_offset + x_offset];
		activs[bot][ball_offset + y_offset] = gamestate[ball_offset + y_offset];
		activs[bot][ball_offset + vel_offset] = gamestate[ball_offset + vel_offset];
		activs[bot][ball_offset + dir_offset] = gamestate[ball_offset + dir_offset];
		// Iteration number
		activs[bot][ball_offset + score_offset] = gamestate[ball_offset + score_offset];
	}

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

	// update the bots' position
	if (tid < 2)
	{
		bot = tid;

		float accel = actions[bot][0] * maxAccel;
		// clamped
		accel = fminf(maxAccel, fmaxf(-maxAccel, accel));
		// Not actually omega, omega is rotational acceleration but we're just using rotational velocity
		float omega = actions[bot][1] * maxRotSpeed;
		// clamped
		omega = fminf(maxRotSpeed, fmaxf(-maxRotSpeed, omega));

		gamestate[bot * actor_state_len + vel_offset] += accel;
		float dir = gamestate[bot * actor_state_len + dir_offset] + omega;
		if (dir < 0) dir += 360;
		if (dir > 360) dir -= 360;
		gamestate[bot * actor_state_len + dir_offset] = dir;

		float speed = gamestate[bot * actor_state_len + vel_offset];
		gamestate[bot * actor_state_len + vel_offset] =
			fminf(maxSpeed, fmaxf(-maxSpeed, speed));
		gamestate[bot * actor_state_len + vel_offset] = speed;

		float dx = speed * cosf(dir * toRads);
		float dy = speed * sinf(dir * toRads);

		// Update the bot's position
		gamestate[bot * actor_state_len + x_offset] += dx;
		gamestate[bot * actor_state_len + y_offset] += dy;
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
			gamestate[2 * actor_state_len + dir_offset] = gamestate[bot * actor_state_len + dir_offset];
			gamestate[2 * actor_state_len + vel_offset] = gamestate[bot * actor_state_len + vel_offset] + .1f;
			gamestate[bot * actor_state_len + score_offset] += 100;
		}
	}

	__syncthreads();

	if (tid == 0) {
		float ballx = gamestate[2 * actor_state_len + x_offset];
		float bally = gamestate[2 * actor_state_len + y_offset];
		float ballSpeed = gamestate[2 * actor_state_len + vel_offset];
		float ballDir = gamestate[2 * actor_state_len + dir_offset];

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
				ballDir = 180 - ballDir;
				if (ballDir < 0) ballDir += 180;
				gamestate[2 * actor_state_len + dir_offset] = ballDir;
			}
		}
		if (abs(bally) > goal_dist) {
			ballDir = 360 - ballDir;
			gamestate[2 * actor_state_len + dir_offset] = ballDir;
		}


		float dx = ballSpeed * cosf(ballDir * toRads);
		float dy = ballSpeed * sinf(ballDir * toRads);

		ballx += dx;
		bally += dy;
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
	float ballDir = gamestate[2 * actor_state_len + dir_offset];

	// Scored
	if (abs(ballx) > goal_dist && abs(bally) < goal_height) {
		return true;
	}
	return false;
}


__device__ void AirHockeySimulation::setOutput(float* output, float* gamestate, const float* startingParams_d)
{
	static int counter = 0;
	// output[block * 2] = (startingParams[2] / gamestate[11]); // Uses efficiency as a metric
	// output[block * 2 + 1] = (startingParams[2] / gamestate[12]); // Uses efficiency as a metric


	if (threadIdx.x == 0)
	{
		output[blockIdx.x * 2] = gamestate[0 * actor_state_len + score_offset];
		output[blockIdx.x * 2 + 1] = gamestate[1 * actor_state_len + score_offset];

		if (blockIdx.x == 0)
		{
			if (counter % 25 == 0)
				printf("Block %d AScore = %f, BScore = %f, counter = %d\n", blockIdx.x, gamestate[score_offset], gamestate[actor_state_len + score_offset], counter);

			counter++;
		}
	}
}

__host__ int AirHockeySimulation::getID()
{
	return 5;
}