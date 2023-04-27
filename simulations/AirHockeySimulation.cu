#include "AirHockeySimulation.cuh"
#include <curand_kernel.h>
#include <random>

// NOTE: this must be present in every derived simulation!
extern __constant__ SimConfig config_d;

__constant__ float MAX_SPEED2 = 50.0f;
__constant__ float MAX_ACCEL = 10.00f;
__constant__ float MAX_ROT_SPEED = 30f;

#define degrees 90.0f
#define ROTATION_ANGLE degrees * 3.141592654f / 180.0f // 90 degrees

#define actor_state_len 5
enum actor_state_offset {x_offset, y_offset, vel_offset, dir_offset, score_offset}
#define gen_num 15

__host__ void AirHockeySimulation::getStartingParams(float *startingParams)
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
    targetX = r * cos(angle);                                 // compute x coordinate
    targetY = r * sin(angle);                                 // compute y coordinate
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
__device__ void AirHockeySimulation::setupSimulation(const float *startingParams, float *gamestate)
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


__device__ unsigned int xorshift(unsigned int x)
{
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    return x * 0x2545F491;
}

// Used for cheap (fast) random numbers in setActivations. Random numbers help the model fit to more general information.
__device__ float rng(float a, float b, unsigned int seed)
{    
    unsigned int r = xorshift(seed);   
    static const float m = 4294967296.0f; // 2^32
    return a + (b - a) * (static_cast<float>(r) / m);
}

__device__ void AirHockeySimulation::setActivations(float *gamestate, float **activs, int iter)
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
    // rand for adding noise to other bot's information

    float rand = 0;
    float randomMagnitude = 100.0f / logf(gamestate[15] + 2.0f) + 100.0f / logf((float)(iter + 2));
    
    if (bot == 0 || bot == 1)
    {
        activs[bot][tid - actor_state_len * bot] = gamestate[tid];

        rand = rng(-randomMagnitude, randomMagnitude, tid + iter + blockIdx.x ^ (int)gamestate[15]);        
        activs[bot][tid + actor_state_len] = gamestate[tid + actor_state_len * otherBot] + rand;
    }

    if (tid < 2)
    {
        bot = tid;

        // Input the ball data
        int ball_offset = actor_state_len * 2;
        actives[bot][ball_offset + x_offset] = gamestate[ball_offset + x_offset];
        actives[bot][ball_offset + y_offset] = gamestate[ball_offset + y_offset] = 0;
        actives[bot][ball_offset + vel_offset] = gamestate[ball_offset + vel_offset] = 0;
        actives[bot][ball_offset + dir_offset] = gamestate[ball_offset + dir_offset] = 0;
        // Iteration number
        actives[bot][ball_offset + score_offset] = gamestate[ball_offset + score_offset] = iter;
    }
    
    __syncthreads();
}

__device__ void AirHockeySimulation::eval(float **actions, float *gamestate)
{
    int tid = threadIdx.x;
    int bot = -1;

    if (tid < 2) bot = tid;

    // Update direction & velocity
    if(bot != -1) {
    }

    __syncthreads();

    // update the bots' position
    if (tid < 2)
    {
        bot = tid;

        float accel = actions[bot][0] * MAX_ACCELL;
        // clamped
        accel = fminf(MAX_ACCEL, fmaxf(-MAX_ACCEL, accel)); 
        // Not actually omega, omega is rotational acceleration but we're just using rotational velocity
        float omega = actions[bot][1] * MAX_ROTAION_SPEED;
        // clamped
        omega = fminf(MAX_ROTAION_SPEED, fmaxf(-MAX_ROTAION_SPEED, omega));

        gamestate[bot * actor_state_len + vel_offset] += accel;
        float dir = gamestate[bot * actor_state_len + dir_offset] + omega;
        gamestate[bot * actor_state_len + dir_offset] = dir;
        
        speed = gamestate[bot * actor_state_len + vel_offset];
        gamestate[bot * actor_state_len + vel_offset] = 
            fminf(MAX_ROTAION_SPEED, fmaxf(-MAX_ROTAION_SPEED, speed));
        gamestate[bot * actor_state_len + vel_offset] = speed;
        
        float dx = speed * cos(radians(dir));
        float dy = speed * sin(radians(dir));

        // Update the bot's position
        gamestate[bot * actor_state_len + x_offset] += dx;
        gamestate[bot * actor_state_len + y_offset] += dy;
    }

    __syncthreads();
}

__device__ int AirHockeySimulation::checkFinished(float *gamestate)
{
    int tid = threadIdx.x;
    if (tid < 2)
    {
        int bot = tid;
        int posOffset = 1;
        const int numBotVars = 4;

        float dx = gamestate[13] - gamestate[bot * numBotVars + posOffset + 0];
        float dy = gamestate[14] - gamestate[bot * numBotVars + posOffset + 1];
        float dist = hypotf(dx, dy);

        gamestate[bot + 11] += dist; // 11 = distOffset

        if (dist < 0.5f && threadIdx.x == 0)
        {
            //   printf("%d at target on iter %f\n", blockIdx.x, gamestate[0]);
        }

        // if (dist < .5f && threadIdx.x == 0 && gamestate[0] > 30)
        // {
        //     printf("dist = %f, iter = %d, bot = %d\n", dist, (int)gamestate[0], blockIdx.x);
        //     printf("Game state:\n");
        //     for(int i = 0; i < 15; i++)
        //         printf("%f, ", gamestate[i]);
        //     printf("\n");
        // }
    }
    __syncthreads();

    int resetInterval = 2500;
    // Check if we need to reset the sim this iteration
    if (((int)gamestate[0] + 1) % resetInterval == 0)
    {
        if (threadIdx.x == 0)
        {
            

            //"Rotate" the target position

            float new_x = gamestate[13] * cosf(ROTATION_ANGLE) - gamestate[14] * sinf(ROTATION_ANGLE);
            float new_y = gamestate[13] * sinf(ROTATION_ANGLE) + gamestate[14] * cosf(ROTATION_ANGLE);

            // Update the coordinates
            gamestate[13] = new_x;
            gamestate[14] = new_y;
        }
    }

    __syncthreads();

    // return dist < epsilon;
    return false;
}


__device__ void AirHockeySimulation::setOutput(float *output, float *gamestate, const float * startingParams_d)
{   
    static int counter = 0;
    // output[block * 2] = (startingParams[2] / gamestate[11]); // Uses efficiency as a metric
    // output[block * 2 + 1] = (startingParams[2] / gamestate[12]); // Uses efficiency as a metric
                        

    if (threadIdx.x == 0)
    {
        if (gamestate[11] != 0)
            output[blockIdx.x * 2] = -gamestate[11];
        else
            output[blockIdx.x * 2] = 0;

        if (gamestate[12] != 0)
            output[blockIdx.x * 2 + 1] = -gamestate[12];
        else
            output[blockIdx.x * 2 + 1] = 0;

        if (blockIdx.x == 0)
        {
            if (counter % 25 == 0)
                printf("Block %d total dist = %f, efficiency = %f, counter = %d\n", blockIdx.x, gamestate[11], (startingParams_d[2] / gamestate[11]), counter);

            counter++;
        }
    }
}

__host__ int AirHockeySimulation::getID()
{
    return 4;
}