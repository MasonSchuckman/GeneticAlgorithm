#include "MultibotSimulation.cuh"
#include <curand_kernel.h>
#include <random>

// NOTE: this must be present in every derived simulation!
extern __constant__ SimConfig config_d;

__constant__ float MAX_SPEED2 = 50.0f;
__constant__ float MAX_ACCEL = 10.00f;

#define degrees 90.0f
#define ROTATION_ANGLE degrees * 3.141592654f / 180.0f // 90 degrees

__host__ void MultibotSimulation::getStartingParams(float *startingParams)
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
The two bots control themselves via acceleration. They have 2 output neurons with (likely) linear activation.
This simulation is mainly to verify a single block can handle multiple bots.
Ax refers to A's x coordinate.
Avx is A's x velocity

Gamestate description
0 : iteration
1 : Ax
2 : Ay
3 : Avx
4 : Avy

5 : Bx
6 : By
7 : Bvx
8 : Bvy

11 : A_total_dist
12 : B_total_dist

13 : targetX
14 : targetY

15 : generation number

*/

// Right now both bots start in the same locaiton. Might change in future
__device__ void MultibotSimulation::setupSimulation(const float *startingParams, float *gamestate)
{
    if (threadIdx.x == 0)
    {
        // iter
        gamestate[0] = 0;

        // pos A
        gamestate[1] = startingParams[3];
        gamestate[2] = startingParams[4];

        // Vel A
        gamestate[3] = 0;
        gamestate[4] = 0;

        // pos B
        gamestate[5] = -startingParams[4];
        gamestate[6] = -startingParams[3];

        // Vel B
        gamestate[7] = 0;
        gamestate[8] = 0;

        // Distances
        gamestate[11] = 0;
        gamestate[12] = 0;

        // Target location
        gamestate[13] = startingParams[0];
        gamestate[14] = startingParams[1];

        gamestate[15] = startingParams[5]; //what generation we're on
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

__device__ void MultibotSimulation::setActivations(float *gamestate, float **activs, int iter)
{
    int bot = -1;
    int tid = threadIdx.x;
    const int numBotVars = 4;

    if (tid < numBotVars)
    {
        bot = 0;
    }
    else if (tid < numBotVars * 2)
    {
        bot = 1;
    }
    // rand for adding noise to other bot's information

    float rand = 0;
    float randomMagnitude = 100.0f / logf(gamestate[15] + 2.0f) + 100.0f / logf((float)(iter + 2));
    
    if (bot == 0)
    {
        activs[bot][tid] = gamestate[tid + 1]; //+1 since iter is 0.

        rand = rng(-randomMagnitude, randomMagnitude, tid + iter + blockIdx.x ^ (int)gamestate[15]);        
        activs[bot][tid + numBotVars] = gamestate[tid + numBotVars + 1] + rand;
        if((int)gamestate[15] % 20 == 0)
            activs[bot][tid + numBotVars] = 0;
    }
    else if (bot == 1)
    {
        activs[bot][tid - numBotVars] = gamestate[tid + 1]; //+1 since iter is 0.

        rand = rng(-randomMagnitude, randomMagnitude, tid + iter + blockIdx.x ^ (int)gamestate[15]);
        activs[bot][tid] = gamestate[tid - numBotVars + 1] + rand;
        if((int)gamestate[15] % 20 == 0)
            activs[bot][tid] = 0;
    }

    if (tid < 2)
    {
        bot = tid;
        gamestate[0] = iter;

        // Input the target position
        activs[bot][8] = gamestate[13];
        activs[bot][9] = gamestate[14];
    }
    
    __syncthreads();
}

__device__ void MultibotSimulation::eval(float **actions, float *gamestate)
{
    const int numBotVars = 4;
    int tid = threadIdx.x;
    int velOffset = 3; // + 3 = 1 (iter) + 2 (pos indecies)
    int posOffset = 1;
    int bot = -1;

    if (tid < 2)
        bot = 0;
    else if (tid < 4)
        bot = 1;

    int direction = tid % 2; // which direction (x or y) this thread updates

    // // update velocities
    // if (tid < 4)
    // {
    //     // Allows precise movement in either direction.
    //     float preference = actions[bot][direction];
        
    //     //float accel = preference * MAX_ACCEL;

    //     // Bound the acceleration change
    //     //accel = fminf(MAX_ACCEL, fmaxf(-MAX_ACCEL, accel));

    //     // Update the bot's velocity
    //     //gamestate[bot * numBotVars + direction + velOffset] += accel;
    //     // gamestate[bot * numBotVars + direction + velOffset] = accel;
    // }



    __syncthreads();

    // update the bots' position
    if (tid < 2)
    {
        bot = tid;
        float accelX = actions[bot][0] * MAX_ACCEL;
        float accelY = actions[bot][1] * MAX_ACCEL;

        float accel = hypotf(accelX, accelY);
        if(accel > MAX_ACCEL){
            float f = MAX_ACCEL / accel;
            accelX *= f;
            accelY *= f;
        }
        
        gamestate[bot * numBotVars + velOffset + 0] += accelX;
        gamestate[bot * numBotVars + velOffset + 1] += accelY;
        float Avx = gamestate[bot * numBotVars + velOffset + 0];
        float Avy = gamestate[bot * numBotVars + velOffset + 1];

        // Make sure the speed doesn't go above max speed
        float speed = hypotf(Avx, Avy);
        if (speed > MAX_SPEED2)
        {
            float f = MAX_SPEED2 / speed;
            gamestate[bot * numBotVars + velOffset + 0] *= f;
            gamestate[bot * numBotVars + velOffset + 1] *= f;
        }

        // Update the bot's position
        gamestate[bot * numBotVars + posOffset + 0] += gamestate[bot * numBotVars + velOffset + 0];
        gamestate[bot * numBotVars + posOffset + 1] += gamestate[bot * numBotVars + velOffset + 1];
    }

    __syncthreads();
}

__device__ int MultibotSimulation::checkFinished(float *gamestate)
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


__device__ void MultibotSimulation::setOutput(float *output, float *gamestate, const float * startingParams_d)
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

__host__ int MultibotSimulation::getID()
{
    return 3;
}