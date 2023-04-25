#include "TargetSimulation.cuh"
#include <random>

// NOTE: this must be present in every derived simulation!
extern __constant__ SimConfig config_d;

__constant__ float MAX_SPEED = 25;

// Dead zone around the target that counts as a hit
__constant__ float epsilon = 0.05f;
__constant__ int resetInterval = 40; // Reset gamestate and change target pos every <resetInterval> iters

#define degrees 90.0f
#define ROTATION_ANGLE degrees * 3.141592654f / 180.0f // 90 degrees

__host__ void TargetSimulation::getStartingParams(float * startingParams){
    static int iterationsCompleted = 0;
    //printf("iters completed = %d\n", iterationsCompleted);
    iterationsCompleted++;

    
    // get random target coordinates
    int minPos = -2;
    int maxPos = 2;
    std::random_device rd;                                 // obtain a random seed from hardware
    std::mt19937 eng(rd());                                // seed the generator
    std::uniform_int_distribution<> distr(minPos, maxPos); // define the range
    float targetX = distr(eng);
    float targetY = distr(eng);

    //random starting pos
    float startingX = distr(eng);
    float startingY = distr(eng);

    double r = 5.0 + iterationsCompleted / 10; // radius of circle
    double angle = ((double)rand() / RAND_MAX) * 2 * 3.14159; // generate random angle between 0 and 2*pi
    targetX = r * cos(angle); // compute x coordinate
    targetY = r * sin(angle); // compute y coordinate
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
}


__device__ void TargetSimulation::setupSimulation(const float *startingParams, float *gamestate)
{
    if (threadIdx.x == 0)
    {
        gamestate[0] = 0;
        gamestate[1] = 0;
        gamestate[2] = startingParams[3];
        gamestate[3] = startingParams[4];
        gamestate[4] = startingParams[0];
        gamestate[5] = startingParams[1];

        gamestate[6] = 0; // total dist
    }
    __syncthreads();
}

__device__ void TargetSimulation::setActivations(float *gamestate, float ** activs, int iter)
{
    int tid = threadIdx.x;
    const int numInputs = 6;
    if (tid < numInputs)
    {
        activs[0][tid] = gamestate[tid];
    }
    if (tid == 0)
    {
        gamestate[7] = iter;
    }
}

__device__ void TargetSimulation::eval(float **actions, float *gamestate)
{
    int tid = threadIdx.x;

    // LIST OF GAMESTATE VARIABLES
    //  0: x vel
    //  1: y vel
    //  2: x pos
    //  3: y pos
    //  4: targetX
    //  5: targetY
    //  6: totalDistance (summed over all iters)
    //  7: iteration

    if (tid < 2)
    {

        // Allows precise movement in either direction.
        float preference = (actions[0][tid * 2]) - (actions[0][tid * 2 + 1]); // Decides if the bots wants to go up or down / left or right
        // float preference = actions[0][tid * 2];
        gamestate[tid] = preference * MAX_SPEED;

        // if(gamestate[tid] > MAX_SPEED && blockIdx.x == 0){
        //     printf("ERROR IN EVAL. activation = %f\n", actions[0][tid]);
        // }
    }

    __syncthreads();

    if (tid == 0)
    {
        float xVel = gamestate[0];
        float yVel = gamestate[1];
        float speed = hypotf(xVel, yVel);
        if (speed > MAX_SPEED)
        {
            float f = MAX_SPEED / speed;
            gamestate[0] *= f;
            gamestate[1] *= f;
        }
        gamestate[2] += gamestate[0];
        gamestate[3] += gamestate[1];
    }
    __syncthreads();
}

__device__ int TargetSimulation::checkFinished(float *gamestate)
{
    // Bot has to be going slowly to the target
    // if (hypotf(gamestate[0], gamestate[1]) < epsilon) return 0;

    float dx = gamestate[4] - gamestate[2];
    float dy = gamestate[5] - gamestate[3];
    float dist = hypotf(dx, dy);

    if (threadIdx.x == 0)
    {
        gamestate[6] += dist;
    }

    float m = hypotf(gamestate[4], gamestate[5]);
    // if (dist < epsilon && threadIdx.x == 0 && m > 100 && blockIdx.x < 10) {
    //     printf("%d at target on iter %f\n", blockIdx.x, gamestate[7]);
    // }
    __syncthreads();

    // Check if we need to reset the sim this iteration
    if (((int)gamestate[7] + 1) % resetInterval == 0)
    {
        if (threadIdx.x == 0)
        {
            // // Reset vel and pos
            // for(int i = 0; i < 4; i++)
            //     gamestate[i] = 0;

            //"Rotate" the target position

            float new_x = gamestate[4] * cosf(ROTATION_ANGLE) - gamestate[5] * sinf(ROTATION_ANGLE);
            float new_y = gamestate[4] * sinf(ROTATION_ANGLE) + gamestate[5] * cosf(ROTATION_ANGLE);

            // Update the coordinates
            gamestate[4] = new_x;
            gamestate[5] = new_y;
        }
    }

    __syncthreads();

    // return dist < epsilon;
    return false;
}

__device__ void TargetSimulation::setOutput(float * output, float * gamestate, const float * startingParams_d){
    static int counter = 0;

    if (threadIdx.x == 0)
    {
        if (gamestate[6] != 0)
            output[blockIdx.x] = -gamestate[6];
        else
            output[blockIdx.x] = 0;

        if (blockIdx.x == 0)
        {
            if (counter % 25 == 0)
                printf("Block %d total dist = %f, efficiency = %f\n", blockIdx.x, gamestate[6], (startingParams_d[2] / gamestate[6]));

            counter++;
        }
    }
}

__host__ int TargetSimulation::getID()
{
    return 2;
}