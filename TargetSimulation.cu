#include "TargetSimulation.cuh"

// NOTE: this must be present in every derived simulation!
extern __constant__ SimConfig config_d;

__constant__ float MAX_SPEED = 25;

// Dead zone around the target that counts as a hit
__constant__ float epsilon = 0.05f;
__constant__ int resetInterval = 40; // Reset gamestate and change target pos every <resetInterval> iters

#define degrees 90.0f
#define ROTATION_ANGLE degrees * 3.141592654f / 180.0f // 90 degrees

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

__device__ void TargetSimulation::setActivations(float *gamestate, float *activs, int iter)
{
    int tid = threadIdx.x;
    const int numInputs = 6;
    if (tid < numInputs)
    {
        activs[tid] = gamestate[tid];
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

__host__ int TargetSimulation::getID()
{
    return 2;
}