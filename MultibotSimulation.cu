#include "MultibotSimulation.cuh"

// NOTE: this must be present in every derived simulation!
extern __constant__ SimConfig config_d;

__constant__ float MAX_SPEED2 = 5.0f;
__constant__ float MAX_ACCEL = 0.30f;

#define degrees 90.0f
#define ROTATION_ANGLE degrees * 3.141592654f / 180.0f // 90 degrees

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
3 : Bx
4 : By
5 : Avx
6 : Avy
7 : Bvx
8 : Bvy

11 : B_total_dist
12 : A_total_dist

13 : targetX
14 : targetY

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

        // pos B
        gamestate[3] = startingParams[3];
        gamestate[4] = startingParams[4];

        // Vel A
        gamestate[5] = 0;
        gamestate[6] = 0;

        // Vel B
        gamestate[7] = 0;
        gamestate[8] = 0;

        // Distances
        gamestate[11] = 0;
        gamestate[12] = 0;

        // Target location
        gamestate[13] = startingParams[0];
        gamestate[14] = startingParams[1];
    }
    __syncthreads();
}

__device__ void MultibotSimulation::setActivations(float *gamestate, float *activs, int iter)
{
    int tid = threadIdx.x;
    const int numInputs = 8; // not including target pos
    if (tid < numInputs)
    {
        activs[tid] = gamestate[tid + 1]; //+1 since iter is 0.
    }
    if (tid == 0)
    {
        gamestate[0] = iter;

        // Input the target position
        activs[8] = gamestate[13];
        activs[9] = gamestate[14];
    }
    __syncthreads();
}

__device__ void MultibotSimulation::eval(float **actions, float *gamestate)
{
    int tid = threadIdx.x;
    int bot = 0;
    if (tid < 2)
        bot = 0;
    else
        bot = 1;

    int posOffset = 1;
    int velOffset = 5;

    int direction = tid % 2; // which direction (x or y) this thread updates

    // update velocities
    if (tid < 4)
    {
        // Allows precise movement in either direction.
        float preference = actions[bot][direction];
        float accel = preference * MAX_ACCEL;

        // Bound the acceleration change
        accel = fminf(MAX_ACCEL, fmaxf(-MAX_ACCEL, accel));

        // Update the bot's velocity
        gamestate[bot * 2 + direction + velOffset] += accel;

       

        // if(gamestate[tid] > MAX_SPEED && blockIdx.x == 0){
        //     printf("ERROR IN EVAL. activation = %f\n", actions[0][tid]);
        // }
    }

    __syncthreads();

    // update the bots' position
    if (tid < 2)
    {
        bot = tid;

        float Avx = gamestate[bot * 2 + velOffset + 0];
        float Avy = gamestate[bot * 2 + velOffset + 1];

        // Make sure the speed doesn't go above max speed
        float speed = hypotf(Avx, Avy);
        if (speed > MAX_SPEED2)
        {
            float f = MAX_SPEED2 / speed;
            gamestate[bot * 2 + velOffset + 0] *= f;
            gamestate[bot * 2 + velOffset + 1] *= f;
        }

        // Update the bot's position
        gamestate[bot * 2 + posOffset + 0] += gamestate[bot * 2 + velOffset + 0];
        gamestate[bot * 2 + posOffset + 1] += gamestate[bot * 2 + velOffset + 1];
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

        float dx = gamestate[13] - gamestate[bot * 2 + posOffset + 0];
        float dy = gamestate[14] - gamestate[bot * 2 + posOffset + 1];
        float dist = hypotf(dx, dy);

        gamestate[bot + 11] += dist; // 11 = distOffset

        if (dist < 0.5f && threadIdx.x == 0)
        {
            //   printf("%d at target on iter %f\n", blockIdx.x, gamestate[0]);
        }

        if (dist < .5f && threadIdx.x == 0 && gamestate[0] > 55)
        {
            //printf("dist = %f, iter = %d\n", dist, gamestate[0]);
        }

        __syncthreads();

        // Check if we need to reset the sim this iteration
        // if (((int)gamestate[7] + 1) % resetInterval == 0)
        // {
        //     if (threadIdx.x == 0)
        //     {
        //         // // Reset vel and pos
        //         // for(int i = 0; i < 4; i++)
        //         //     gamestate[i] = 0;

        //         //"Rotate" the target position

        //         float new_x = gamestate[4] * cosf(ROTATION_ANGLE) - gamestate[5] * sinf(ROTATION_ANGLE);
        //         float new_y = gamestate[4] * sinf(ROTATION_ANGLE) + gamestate[5] * cosf(ROTATION_ANGLE);

        //         // Update the coordinates
        //         gamestate[4] = new_x;
        //         gamestate[5] = new_y;
        //     }
        // }
    }

    __syncthreads();

    // return dist < epsilon;
    return false;
}

__host__ int MultibotSimulation::getID()
{
    return 3;
}