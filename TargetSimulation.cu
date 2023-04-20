#include "TargetSimulation.cuh"

// NOTE: this must be present in every derived simulation!
extern __constant__ SimConfig config_d;

__constant__ float MAX_SPEED = 4;

// Dead zone around the target that counts as a hit
__constant__ float epsilon = 0.05f;
__constant__ int resetInterval = 2500000000; //Reset gamestate and change target pos every <resetInterval> iters

#define ROTATION_ANGLE 3.141592654f / 2.0f

__device__ void TargetSimulation::eval(float **actions, float *gamestate)
{
    int tid = threadIdx.x;
    
    //LIST OF GAMESTATE VARIABLES
    // 0: x vel
    // 1: y vel
    // 2: x pos
    // 3: y pos
    // 4: targetX
    // 5: targetY
    // 6: totalDistance (summed over all iters)
    // 7: iteration
    
    if (tid < 2) {

        // Allows precise movement in either direction.
        gamestate[tid] = (actions[0][tid] - 0.5) * MAX_SPEED * 2;

        if(gamestate[tid] > MAX_SPEED && blockIdx.x == 0){
            printf("ERROR IN EVAL. activation = %f\n", actions[0][tid]);
        }

    }

    __syncthreads();

    if (tid == 0) {
        int xVel = gamestate[0];
        int yVel = gamestate[1];
        float speed = hypotf(xVel, yVel);
        if(speed > MAX_SPEED){
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
    //if (hypotf(gamestate[0], gamestate[1]) < epsilon) return 0;
    
    
    float dx = gamestate[4] - gamestate[2];
    float dy = gamestate[5] - gamestate[3];
    float dist = hypotf(dx, dy);

    if(threadIdx.x == 0){
        gamestate[6] += dist;
    }

    // if (dist < epsilon && threadIdx.x == 0) {
    //     printf("%d at target on iter %f\n", blockIdx.x, gamestate[7]);
    // }
    __syncthreads();
    
    
    // Check if we need to reset the sim this iteration
    if(((int)gamestate[7] + 1) % resetInterval == 0){
        if(threadIdx.x == 0){
            // Reset vel and pos
            for(int i = 0; i < 4; i++)
                gamestate[i] = 0;

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