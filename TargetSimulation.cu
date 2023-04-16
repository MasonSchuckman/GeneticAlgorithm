#include "TargetSimulation.cuh"

// NOTE: this must be present in every derived simulation!
extern __constant__ SimConfig config_d;

const MAX_SPEED = 1;
const float target_x = 5; const float target_y = -2;
// Dead zone around the target that counts as a hit
const float epsilon = .2;

__device__ void TargetSimulation::eval(float **actions, float *gamestate)
{
    int tid = threadIdx.x;

    // 0: x vel
    // 1: y vel
    // 2: x pos
    // 3: y vel
    if (tid < 2) {
        if (actions[0][tid] < .5)
        {
            gamestate[tid] = 1;
        }
        else if (actions[0][tid] < .5) {
            gamestate[tid] = -1;
        }
    }

    __syncthreads();

    if (tid == 0) {
        int x = gamestate[0]; int y = gamestate[1];
        float n = sqrt(pow(x, 2) + pow(y, 2));
        float f = MAX_SPEED / n;
        gamestate[0] *= f; gamestate[1] *= f;

        gamestate[2] += gamestate[0]; gamestate[3] += gamestate[1];
    }
}

__device__ int TargetSimulation::checkFinished(float *gamestate)
{
    // Bot has to be going slowly to the target
    if (gamestate[2] > epsilon) return 0;
    if (gamestate[3] > epsilon) return 0;

    float dx = target_x - gamestate[0];
    float dy = target_y - gamestate[1];
    float dist = sqrt(pow(dx, 2) + pow(dy, 2));

    return dist < epsilon;
}

__host__ int TargetSimulation::getID()
{
    return 1;
}