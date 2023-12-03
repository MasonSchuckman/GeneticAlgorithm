#include "TargetSimulation.h"
#include <random>

// NOTE: this must be present in every derived simulation!

const float MAX_SPEED = 25;

// Dead zone around the target that counts as a hit
const float epsilon = 0.05f;
const int resetInterval = 40; // Reset gamestate and change target pos every <resetInterval> iters

#define degrees 90.0f
#define ROTATION_ANGLE degrees * 3.141592654f / 180.0f // 90 degrees

void TargetSimulation::getStartingParams(float * startingParams){
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

    //random starting pos
    float startingX = distr(eng);
    float startingY = distr(eng);
    srand(iterationsCompleted);
    double r = 5.0 + iterationsCompleted / 50; // radius of circle
    double angle = ((double)rand() / RAND_MAX) * 2 * 3.14159; // generate random angle between 0 and 2*pi
    targetX = r * std::cos(angle); // compute x coordinate
    targetY = r * std::sin(angle); // compute y coordinate
    // targetX = 10;
    // targetY = 0;
    startingX = 0;
    startingY = 0;
    if (targetX == 0 && targetY == 0)
        targetX = 2;
    
    float optimal = std::hypotf(targetX, targetY) / 2.0 * std::hypotf(targetX, targetY);

    // transfer target coordinates to GPU
    
    startingParams[0] = targetX;
    startingParams[1] = targetY;
    startingParams[2] = optimal;
    startingParams[3] = startingX;
    startingParams[4] = startingY;

    iterationsCompleted++;
}


void TargetSimulation::setupSimulation(int tid, int block, const float *startingParams, float *gamestate)
{
    if (tid == 0)
    {
        gamestate[0] = 0;
        gamestate[1] = 0;
        gamestate[2] = startingParams[3];
        gamestate[3] = startingParams[4];
        gamestate[4] = startingParams[0];
        gamestate[5] = startingParams[1];

        gamestate[6] = 0; // total dist
    }
}

void TargetSimulation::setActivations(int tid, int block, float *gamestate, float ** activs, int iter)
{
    // if(block == 1){
    //     printf("gamestate:\n");
    //     for(int i = 0; i < 10; i++){
    //         printf("%f ", gamestate[i]);
    //     }
    //     printf("\n");
    // }
    const int numInputs = 6;
    for(int i = 0; i < numInputs; i++)
        activs[0][i] = gamestate[i];
    
    
    gamestate[7] = iter;
    
}

void TargetSimulation::eval(int tid, int block, float **actions, float *gamestate)
{

    // LIST OF GAMESTATE VARIABLES
    //  0: x vel
    //  1: y vel
    //  2: x pos
    //  3: y pos
    //  4: targetX
    //  5: targetY
    //  6: totalDistance (summed over all iters)
    //  7: iteration

    for(tid = 0; tid < 2; tid++){

        // Allows precise movement in either direction.
        float preference = (actions[0][tid * 2]) - (actions[0][tid * 2 + 1]); // Decides if the bots wants to go up or down / left or right
        gamestate[tid] = preference * MAX_SPEED;
    }
    
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

int TargetSimulation::checkFinished(int tid, int block, float *gamestate)
{
    // Bot has to be going slowly to the target
    // if (hypotf(gamestate[0], gamestate[1]) < epsilon) return 0;

    float dx = gamestate[4] - gamestate[2];
    float dy = gamestate[5] - gamestate[3];
    float dist = hypotf(dx, dy);

    gamestate[6] += dist;
    

    float m = hypotf(gamestate[4], gamestate[5]);
    // if (dist < epsilon && tid == 0 && m > 100 && block < 10) {
    //     printf("%d at target on iter %f\n", block, gamestate[7]);
    // }

    // Check if we need to reset the sim this iteration
    if (((int)gamestate[7] + 1) % resetInterval == 0)
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

    // if(block == 0 && dist < epsilon){
    //     printf("Reached target, iter = %d\n", (int)gamestate[7]);
    // }

    // return dist < epsilon;
    return false;
}

void TargetSimulation::setOutput(int tid, int block, 
float * output, float * gamestate, const float * startingParams_d){
    static int counter = 0;

    if (gamestate[6] != 0)
        output[block] = -gamestate[6];
    else
        output[block] = 0;

    if (block == 0)
    {
        if (counter % 25 == 0)
            printf("Block %d total dist = %f, efficiency = %f\n", block, gamestate[6], (startingParams_d[2] / gamestate[6]));

        counter++;
    }
    
}

Eigen::MatrixXd TargetSimulation::getState(int& action, float & reward, float *gamestate)
{
	Eigen::MatrixXd state(1,10);
	return state;
}

int TargetSimulation::getID()
{
    return 2;
}