#include "CartPoleSimulation.h"
#include <cmath>
#include <random>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <Eigen/Dense>
using Eigen::MatrixXd;

#define GRAVITY 9.8f
#define CART_MASS 1.0f
#define POLE_MASS 0.1f
#define TOTAL_MASS (CART_MASS + POLE_MASS)
#define POLE_LENGTH 0.5f // actually half the pole's length
#define FORCE_MAG 10.0f
#define TAU 0.02f // time interval for updates

// Compile command similar to PongSimulation2.cpp

/*
gamestate[0] // Cart Position
gamestate[1] // Cart Velocity
gamestate[2] // Pole Angle
gamestate[3] // Pole Angular Velocity
gamestate[4] // Iter
gamestate[5] // Reward for iteration
gamestate[6] // Action taken this iteration

*/

void CartPoleSimulation::getStartingParams(float *startingParams)
{
    // Initialize cartpole starting parameters
    double scale = .05;
    double offset = .000;
    startingParams[0] = (((double)rand() / RAND_MAX) - 0.5) * scale + offset; // Cart Position
    startingParams[1] = (((double)rand() / RAND_MAX) - 0.5) * scale + offset; // Cart Velocity
    startingParams[2] = (((double)rand() / RAND_MAX) - 0.5) * scale + offset; // Pole Angle (radians)
    startingParams[3] = (((double)rand() / RAND_MAX) - 0.5) * scale + offset; // Pole Angular Velocity
}

void CartPoleSimulation::setupSimulation(int tid, int block, const float *startingParams, float *gamestate)
{
    for (int i = 0; i < 4; i++) {
        gamestate[i] = startingParams[i];
    }
    gamestate[4] = 0; // iteration
    gamestate[5] = 0; // score or other metric
}

void CartPoleSimulation::setActivations(int tid, int block, float *gamestate, float **activs, int iter)
{
    // Normalize gamestate values for neural network input
    activs[0][0] = gamestate[0]; // Cart Position
    activs[0][1] = gamestate[1]; // Cart Velocity
    activs[0][2] = gamestate[2]; // Pole Angle
    activs[0][3] = gamestate[3]; // Pole Angular Velocity
}

void CartPoleSimulation::eval(int tid, int block, float **actions, float *gamestate)
{
    // Apply action to the cart-pole system
    
    float force = (actions[0][0] < actions[0][1]) ? FORCE_MAG : -FORCE_MAG; //discrete action space
    
    //Enables continuous action space
    // if(abs(actions[0][0]) < FORCE_MAG){
    //     force = actions[0][0];
    // }


    gamestate[6] = (actions[0][0] < actions[0][1]) ? 0 : 1; //Record action taken this iteration

    float cosTheta = cos(gamestate[2]);
    float sinTheta = sin(gamestate[2]);
    //printf("theta = %f\n", gamestate[2]); 
    float temp = (force + POLE_MASS * POLE_LENGTH * gamestate[3] * gamestate[3] * sinTheta) / TOTAL_MASS;
    float angularAccel = (GRAVITY * sinTheta - cosTheta * temp) / (POLE_LENGTH * (4.0 / 3.0 - POLE_MASS * cosTheta * cosTheta / TOTAL_MASS));
    float linearAccel = temp - POLE_MASS * POLE_LENGTH * angularAccel * cosTheta / TOTAL_MASS;

    // Update the gamestate    
    gamestate[1] += TAU * linearAccel;
    gamestate[0] += TAU * gamestate[1];

    gamestate[3] += TAU * angularAccel;
    gamestate[2] += TAU * gamestate[3];
    
    gamestate[5] = 1; //reward for this iteration
    gamestate[4] += 1; // increment iteration

    //Reduce reward for large angles
    //const float angleScale = 0.02;
    //gamestate[5] -= angleScale * abs(gamestate[2]) - angleScale * abs(angularAccel); //we want theta close to zero.
}

int CartPoleSimulation::checkFinished(int tid, int block, float *gamestate)
{
    // The simulation ends if the pole is more than 12 degrees from vertical or the cart moves more than 2.4 units from the center
    if (abs(gamestate[2]) > 12 * M_PI / 180 || abs(gamestate[0]) > 2.4) {
        //gamestate[5] = -10; //negative reward for failing task
        return 1;
    }
    return 0;
}

// Score
void CartPoleSimulation::setOutput(int tid, int block, float *output, float *gamestate, const float *startingParams_d)
{
    output[block] = gamestate[4]; // Number of iterations the pole was balanced
}

Eigen::MatrixXd CartPoleSimulation::getState(int& action, float & reward, float *gamestate)
{
    action = gamestate[6];
    reward = gamestate[5];
    MatrixXd state(4, 1);
    state << gamestate[0], gamestate[1], gamestate[2], gamestate[3];
    return state;
}


// The ID is a unique identifier for this simulation type
int CartPoleSimulation::getID()
{
    return 8; 
}
