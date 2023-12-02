#ifndef CARTPOLE_SIMULATION_H
#define CARTPOLE_SIMULATION_H

#pragma once

#include "Simulation.h"

class CartPoleSimulation : public Simulation {
public:
    CartPoleSimulation(){}    
    ~CartPoleSimulation(){}

    void getStartingParams(float * startingParams);
    void setupSimulation(int tid, int block, const float * startingParams, float * gamestate);
    void setActivations(int tid, int block, float * gamestate, float ** activs, int iter);
    void eval(int tid, int block, float ** actions, float * gamestate);
    int checkFinished(int tid, int block, float * gamestate);
    void setOutput(int tid, int block, float * output, float * gamestate, const float * startingParams_d);
    Eigen::MatrixXd getState(float & action, float & reward, float *gamestate);
    int getID();
    
};

#endif
