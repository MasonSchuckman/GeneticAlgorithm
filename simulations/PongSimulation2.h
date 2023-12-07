#ifndef PONG_SIMULATION2_H
#define PONG_SIMULATION2_H

#pragma once

#include "Simulation.h"


class PongSimulation2 : public Simulation {
public:
    PongSimulation2(){}    
    ~PongSimulation2(){}

    void getStartingParams(float * startingParams);

    //Called at the beginning of the kernel. Used to do things like place the bots at their starting positions and such
    void setupSimulation(int tid, int block, const float * startingParams, float * gamestate);

    //Called at the beginning of each sim iteration. 
    void setActivations(int tid, int block, float * gamestate, float ** activs, int iter);

    void eval(int tid, int block, float ** actions, float * gamestate);

    int checkFinished(int tid, int block, float * gamestate);

    void setOutput(int tid, int block, float * output, float * gamestate, const float * startingParams_d);
    
    Eigen::MatrixXd getState(int & action, float & reward, float *gamestate);

    Eigen::MatrixXd getStateP1(int& action, float& reward, float** activs);
    Eigen::MatrixXd getStateP2(int& action, float& reward, float** activs);


    int getID();

    
};

#endif