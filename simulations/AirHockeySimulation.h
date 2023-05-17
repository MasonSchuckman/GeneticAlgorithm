#ifndef AIRHOCKEY_SIMULATION_H
#define AIRHOCKEY_SIMULATION_H

#pragma once

#include "Simulation.h"


class AirHockeySimulation : public Simulation {
public:
    AirHockeySimulation(){}    
    ~AirHockeySimulation(){}

    void getStartingParams(float * startingParams);

    //Called at the beginning of the kernel. Used to do things like place the bots at their starting positions and such
    void setupSimulation(int tid, int block, const float * startingParams, float * gamestate);

    //Called at the beginning of each sim iteration. 
    void setActivations(int tid, int block, float * gamestate, float ** activs, int iter);

    void eval(int tid, int block, float ** actions, float * gamestate);

    int checkFinished(int tid, int block, float * gamestate);

    void setOutput(int tid, int block, float * output, float * gamestate, const float * startingParams_d);

    
    int getID();

    
};

#endif