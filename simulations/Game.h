#include <string>
#include <Eigen/Dense>
#include <vector>
#include "../Agent.h"
#include "../json.hpp"
#include <fstream>

#ifndef SIMULATION_H
#define SIMULATION_H

class Game
{
public:
    int numActions;

    Game() {}
    Game(const std::string & gameConfigFile){}
    virtual ~Game() {}    
    virtual void setAgents(std::vector<Agent*> agents_){agents = agents_;}
    virtual void step() = 0; // Steps the environment (updates gamestate based on player actions)
    virtual bool checkFinished() = 0;

    // These two are plural in case the game wew're implementing has multiple players. 
    //      In that case, then RETURN_VALUE[i] would corresponse to the ith player.
    std::vector<std::vector<Experience>> getAllExperiences(){ return experiences; }
    std::vector<Experience> getLatestExperiences(){ return experiences[experiences.size() - 1]; }

protected:

    virtual MatrixXd getActions() = 0;
    virtual MatrixXd getState(int player){} // Returns a nx1 Matrix for specified player
    virtual MatrixXd getStates(){} // Returns all players as a combined matrix, with each player being 1 column
    
    virtual void loadConfigFile(const std::string & gameConfigFile){}

    std::vector<Agent*> agents;
    std::vector<std::vector<Experience>> experiences;
};

#endif