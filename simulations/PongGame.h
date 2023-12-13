#ifndef PONGGAME_H
#define PONGGAME_H

#include "Game.h"

struct PlayerInfo
{
    float x, y, dy, prevY;
    int score, action, team;
    bool hit;
}

class PongGame : Game
{

    PongGame();
    PongGame(const std::string & gameConfigFile);
    ~PongGame() {}    
    void setAgents(std::vector<Agent*> agents_){agents = agents_;}
    void step(); // Steps the environment (updates gamestate based on player actions)
    bool checkFinished();

    std::vector<std::vector<Experience>> getAllExperiences();
    std::vector<Experience> getLatestExperiences();
private:

    MatrixXd getActions();
    MatrixXd getState(int player); // Returns a nx1 Matrix for specified player
    MatrixXd getStates(); // Returns all players as a combined matrix, with each player being 1 column
    
    void loadConfigFile(const std::string & gameConfigFile);


    // Gamestate variables:
    float ballX;
    float ballY;
    float ballVx;
    float ballVy
   
    int stepNumber = 0;
    int generationNumber;

    std::vector<PlayerInfo> players;
    int winner = -1; // (-1 for game not over, 0 is left team, 1 is right team)
    
    
    // (Config constants)
    float WIDTH = 640.0f;
    float HEIGHT = 480.0f;
    float PADDLE_WIDTH = 10.0f;
    float PADDLE_HEIGHT = 50.0f;
    float BALL_RADIUS = 10.0f;
    float BALL_SPEED = 8.0f;
    float PADDLE_SPEED = 6.5f;
    float SPEED_UP_RATE = 1.0f;

    int NUM_BALLS = 1;
    int MAX_ITERS = 0;
    int BOTS_PER_TEAM = 1;
    

};


#endif