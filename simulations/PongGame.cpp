#include "PongGame.h"

PongGame::PongGame()
{
    numActions = 3;
}

PongGame::PongGame(const std::string & gameConfigFile)
{
    PongGame();
    loadConfigFile(gameConfigFile);


}

void PongGame::loadConfigFile(const std::string & gameConfigFile)
{
    using json = nlohmann::json;


    std::ifstream file("C:\\Users\\suprm\\git\\GeneticAlgorithm\\games\\" + gameConfigFile);
    json configFile;

    // Parse the JSON file
    try
    {
        file >> configFile;
    }
    catch (const json::parse_error &e)
    {
        std::cerr << "Failed to parse config file " << gameConfigFile << ": " << e.what() << std::endl;
        exit(1);
    }

    // Read the rest of the simulation configuration from the config
    BOTS_PER_TEAM = configFile["bots_per_team"].get<int>();
    MAX_ITERS = configFile["max_iters"].get<int>();
    NUM_BALLS = configFile["num_balls"].get<int>();
    SPEED_UP_RATE = configFile["speed_up_rate"].get<float>();



}



void PongGame::step()
{
    
}

bool PongGame::checkFinished()
{
    return false;
}

MatrixXd PongGame::getActions()
{
    MatrixXd actions(1, 3);
    return actions;
}

MatrixXd PongGame::getState(int player)
{
    MatrixXd actions(1, 3);
    return actions;
}

MatrixXd PongGame::getStates()
{
    MatrixXd actions(1, 3);
    return actions;
}