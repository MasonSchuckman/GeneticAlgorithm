#ifndef AGENT_H
#define AGENT_H

#include "Net.h"

class Agent {
public:
    Agent(int numActions, int numInputs);
    Eigen::VectorXd chooseAction(const Eigen::MatrixXd& state);
    double train(const std::vector<Eigen::MatrixXd>& states, const std::vector<int>& actions, const std::vector<float>& rewards, int endIter);
    double update(const std::vector<episodeHistory>& history);
    void saveNeuralNet();

public:
    NeuralNetwork qNet; // Q-Network
    NeuralNetwork targetNet; // Target Network
    int numActions;
    int numInputs;
    std::random_device rd;
    std::mt19937 gen;

    float gamma;
    float epsilon;
    float epsilonMin;
    float epsilonDecay;
};

#endif // AGENT_H
