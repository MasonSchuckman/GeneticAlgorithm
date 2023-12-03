#ifndef AGENT_H
#define AGENT_H

#include "Net.h"
#include <deque>
const size_t replayBufferSize = 1000000;
const size_t minibatchSize = 128 * 2;
const int targetUpdateFrequency = 3;

// Experience Replay Buffer
struct Experience {
    MatrixXd state;
    int action;
    double reward;
    MatrixXd nextState;
    bool done;
};

class ReplayBuffer {
    std::deque<Experience> buffer;
    size_t capacity;

public:
    ReplayBuffer(size_t cap) : capacity(cap) {}

    void add(const Experience& experience) {
        if (buffer.size() >= capacity) {
            buffer.pop_front();
        }
        buffer.push_back(experience);
    }

    vector<Experience> sample(size_t batchSize) {
        vector<Experience> batch;
        std::sample(buffer.begin(), buffer.end(), std::back_inserter(batch),
            batchSize, std::mt19937{ std::random_device{}() });
        return batch;

    }

    bool isSufficient() {
        return buffer.size() >= minibatchSize;
    }
};


class Agent {
public:
    Agent(int numActions, int numInputs);
    Eigen::VectorXd chooseAction(const Eigen::MatrixXd& state);
    double train();
    double update(const std::vector<episodeHistory>& history);
    void saveNeuralNet();
    void formatData(const std::vector<episodeHistory>& history);
public:
    ReplayBuffer replayBuffer;
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
