#ifndef AGENT_H
#define AGENT_H

#include "Net.h"
#include <deque>
#include <algorithm>
#include <cmath>

const size_t replayBufferSize = 1200 * 5;
const size_t minibatchSize = 128 * 1;
const int targetUpdateFrequency = 1;


template<class ForwardIt, class T>
constexpr // since C++20
void iota(ForwardIt first, ForwardIt last, T value)
{
    for (; first != last; ++first, ++value)
        *first = value;
}

// Experience Replay Buffer
struct Experience {
    MatrixXd state;
    int action;
    double reward;
    MatrixXd nextState;
    bool done;    
    int endIter;
    double tdError = 1;
};

class ReplayBuffer {
    std::vector<Experience> buffer;
    size_t capacity;
    std::vector<double> priorities;
    double defaultPriority = 10000000; // Default high priority for new experiences
    double epsilon = 0.01; // Small constant to avoid zero priority

    size_t currentSize = 0;
    size_t nextIndex = 0;

public:
    ReplayBuffer(size_t cap) : capacity(cap) {}

    void add(const Experience& experience) {
        if (currentSize < capacity) {
            // Buffer is not yet full, just push back
            buffer.push_back(experience);
            priorities.push_back(defaultPriority + experience.endIter / 300.0f);
            ++currentSize;
        }
        else {
            // Buffer is full, overwrite the oldest experience
            buffer[nextIndex] = experience;
            priorities[nextIndex] = defaultPriority + experience.endIter / 300.0f;
        }

        nextIndex = (nextIndex + 1) % capacity;
    }


    vector<Experience> sample(size_t batchSize, vector<int> &indices) {
        bool uniform = false;

        if(uniform)
            return sample(batchSize);

        else {

            vector<Experience> batch(batchSize);
        
            std::vector<double> distribution = computeDistribution();

            std::random_device rd;
            std::mt19937 gen(rd());
        
            std::discrete_distribution<> dist(distribution.begin(), distribution.end());

            for (size_t i = 0; i < batchSize; ++i) {
                int idx = dist(gen);
                /*if(i == 0)
                printf("Idx = %d\n", idx);*/
                indices[i] = idx;
                batch[i] = (buffer[idx]);
            }

            return batch;
        }        
    }

    //uniform sampling
    vector<Experience> sample(size_t batchSize) {
        bool overfitTest = false;
        vector<Experience> batch;

        if (overfitTest) {
            for (int i = 0; i < batchSize; i++)
                batch.push_back(buffer[i]);
        }
        else {
            std::sample(buffer.begin(), buffer.end(), std::back_inserter(batch),
                batchSize, std::mt19937{ std::random_device{}() });
        }
       
        return batch;

    }

    void updatePriority(size_t index, double newTdError) {
        priorities[index] = std::max(std::abs(newTdError), epsilon);
        buffer[index].tdError = newTdError;
    }

    vector<double> computeDistribution() {
        double sum = 0;
        for (auto num : priorities)
            sum += num;
        
        //double sum = std::accumulate(priorities.begin(), priorities.end(), 0.0);
        
        vector<double> distribution;
        for (auto& priority : priorities) {
            distribution.push_back(priority / sum);
        }
        return distribution;
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
    double update(std::vector<episodeHistory>& history);
    void saveNeuralNet();
    void formatData(std::vector<episodeHistory>& history);
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
