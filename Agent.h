#ifndef AGENT_H
#define AGENT_H

#include "Net.h"
//Deep q learning approach
class Agent
{
    public:
    NeuralNetwork qNet; // Network to estimate Q-values
    int numActions;
    int numInputs;
    std::random_device rd;  
    std::mt19937 gen;

    float gamma = 0.95f; // Discount factor
    float epsilon = 1.0f; // Exploration rate
    float epsilonMin = 0.01f;
    float epsilonDecay = 0.995f;

    Agent(int numActions, int numInputs) : numActions(numActions), numInputs(numInputs), rd(), gen(rd()), qNet(0.003)
    {
        // Add layers to the Q-network
        qNet.addLayer(DenseLayer(numInputs, 12, relu, reluDerivative));
        qNet.addLayer(DenseLayer(12, 10, relu, reluDerivative));
        qNet.addLayer(DenseLayer(10, numActions, linear, linearDerivative)); // Output layer has numActions neurons
    }

    VectorXd chooseAction(const MatrixXd& state) {
        VectorXd action(numActions);

        // Epsilon-greedy policy
        if (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < epsilon) {
            // Choose a random action
            std::uniform_int_distribution<> dist(0, numActions - 1);
            action = VectorXd::Zero(numActions);
            action(dist(gen)) = 1.0;
        } else {
            VectorXd actionValues = qNet.forward(state); // Get Q-values for each action

            // Choose the best action
            auto maxIndex = std::distance(actionValues.data(), std::max_element(actionValues.data(), actionValues.data() + actionValues.size()));
            action = VectorXd::Zero(numActions);
            action(maxIndex) = 1.0;
        }

        // Update epsilon
        epsilon = std::max(epsilon * epsilonDecay, epsilonMin);

        return action;
    }

    double train(const vector<MatrixXd>& states, const vector<int>& actions, const vector<float>& rewards, int endIter) {
        //printf("Enter train, states size = %d, endIter = %d\n", states.size(), endIter);
        MatrixXd totalGrad;
        double loss = 0.0;

        //, const vector<MatrixXd>& nextStates, const vector<bool>& dones

        for (size_t i = 0; i < states.size() - 1; ++i) {
            // printf("Iter %d\n", i);
            // std::cout << "State: " << states[i] << std::endl;
            // std::cout << "Action: " << actions[i] << std::endl;
            // std::cout << "Reward: " << rewards[i] << std::endl;

            // Get current Q values for each state
            VectorXd currentQs = qNet.forward(states[i]);

            // Estimate optimal future value from next state
            VectorXd nextQs = i < endIter ? VectorXd::Zero(numActions) : qNet.forward(states[i + 1]);

            // Calculate target Q value for each state
            float maxNextQ = nextQs.maxCoeff();
            VectorXd targetQs = currentQs;
            int actionIndex = actions[i];
            targetQs(actionIndex) = rewards[i] + gamma * maxNextQ;
            
            // Calculate the loss for this state-action pair
            VectorXd lossVec = (targetQs - currentQs).array().square();
            loss += lossVec.sum();

            MatrixXd grad = targetQs - currentQs;
            if (i == 0) {
                totalGrad = grad; // For the first sample, initialize totalGrad
            } else {
                totalGrad += grad; // Accumulate gradients
            }
            
        }

        // Average the gradients
        totalGrad /= static_cast<double>(states.size());

        // Compute the loss by averaging the loss for all state-action pairs
        loss /= static_cast<double>(states.size());

        // printf("Going into backward\n");
        // Perform a single gradient descent step using the accumulated gradients
        //printf("Total gradient:\n");
        //std::cout << totalGrad << std::endl;
        qNet.backward(-totalGrad);
        // printf("exit backward\n");



        return loss;
    }

    double update(const std::vector<episodeHistory>& history) {
        double totalLoss = 0;
        for (int episode = 0; episode < history.size(); episode++) {
            //printf("Update Episode %d\n\n", episode);
            totalLoss += train(history[episode].states, history[episode].actions, history[episode].rewards, history[episode].endIter);
        }
        //printf("Loss : %f\n", totalLoss);


        if(epsilon > epsilonMin){
            epsilon *= epsilonDecay;
        }

        return totalLoss;
    }

    void saveNeuralNet(){
        qNet.writeWeightsAndBiases();
    }


};


//Implementation for policy gradient (wasn't working...gradients were exploding)
// class Agent
// {
//     public:
//     NeuralNetwork policyNet;
//     int numActions = 1;
//     int numInputs = 1;
//     std::random_device rd;  
//     std::mt19937 gen;

//     float gamma = 0.95f;
//     float epsilon = 1.0f;
//     float epsilonMin = 0.0001f;
//     float epsilonDecay = 0.97f;


//     Agent(int numActions, int numInputs) : numActions(numActions), numInputs(numInputs), rd(), gen(rd()), policyNet(0.003)
//     {
//         policyNet.addLayer(DenseLayer(numInputs, 6, LeakyRelu, LeakyReluDerivative)); // Input layer
//         policyNet.addLayer(DenseLayer(6, 6, LeakyRelu, LeakyReluDerivative)); 
//         policyNet.addLayer(DenseLayer(6, 2 * numActions, linear, linearDerivative));

//     }

    
    

//     VectorXd chooseAction(const MatrixXd& output) {
//         VectorXd actions(numActions);
//         for (int i = 0; i < numActions; ++i) {
//             float mean = output(0, 2 * i);
//             float stdDev = std::exp(output(0, 2 * i + 1)); // Example: using exp to ensure stdDev is positive

//             std::normal_distribution<float> distribution(mean, stdDev);
//             actions(i) = distribution(gen);
//         }
//         return actions;
//     }

//     // Policy network update function
//     void updatePolicy(NeuralNetwork& net, const vector<MatrixXd>& episodeStates, const vector<float>& episodeActions, const vector<float>& episodeRewards, int endIter) {
//         float discountFactor = gamma; // Discount factor for future rewards
//         float runningAdditiveReward = 0; // For discounted future rewards

//         // We go backwards through the episode and compute the discounted future reward
//         vector<float> discountedRewards(endIter);
//         for (int i = endIter - 1; i >= 0; --i) {
//             runningAdditiveReward = runningAdditiveReward * discountFactor + episodeRewards[i];
//             discountedRewards[i] = runningAdditiveReward;
//         }

//         // Normalize the rewards to reduce variance
//         float mean = std::accumulate(discountedRewards.begin(), discountedRewards.end(), 0.0) / discountedRewards.size();
//         float squaredSum = std::inner_product(discountedRewards.begin(), discountedRewards.end(), discountedRewards.begin(), 0.0);
//         float stdDev = std::sqrt(squaredSum / discountedRewards.size() - mean * mean);

//         // Normalize and avoid division by zero
//         for (size_t i = 0; i < discountedRewards.size(); ++i) {
//             discountedRewards[i] = (discountedRewards[i] - mean) / (stdDev + 1e-9);
//         }

//         // Update the network for each step in the episode
//         for (int i = 0; i < endIter - 1; ++i) {
            
//             printf("\n\n\nTop of for loop %d\n", i);
//             MatrixXd currentState = episodeStates[i];
            
//             std::cout << "current state:\n" << currentState << std::endl;
//             std::cout << "action:\n" << episodeActions[i] << std::endl;
//             std::cout << "reward:\n" << discountedRewards[i] << std::endl;
//             std::cout << "policyOutput:\n" << net.forward(currentState).transpose() << std::endl;
            
//             // TODO:: Support multiple actions per episode iteration 
//             VectorXd actionTaken (1); //Only 1 action is supported right now 
//             actionTaken[0] = episodeActions[i];

//             float reward = discountedRewards[i];

//             MatrixXd policyOutput = net.forward(currentState).transpose(); // Get the policy output for current state
//             MatrixXd gradOutput = MatrixXd::Zero(policyOutput.rows(), policyOutput.cols());
//             printf("Going into compute gradient\n");

//             if(std::isnan(policyOutput(0,0))){
//                 exit(0);
//             }

//             // Compute the gradient for each action dimension
//             for (int j = 0; j < numActions; ++j) {
//                 float mean = policyOutput(0, 2 * j);
//                 float stdDev = std::exp(policyOutput(0, 2 * j + 1)); // stdDev is positive

//                 printf("Mean = %f, Std = %f\n", mean, stdDev);

//                 // Compute log probability of the action taken
//                 float logProb = -0.5 * std::log(2 * M_PI * stdDev * stdDev + 1e-9) - 
//                                 0.5 * std::pow(actionTaken(j) - mean, 2) / (stdDev * stdDev + 1e-9);
//                 printf("Log prob = %f\n", logProb);
//                 // Compute gradients for mean and standard deviation
//                 gradOutput(0, 2 * j) = (actionTaken(j) - mean) / (stdDev * stdDev + 1e-9) * reward * logProb; // For mean
//                 gradOutput(0, 2 * j + 1) = (std::pow(actionTaken(j) - mean, 2) / (stdDev + 1e-9) - 1) * reward * logProb; // For stdDev

//                 printf("grad output:\n");
//                 std::cout << gradOutput << std::endl;
//             }
//             printf("Going into backward()\n");
//             net.backward(gradOutput.transpose()); // Update the policy network
//             printf("Leaving backward()\n");
//             CURRENT_ITERATION++;
//         }
//     }

//     void update(std::vector<episodeHistory> history)
//     {
//         for(int episode; episode < history.size(); episode++){
//             printf("Update Episode %d\n\n", episode);
//             updatePolicy(policyNet, history[episode].states, history[episode].actions, history[episode].rewards, history[episode].endIter);
//         }
        
//     }

// };
#endif // AGENT_H