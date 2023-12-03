#include "Agent.h"
#include "Net.h"
#include <deque>
#include <memory>
#include <algorithm>

extern int debugging;
extern int print_iterval;
extern int CURRENT_ITERATION;


int trainCalls = 0;

Agent::Agent(int numActions, int numInputs)
    : numActions(numActions), numInputs(numInputs), rd(), gen(rd()), qNet(0.03), replayBuffer(replayBufferSize)
{
    // Add layers to the Q-network
    qNet.addLayer(DenseLayer(numInputs, 12, LeakyRelu, LeakyReluDerivative));
    qNet.addLayer(DenseLayer(12, 6, LeakyRelu, LeakyReluDerivative));
    qNet.addLayer(DenseLayer(6, numActions, linear, linearDerivative)); // Output layer has numActions neurons
    
    targetNet = qNet;

    gamma = 0.95f; // Discount factor
    epsilon = 1.0f; // Exploration rate
    epsilonMin = 0.05f;
    epsilonDecay = 0.9998f;
}

Eigen::VectorXd normalizeState(const Eigen::VectorXd &state) {
    Eigen::VectorXd normalizedState(state.size());
    // Assuming state is [cart_position, cart_velocity, pole_angle, pole_velocity_at_tip]
    // Replace these with the actual ranges for your environment
    double minValue[4] = {-12.4, -50, -12, -500}; // Replace with actual min values
    double maxValue[4] = {12.4, 50, 12, 500};     // Replace with actual max values

    for (int i = 0; i < state.size(); ++i) {
        normalizedState[i] = (state[i] - minValue[i]) / (maxValue[i] - minValue[i]);
    }
    return normalizedState;
}

Eigen::VectorXd Agent::chooseAction(const Eigen::MatrixXd &state_)
{
    
    VectorXd action(numActions);

    // Epsilon-greedy policy
    if (static_cast<float>(rand()) / static_cast<float>(RAND_MAX) < epsilon)
    {
        // Choose a random action
        std::uniform_int_distribution<> dist(0, numActions - 1);
        action = VectorXd::Zero(numActions);
        int index = dist(gen);
        action(index) = 1.0;

    }
    else
    {
        Eigen::VectorXd state = normalizeState(state_);
        VectorXd actionValues = qNet.forward(state); // Get Q-values for each action

        // Choose the best action
        auto maxIndex = std::distance(actionValues.data(), std::max_element(actionValues.data(), actionValues.data() + actionValues.size()));
        action = VectorXd::Zero(numActions);
        action(maxIndex) = 1.0;
    }

    
    // if(((double)rand() / RAND_MAX) > 0.99f){
    //     printf("\n\n\nAction list:\n");
    //     std::cout << action << std::endl;
    // }

    return action;
}

float _train(NeuralNetwork& net, const MatrixXd& inputs, const MatrixXd& targets, int timestep) {
    trainCalls++;
    // Update the learning rate
    net.optimizer.updateLearningRate(timestep);
    /*if (trainCalls % 10000 == 0)
        printf("\n\nLEARNING RATE = %f\n", net.optimizer.learningRate);*/
    // Forward pass
    MatrixXd predictions = net.forward(inputs, true);

    // Calculate loss
    //MatrixXd loss = meanSquaredError(predictions, targets);


    //MSE
    //MatrixXd gradient = (predictions - targets);

    //// Clip gradients
    //double clipValue = 5.000; // You can adjust this value
    //double maxCoeff = gradient.maxCoeff();
    //double minCoeff = gradient.minCoeff();
   
    //if (maxCoeff > clipValue) {
    //    gradient = gradient * (clipValue / maxCoeff);
    //} else if (minCoeff < -clipValue) {
    //    gradient = gradient * (-clipValue / minCoeff);
    //}


    //Huber loss
    MatrixXd gradient = predictions - targets;
    double delta = 1.0; // You can tune this parameter
    for (int i = 0; i < gradient.rows(); ++i) {
        for (int j = 0; j < gradient.cols(); ++j) {
            if (std::abs(gradient(i, j)) <= delta) {
                // In the quadratic zone (MSE)
                gradient(i, j) = gradient(i, j);
            }
            else {
                // In the linear zone
                gradient(i, j) = delta * ((gradient(i, j) > 0) ? 1 : -1);
            }
        }
    }
    
    // Backward pass
    net.backward(gradient);

    // Update weights and biases in all layers
    //net.updateParameters();    

    return (predictions - targets).array().square().sum();
}


double Agent::train()
{
    int K = CURRENT_ITERATION;
    double loss = 0;
    if (replayBuffer.isSufficient()) {
        auto experiences = replayBuffer.sample(minibatchSize);
        for (const auto& e : experiences) {
            K++;
            MatrixXd target = qNet.forward(e.state);
            double qUpdate = e.reward;
            if (!e.done) {
                MatrixXd nextQ = targetNet.forward(e.nextState);
                qUpdate += gamma * nextQ.maxCoeff();
            }
            target(e.action, 0) = qUpdate;
            loss += _train(qNet, e.state, target, CURRENT_ITERATION);

            
        }

        qNet.timestep++;        
    }

    
    if (CURRENT_ITERATION % 25 == 0) {
        targetNet = qNet;
    }


    return loss;
}

void Agent::formatData(const std::vector<episodeHistory>& history)
{
    for (auto it = history.begin(); it != history.end(); it++)
    {
        for (size_t i = 0; i < it->states.size() - 1; i++)
        {
            bool done = i >= it->endIter;
            replayBuffer.add({ it->states[i], it->actions[i], it->rewards[i], it->states[i + 1], done});
        }        
    }
}

double Agent::update(const std::vector<episodeHistory> &history)
{
    formatData(history);
    double totalLoss = 0;
    /*for (int episode = 0; episode < history.size(); episode+=)
    {*/
        // printf("Update Episode %d\n\n", episode);
    CURRENT_ITERATION++;
    totalLoss += train();
    //}

    if (epsilon > epsilonMin)
    {
        epsilon *= epsilonDecay;
    }

    return totalLoss;
}

void Agent::saveNeuralNet()
{
    qNet.writeWeightsAndBiases();
}




//////////////////



    //// printf("Enter train, states size = %d, endIter = %d\n", states.size(), endIter);
    //double loss = 0.0;
    //
    ////, const vector<MatrixXd>& nextStates, const vector<bool>& dones
    //CURRENT_ITERATION++;
    //int K = CURRENT_ITERATION;
    //for (size_t i = 0; i < states.size(); i++)
    //{        
    //    K++;
    //    // Get current Q values for each state
    //    VectorXd target = qNet.forward(states[i], true);
    //    double qUpdate = rewards[i];

    //    // Estimate optimal future value from next state
    //    VectorXd nextQs = (i == states.size() - 1) ? VectorXd::Zero(numActions) : targetNet.forward(states[i + 1], true);
    //    // Calculate target Q value for each state
    //    float maxNextQ = nextQs.maxCoeff();
    //    qUpdate += gamma * maxNextQ;
    //    VectorXd targetQs = target;

    //    int actionIndex = actions[i];
    //    targetQs(actionIndex) = qUpdate;

    //    // Calculate the loss for this state-action pair
    //    VectorXd lossVec = (targetQs - target).array().square();
    //    loss += lossVec.sum();

    //    MatrixXd grad = targetQs - target;
    //    
    //    _train(qNet, states[i], targetQs, CURRENT_ITERATION / 100);

    //    // if(isPrintIteration()){
    //    //     printf("\n\nAction = %d, epoch = %d\n", actionIndex, epoch);
    //    //     //std::cout << "currentQs:\n " << target << std::endl;
    //    //     //std::cout << "nextQ: \n" << nextQs << std::endl;
    //    //     //std::cout << "Target: \n" << targetQs << std::endl;
    //    //     std::cout << "Loss:\n " << lossVec << std::endl;
    //    //     //std::cout << "Grad:\n " << grad << std::endl;
    //    //     //std::cout << "Epsilon:\n " << epsilon << std::endl;

    //    // }

    //    if(K % 5 == 0){
    //        targetNet = qNet;
    //    }
    //}


// double Agent::train(const std::vector<Eigen::MatrixXd> &states, const std::vector<int> &actions, const std::vector<float> &rewards, int endIter)
// {
//     // printf("Enter train, states size = %d, endIter = %d\n", states.size(), endIter);
//     MatrixXd totalGrad;
//     double loss = 0.0;
//     CURRENT_ITERATION++;
//     //, const vector<MatrixXd>& nextStates, const vector<bool>& dones

//     for (size_t i = 0; i < states.size(); ++i)
//     {
        
//         // if(rewards[i] != 1){
//         //     printf("Iter %d\n", i);
//         //     std::cout << "State: " << states[i] << std::endl;
//         //     std::cout << "Action: " << actions[i] << "\t";
//         //     std::cout << "Reward: " << rewards[i] << "\t";
//         //     printf("Epsilon = %f\n", epsilon);
//         // }
//         // Get current Q values for each state
//         VectorXd currentQs = qNet.forward(states[i], true);

//         // Estimate optimal future value from next state
//         VectorXd nextQs = (i == states.size() - 1) ? VectorXd::Zero(numActions) : qNet.forward(states[i + 1], true);

//         // Calculate target Q value for each state
//         float maxNextQ = nextQs.maxCoeff();
//         VectorXd targetQs = currentQs;
//         int actionIndex = actions[i];
//         targetQs(actionIndex) = rewards[i] + gamma * maxNextQ;

//         // Calculate the loss for this state-action pair
//         VectorXd lossVec = (targetQs - currentQs).array().square();
//         loss += lossVec.sum();

//         MatrixXd grad = targetQs - currentQs;
//         if (i == 0)
//         {
//             totalGrad = grad; // For the first sample, initialize totalGrad
//         }
//         else
//         {
//             totalGrad += grad; // Accumulate gradients
//         }

//         if(isPrintIteration() && i % 10 == 0){
//             printf("\n\nAction = %d\n", actionIndex);
//             std::cout << "currentQs:\n " << currentQs << std::endl;
//             std::cout << "nextQ: \n" << nextQs << std::endl;
//             std::cout << "Target: \n" << targetQs << std::endl;
//             std::cout << "Loss:\n " << lossVec << std::endl;
//             std::cout << "Grad:\n " << grad << std::endl;
//         }
//     }

//     // Average the gradients
//     totalGrad /= static_cast<double>(states.size());

//     // Compute the loss by averaging the loss for all state-action pairs
//     loss /= static_cast<double>(states.size());

//     // printf("Going into backward\n");
//     // Perform a single gradient descent step using the accumulated gradients
//     // printf("Total gradient:\n");
//     // std::cout << totalGrad << std::endl;

//     // Clip gradients
//     double clipValue = 0.1; // You can adjust this value
//     double maxCoeff = totalGrad.maxCoeff();
//     double minCoeff = totalGrad.minCoeff();
//     if (maxCoeff > clipValue) {
//         totalGrad = totalGrad * (clipValue / maxCoeff);
//     } else if (minCoeff < -clipValue) {
//         totalGrad = totalGrad * (-clipValue / minCoeff);
//     }
    
//     if(isPrintIteration()){
//         printf("Total gradient:\n");
//         std::cout << totalGrad << std::endl;
//     }
//     qNet.backward(totalGrad);
//     // printf("exit backward\n");

//     return loss;
// }
