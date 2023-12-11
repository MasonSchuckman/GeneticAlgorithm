#include "Agent.h"
#include "Net.h"
#include <deque>
#include <memory>
#include <algorithm>

extern int debugging;
extern int print_iterval;
extern int CURRENT_ITERATION;
template <typename MatrixType>
void printDims(const MatrixType& matrix)
{
    std::cout << "Rows: " << matrix.rows() << ", Columns: " << matrix.cols() << std::endl;
}
template <typename MatrixType>
void printDims(const std::string matName, const MatrixType& matrix)
{
    std::cout << matName << "\n";
    std::cout << "Rows: " << matrix.rows() << ", Columns: " << matrix.cols() << std::endl;
}


int trainCalls = 0;

Agent::Agent(int numActions, int numInputs)
    : numActions(numActions), numInputs(numInputs), rd(), gen(rd()), qNet(0.03, 0.9, 0.999), replayBuffer(replayBufferSize)
{
    // Add layers to the Q-network
    qNet.addLayer(DenseLayer(numInputs, 24, LeakyRelu, LeakyReluDerivative));
    //// qNet.addLayer(BatchNormalizationLayer(64));
    qNet.addLayer(DenseLayer(24, 12, LeakyRelu, LeakyReluDerivative));
    qNet.addLayer(DenseLayer(12, 10, LeakyRelu, LeakyReluDerivative));
    qNet.addLayer(DenseLayer(10, 10, LeakyRelu, LeakyReluDerivative));
    //qNet.addLayer(DenseLayer(10, 10, LeakyRelu, LeakyReluDerivative));

    //qNet.addLayer(DenseLayer(16, 12, LeakyRelu, LeakyReluDerivative));
    //qNet.addLayer(DenseLayer(12, 8, LeakyRelu, LeakyReluDerivative));

   // qNet.addLayer(DenseLayer(6, 6, LeakyRelu, LeakyReluDerivative));
   // qNet.addLayer(DenseLayer(6, 6, LeakyRelu, LeakyReluDerivative));
    //qNet.addLayer(DenseLayer(6, 6, LeakyRelu, LeakyReluDerivative));
   // qNet.addLayer(DenseLayer(6, 6, LeakyRelu, LeakyReluDerivative));

    //qNet.addLayer(DenseLayer(10, 5, LeakyRelu, LeakyReluDerivative));    
    //qNet.addLayer(DenseLayer(4, 3, LeakyRelu, LeakyReluDerivative));


    qNet.addLayer(DenseLayer(10, numActions, linear, linearDerivative)); // Output layer has numActions neurons
    
    targetNet = qNet;

    gamma = 0.98f; // Discount factor
    epsilon = 1.0f; // Exploration rate
    epsilonMin = 0.01f;
    epsilonDecay = 0.99984;
}

Eigen::VectorXd normalizeState(const Eigen::VectorXd &state) {
    Eigen::VectorXd normalizedState(state.size());
    // Assuming state is [cart_position, cart_velocity, pole_angle, pole_velocity_at_tip]
    // Replace these with the actual ranges for your environment
    double minValue[4] = {-2.4, -5, -12, -500}; // Replace with actual min values
    double maxValue[4] = {2.4, 5, 12, 500};     // Replace with actual max values

    for (int i = 0; i < state.size(); ++i) {
        normalizedState[i] = (state[i] - minValue[i]) / (maxValue[i] - minValue[i]);
    }
    return normalizedState;
}

//Eigen::MatrixXd normalizeState(const Eigen::MatrixXd& state) {
//    return normalizeState(state.);
//}

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
        //Eigen::VectorXd state = normalizeState(state_);
        VectorXd actionValues = qNet.forward(state_); // Get Q-values for each action
        //if(std::abs(actionValues(0) - actionValues(1)) > 1){
        /*if (isPrintIteration()) {
            std::cout << "State : \n" << state_ << std::endl;
            std::cout << "Action values\n" << actionValues << std::endl;
        }
        */
        // Choose the best action
        auto maxIndex = std::distance(actionValues.data(), std::max_element(actionValues.data(), actionValues.data() + actionValues.size()));
        action = VectorXd::Zero(numActions);
        action(maxIndex) = 1.0;
        //std::cout << "Action vec \n" << action << std::endl;

        //if (isPrintIteration()) {
        //    //if(((double)rand() / RAND_MAX) > 0.99f){
        //    printf("\nAction list:\n");
        //    std::cout << actionValues.transpose() << std::endl;
        //}

        
    }

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


    //std::cout << "\ntargets:\n" << targets << std::endl;
    //std::cout << "\npredictions:\n" << predictions << std::endl;
    //std::cout << "\ninputs:\n" << inputs << std::endl;

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
   // std::cout << "Predictions:\n" << (predictions.leftCols(5)).transpose() << std::endl;
    //std::cout << "Targets    :\n" << (targets.leftCols(5)).transpose() << std::endl;

    MatrixXd gradient = predictions - targets;
    //printf("Loss = %f\t", (gradient).array().square().sum() / (gradient.cols()));
    //std::cout << "Gradient:\n" << (gradient.leftCols(5)).transpose() << std::endl;

    gradient /= (predictions.cols());
     double delta = 1; 
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
 

    return (gradient).array().square().sum() * predictions.cols();
}

//Batch training
double Agent::train()
{
    double total = 0;
    double abTotal = 0;
    int large = 0;
    int small = 0;
    int K = CURRENT_ITERATION;
    double loss = 0;
    if (replayBuffer.isSufficient()) {
        //if (K % 10 == 0)
        {
            //printf("L2 norm between Q and Target : %f\n", qNet.computeL2NormWith(targetNet));
           //targetNet.polyakUpdate(qNet, 0.9995);
            //targetNet = qNet;
        }
        
        //targetNet.polyakUpdate(qNet, 0.99);

        std::vector<int> indices(minibatchSize); // To store indices of experiences in the batch
        auto experiences = replayBuffer.sample(minibatchSize, indices);


        MatrixXd states(experiences[0].state.rows(), minibatchSize);
        MatrixXd nextStates(experiences[0].nextState.rows(), minibatchSize);
        MatrixXd rewards(1, minibatchSize);
        std::vector<int> actions(minibatchSize);
        std::vector<bool> dones(minibatchSize);

       

        for (int i = 0; i < minibatchSize; i++)
        {
            states.col(i) = experiences[i].state.col(0);
            nextStates.col(i) = experiences[i].nextState.col(0);

            //std::cout << "State i \n" << experiences[i].state.col(0) << std::endl;
            //std::cout << "State i=1 \n" << experiences[i].nextState.col(0) << std::endl;

            rewards(0, i) = experiences[i].reward;
            actions[i] = experiences[i].action;
            dones[i] = experiences[i].done;
        }

        
        
        MatrixXd qValues = qNet.forward(states);
        MatrixXd qUpdates = rewards;
        MatrixXd q_values_next = targetNet.forward(nextStates);
        MatrixXd targets = qValues;

        if ((qNet.forward(states) - targetNet.forward(nextStates)).maxCoeff() > 10) {
            //std::cout << "\MAX DIFF = " << (qNet.forward(states) - targetNet.forward(nextStates)).maxCoeff() << std::endl;
            //std::cout << "\nDIFF = " << (qNet.forward(states) - targetNet.forward(nextStates)) << std::endl;
            //std::cout << "\nMAX DIFF states= " << (nextStates - states).maxCoeff() << std::endl;
           // std::cout << "\nDIFF STATES = " << (nextStates - states) << std::endl;
            //std::cout << "\REWARDS = " << rewards << std::endl;
           // std::cout << "\MAX REWARDS = " << rewards.maxCoeff() << std::endl;       
        }
         
        
        for (int i = 0; i < minibatchSize; i++) {
            targets(actions[i], i) = rewards(0,i);
            if (!dones[i]) {
                /*if(i < 5)

                printf("Max coef = %f\n", );
                double qq = nextQs.col(i).maxCoeff();*/
                qUpdates(0,i) += gamma * q_values_next.col(i).maxCoeff();
                targets(actions[i], i) += gamma * q_values_next.col(i).maxCoeff();

            }
        }

        std::vector<double> td_errors(minibatchSize);
        //printf("\n");
        // update target
        for (int i = 0; i < minibatchSize; i++) {     
            td_errors[i] = qUpdates(0, i) - qValues(actions[i], i);
            
            
            //if (qUpdates(0, i) > 10) {
            //    //printf("qUpdate = %f, original = %f\n", qUpdates(0, i), targets(actions[i], i));
            //    //std::cout << "States : \n" << states.col(i) << "\nNext States :\n" << nextStates.col(i) << std::endl;
            //}

            qValues(actions[i], i) = qUpdates(0, i);
            
            
        }
        //printf("\n");


       /* if (true || CURRENT_ITERATION > 5000 && (isPrintIteration())) {
            std::cout << "Rewards:\n" << rewards.leftCols(5) << std::endl;
            std::cout << "Next Q's:\n" << (q_values_next.leftCols(5)).transpose() << std::endl;
            std::cout << "Targets:\n" << (targets.leftCols(5)).transpose() << std::endl;

        }*/

        for (int i = 0; i < 2; i++)
            loss += _train(qNet, states, targets, CURRENT_ITERATION);

        

        // Update priorities
        for (int i = 0; i < minibatchSize; i++) {
            replayBuffer.updatePriority(indices[i], td_errors[i]);
        }
        

        int zz = 0;
        for (auto e : experiences) {
            double pre = 0;
            if (e.tdError != 1) {

                pre = e.tdError;
            }



            if (std::abs(td_errors[zz]) > 10.001) {
                large++;
                total += td_errors[zz];
                abTotal += std::abs(td_errors[zz]);
            }
            else {
                small++;
            }

            if (pre != 0) {
               // printf("Seen again %d : %f\t", indices[zz], pre);
                //printf("New : %f, Delta TD error = %f\n", td_errors[zz], (pre - td_errors[zz]));
            }
            zz++;
        }

        
            
        //qNet.timestep++;
    }

    //if (isPrintIteration())
    //    printf("\nLarge = %d, small = %d, avg for large: %f, avg ab: %f\n", large, small, (total / large), (abTotal / large));
    return loss;
}


void Agent::formatData(std::vector<episodeHistory>& history)
{
    int start = 0;

    // Eligibility tracing? (Give reward for actions leading up to positive reward
    float lambda = 0.996;  // Eligibility trace decay factor
    //if(isPrintIteration())
        //printf("\nTraces:\n");
    for (auto& ep : history)
    {
        // If the episode has rewards, apply eligibility traces
        if (!ep.rewards.empty())
        {
            float trace = 0.0;
            // Traverse the rewards in reverse
            for (int i = ep.rewards.size() - 1; i >= 0; --i)
            {
                trace = ep.rewards[i] + lambda * trace;
                ep.rewards[i] = trace;
                if (ep.actions[i] == 2)
                    ep.rewards[i] += 0.0001; // Incentivizes minimal energy gameplay
                //if (isPrintIteration())
                    //printf("%f\n", trace);
            }
        }
    }


    int c = 0;
    for (auto it = history.begin(); it != history.end(); it++)
    {
        for (size_t i = start; i < it->states.size() - 1; i++)
        {
            bool adding = true;// (CURRENT_ITERATION < 1500) || (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) > 0.6;

            if (adding)
            {
                bool done = i >= it->endIter - 2;
                replayBuffer.add({ it->states[i], it->actions[i], it->rewards[i], it->states[i + 1], done, it->endIter });
                //std::cout << "History :\n" << it->states[i] << std::endl;;
                c++;
            }
           
        }        
    }

}

double Agent::update(std::vector<episodeHistory> &history)
{
    //if(CURRENT_ITERATION < 2500) //turn on for overfit experiment
        formatData(history);
    double totalLoss = 0;
    CURRENT_ITERATION++;

    /*if (CURRENT_ITERATION > 20)
        for (int i = 0; i < 100000; i++) {
            CURRENT_ITERATION++;
            printf("\ni = %d, \n\nLoss : %f\n\n\n", i, train());
        }
      */      
    
    if (CURRENT_ITERATION % 600 == 0)
        targetNet = qNet;

    for(int i = 0; i < 3; i++)
        totalLoss += train();
    //qNet.timestep++;

    if (replayBuffer.isSufficient() && epsilon > epsilonMin)
    {
        epsilon *= epsilonDecay;
    }

    return totalLoss;
}

void Agent::saveNeuralNet()
{
    qNet.writeWeightsAndBiases("RL-bot-extra.data");
}



/////////

// batchSize = 1 updates (OLD)
//double Agent::train()
//{
//    double total = 0;
//    double abTotal = 0;
//    int large = 0;
//    int small = 0;
//    int K = CURRENT_ITERATION;
//    double loss = 0;
//    if (replayBuffer.isSufficient()) {
//        std::vector<int> indices(minibatchSize); // To store indices of experiences in the batch
//        auto experiences = replayBuffer.sample(minibatchSize, indices);
//
//        int c = 0;
//        for (const auto& e : experiences) {
//            K++;
//            MatrixXd target = qNet.forward(e.state);
//            double qUpdate = e.reward;
//            if (!e.done) {
//                MatrixXd nextQ = targetNet.forward(e.nextState);
//                qUpdate += gamma * nextQ.maxCoeff();
//            }
//            double pre = 0;
//            if (e.tdError != 1) {
//                
//                pre = e.tdError;
//            }
//
//            double td_error = qUpdate - target(e.action, 0);
//            if (std::abs(td_error) > 10.001) {
//                large++;
//                total += td_error;
//                abTotal += std::abs(td_error);
//            }
//            else {
//                small++;
//            }
//            replayBuffer.updatePriority(indices[c], td_error);
//
//
//            target(e.action, 0) = qUpdate;
//            for(int i = 0; i < 5; i++)
//            loss += _train(qNet, e.state, target, CURRENT_ITERATION);
//            
//
//
////            if (isPrintIteration()) {
////                if (e.endIter > 300)
////                    printf("Long episode td error = %f\n", td_error);
////
////               /* if (pre != 0) {
////                    printf("Seen again %d : %f\t", indices[c], pre);
////                    printf("New : %f, Delta TD error = %f\n", td_error, (pre - td_error));
////                }
////;*/
////            }
//                
//
//            if (K % 3 == 0) {
//                targetNet = qNet;
//            }
//
//            c++;
//        }
//
//        qNet.timestep++;        
//    }
//
//    if(isPrintIteration())
//    printf("\n\nLarge = %d, small = %d, avg for large: %f, avg ab: %f\n", large, small, (total / large), (abTotal / large));
//
//    return loss;
//}


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
