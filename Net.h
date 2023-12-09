#ifndef NET_H
#define NET_H

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <memory>
#include <algorithm>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;



struct episodeHistory {
    int endIter = 0;
    std::vector<Eigen::MatrixXd> states;
    std::vector<int> actions;
    std::vector<float> rewards;    
};

// Utility functions
bool isPrintIteration();
template <typename MatrixType> void printDims(const MatrixType& matrix);
template <typename MatrixType> void printDims(const std::string matName, const MatrixType& matrix);

// Activation functions
double linear(double x);
double linearDerivative(double x);
double relu(double x);
double reluDerivative(double x);
double LeakyRelu(double x);
double LeakyReluDerivative(double x);
double sigmoid(double x);
double sigmoidDerivative(double x);
double tanh_(double x);
double tanhDerivative_(double x);
MatrixXd softmax(const MatrixXd& x);

// Initialization
MatrixXd heInitialization(int rows, int cols);

// Optimizer
class AdamOptimizer {
public:
    AdamOptimizer(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-7, 
                  int decay_steps = 100, double decay_rate = 0.98);

    void update(MatrixXd& params, const MatrixXd& dParams, MatrixXd& m, MatrixXd& v, int t);
    void updateLearningRate(int timestep);

    double initialLearningRate;
    double learningRate;
    double beta1;
    double beta2;
    double epsilon;
    int decaySteps;
    double decayRate;
};

// Layer
class Layer {
public:
    virtual MatrixXd forward(const MatrixXd& inputs, bool isTraining = false) = 0;
    virtual MatrixXd backward(const MatrixXd& gradOutput) = 0;
    virtual ~Layer() {}

    int numNeurons;
    int numWeights;
    MatrixXd input;
    MatrixXd output;
    MatrixXd unactivated_output;
    MatrixXd dInput;
};

typedef double (*ActivationFunction)(double);
typedef MatrixXd (*MatrixActivationFunction)(const MatrixXd&);

class DenseLayer : public Layer {
public:
    DenseLayer(int n_inputs, int n_neurons, ActivationFunction act, ActivationFunction actDeriv);
    MatrixXd forward(const MatrixXd& inputs, bool isTraining = false) override;
    MatrixXd backward(const MatrixXd& error) override;

    MatrixXd weights;
    MatrixXd biases;
    MatrixXd dWeights;
    MatrixXd dBias;
    MatrixXd mWeights; // For Adam optimizer, for weights
    MatrixXd vWeights; // For Adam optimizer, for weights

    MatrixXd mBiases;  // For Adam optimizer, for biases
    MatrixXd vBiases;  // For Adam optimizer, for biases

//private:
    ActivationFunction activation;
    ActivationFunction activationDerivative;
};

//class BatchNormalizationLayer : public Layer {
//public:
//    BatchNormalizationLayer(int n_inputs);
//    MatrixXd forward(const MatrixXd& inputs, bool isTraining) override;
//    MatrixXd backward(const MatrixXd& gradOutput) override;
//
//    double learningRate = 0.001;
//private:
//    MatrixXd gamma; // Scale parameters
//    MatrixXd beta;  // Shift parameters
//    double momentum;
//    MatrixXd runningMean;
//    MatrixXd runningVariance;
//    MatrixXd batchMean;
//    MatrixXd batchVariance;
//};
//

class DropoutLayer : public Layer {
public:
    DropoutLayer(double rate);
    MatrixXd forward(const MatrixXd& inputs, bool isTraining) override;
    MatrixXd backward(const MatrixXd& gradOutput) override;

//private:
    double dropoutRate;
    MatrixXd mask;
    std::default_random_engine generator;
};

// Neural Network
class NeuralNetwork {
public:
    NeuralNetwork(double lr = 0.0001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-7);
    template<typename LayerType>
    void addLayer(const LayerType& layer) {
        layers.push_back(std::make_unique<LayerType>(layer));
    }

    NeuralNetwork& operator=(const NeuralNetwork& other) {
        // Guard self assignment
        if (this == &other) {
            return *this;
        }

        // Clear existing layers
        layers.clear();

        // Copy layers from other network
        for (const auto& layer : other.layers) {
            if (auto denseLayer = dynamic_cast<const DenseLayer*>(layer.get())) {
                layers.push_back(std::make_unique<DenseLayer>(*denseLayer));
            }
        }

        // Copy optimizer settings
        optimizer.learningRate = other.optimizer.learningRate;
        optimizer.beta1 = other.optimizer.beta1;
        optimizer.beta2 = other.optimizer.beta2;
        optimizer.epsilon = other.optimizer.epsilon;

        // Copy other necessary members if there are any

        return *this;
    }

    // You might also need a copy constructor
    NeuralNetwork(const NeuralNetwork& other) {
        *this = other;  // Delegate to assignment operator
    }


    double computeL2NormWith(const NeuralNetwork& otherNetwork) const {
        double l2Norm = 0.0;

        for (size_t i = 0; i < layers.size(); ++i) {
            const auto& thisLayer = dynamic_cast<const DenseLayer&>(*layers[i]);
            const auto& otherLayer = dynamic_cast<const DenseLayer&>(*otherNetwork.layers[i]);

            // Compute the difference in weights and biases, and add their L2 norm to the total
            MatrixXd weightDiff = thisLayer.weights - otherLayer.weights;
            MatrixXd biasDiff = thisLayer.biases - otherLayer.biases;

            l2Norm += weightDiff.squaredNorm() + biasDiff.squaredNorm();
        }

        return sqrt(l2Norm);
    }

    MatrixXd forward(const MatrixXd& inputs, bool isTraining = false);

    // Polyak Averaging function for soft target nextwork updates
    void polyakUpdate(const NeuralNetwork& primaryNetwork, double polyakCoefficient);

    void backward(const MatrixXd& gradOutput);
    void updateParameters();
    void writeWeightsAndBiases();
    void write_weights_and_biases2();

    void printWeightsAndBiases();

//private:
    vector<std::unique_ptr<Layer>> layers;
    AdamOptimizer optimizer;
    double lambda; // for L2 Regularization
    int timestep;
};

// Loss function
double binaryCrossEntropy(const MatrixXd& Y, const MatrixXd& Y_pred);

#endif // NET_H
