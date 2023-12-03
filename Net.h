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
double tanh(double x);
double tanhDerivative(double x);
MatrixXd softmax(const MatrixXd& x);

// Initialization
MatrixXd heInitialization(int rows, int cols);

// Optimizer
class AdamOptimizer {
public:
    AdamOptimizer(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8, 
                  int decay_steps = 1000, double decay_rate = 0.96);
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
    NeuralNetwork(double lr = 0.01, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8);
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
            // If you have other Layer types, handle them similarly
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

    MatrixXd forward(const MatrixXd& inputs, bool isTraining = false);
    void backward(const MatrixXd& gradOutput);
    void updateParameters();
    void writeWeightsAndBiases();

//private:
    vector<std::unique_ptr<Layer>> layers;
    AdamOptimizer optimizer;
    int timestep;
};

// Loss function
double binaryCrossEntropy(const MatrixXd& Y, const MatrixXd& Y_pred);

#endif // NET_H
