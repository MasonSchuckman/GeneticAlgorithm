#ifndef NET_H
#define NET_H

#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <memory>
#include <algorithm> //for std::shuffle
#include <fstream>


using Eigen::MatrixXd;
using Eigen::VectorXd;

using std::vector;


struct episodeHistory
{
    int endIter = 0;
    std::vector<Eigen::MatrixXd> states;
    std::vector<int> actions;
    std::vector<float> rewards;    
};


int debugging_ = 0;
int print_iterval = 1;
int CURRENT_ITERATION = 0;

bool isPrintIteration(){
    return CURRENT_ITERATION % print_iterval == 0;
}

template <typename MatrixType>
void printDims(const MatrixType& matrix) {
    std::cout << "Rows: " << matrix.rows() << ", Columns: " << matrix.cols() << std::endl;
}

template <typename MatrixType>
void printDims(const std::string matName, const MatrixType& matrix) {
    std::cout << matName << "\n";
    std::cout << "Rows: " << matrix.rows() << ", Columns: " << matrix.cols() << std::endl;
}



// Define some activation functions

double linear(double x) {
    return x;
}

double linearDerivative(double x) {
    return 1;
}

// Define the ReLU activation function and its derivative
double relu(double x) {
    return x > 0 ? x : 0; // If x is greater than 0, return x, otherwise return 0
}

double reluDerivative(double x) {
    return x > 0 ? 1 : 0; // If x is greater than 0, return 1, otherwise return 0
}

double LeakyRelu(double x){
    return x > 0 ? x : x / 8;
}

double LeakyReluDerivative(double x){
    return x > 0 ? 1 : 1.0 / 8.0;
}

// Sigmoid and its derivative
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1 - s);
}

// Tanh and its derivative
double tanh(double x) {
    return std::tanh(x);
}

double tanhDerivative(double x) {
    double t = tanh(x);
    return 1 - t * t;
}

// Softmax function (useful for multi-class classification)
MatrixXd softmax(const MatrixXd& x) {
    MatrixXd expX = x.unaryExpr([](double v) { return exp(v); });
    return expX.array().colwise() / expX.rowwise().sum().array();
}

/////////////////


// He initialization for weight matrix based on number of inputs
MatrixXd heInitialization(int rows, int cols) {
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, sqrt(2.0 / cols)); // Using He-et-al initialization
    MatrixXd m = MatrixXd::Zero(rows, cols);
    for (int i = 0; i < m.rows(); ++i)
        for (int j = 0; j < m.cols(); ++j)
            m(i, j) = distribution(generator);
    return m;
}

class AdamOptimizer {
public:
    double learningRate;
    double beta1;
    double beta2;
    double epsilon;

    AdamOptimizer(double lr = 0.001, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
        : learningRate(lr), beta1(b1), beta2(b2), epsilon(eps) {}

    void update(MatrixXd& weights, const MatrixXd& dWeights, MatrixXd& m, MatrixXd& v, int t) {

        m = beta1 * m + (1 - beta1) * dWeights;
        v = beta2 * v + (1 - beta2) * dWeights.array().square().matrix();

        MatrixXd mHat = m / (1.0 - pow(beta1, t));
        MatrixXd vHat = v / (1.0 - pow(beta2, t));

        weights.array() -= learningRate * mHat.array() / (vHat.array().sqrt() + epsilon);

    }
};


class Layer {
//protected:
    // MatrixXd input;
    // MatrixXd output;
    // MatrixXd dInput;

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

class DenseLayer : public Layer{
    ActivationFunction activation;
    ActivationFunction activationDerivative;

public:
    MatrixXd weights;
    MatrixXd biases;
    MatrixXd dWeights;
    MatrixXd dBias;
    MatrixXd m; // For Adam optimizer
    MatrixXd v; // For Adam optimizer

    DenseLayer(int n_inputs, int n_neurons, ActivationFunction act, ActivationFunction actDeriv)
    : weights(heInitialization(n_neurons, n_inputs)),
      biases(MatrixXd::Zero(n_neurons, 1)),
      activation(act),
      activationDerivative(actDeriv),
      m(MatrixXd::Zero(n_neurons, n_inputs)),
      v(MatrixXd::Zero(n_neurons, n_inputs))
    {
        numNeurons = n_neurons;
        numWeights = n_neurons * n_inputs;
    }

    MatrixXd forward(const MatrixXd& inputs, bool isTraining = false) override {
        Layer::input = inputs;

        // Compute the matrix product
        MatrixXd z = weights * inputs;
        // Broadcast the bias addition across columns
        unactivated_output = z.colwise() + biases.col(0);

        output = unactivated_output.unaryExpr(activation); // Apply activation function

        // Print layer information
        if (debugging_ >= 3 && isPrintIteration()) {
            std::cout << "Layer Information:" << std::endl;
            std::cout << "Inputs:" << std::endl << inputs << std::endl;
            std::cout << "Weights:" << std::endl << weights << std::endl;
            std::cout << "Biases:" << std::endl << biases << std::endl;
            std::cout << "Activations:" << std::endl << output << std::endl;
        }
        return output;
    }
   

    MatrixXd backward(const MatrixXd& error) override {
       
        MatrixXd output_delta = error.array() * unactivated_output.unaryExpr(activationDerivative).array();
        dWeights = output_delta * input.transpose();
        dBias = output_delta.rowwise().sum();

        return output_delta;
    }
};


class DropoutLayer : public Layer{
    double dropoutRate;
    MatrixXd mask;
    std::default_random_engine generator;
    MatrixXd output;

public:
    DropoutLayer(double rate) : dropoutRate(rate)
       {
        generator.seed(static_cast<unsigned int>(time(0))); // Seed for randomness
        numNeurons = 0;
    }

    MatrixXd forward(const MatrixXd& inputs, bool isTraining) override {
        Layer::input = inputs;   

        if (isTraining) {
            std::bernoulli_distribution distribution(1 - dropoutRate);
            mask = MatrixXd(inputs.rows(), inputs.cols());
            for (int i = 0; i < mask.rows(); ++i) {
                for (int j = 0; j < mask.cols(); ++j) {
                    mask(i, j) = distribution(generator) ? 1.0 : 0.0;
                }
            }
            mask = mask / (1.0 - dropoutRate); // Scale the activations to not reduce the expected sum
            output = inputs.cwiseProduct(mask);
        } else {
            output = inputs; // Do not apply dropout during testing
        }
        
        return output;
    }

    MatrixXd backward(const MatrixXd& gradOutput) {
        dInput = gradOutput.cwiseProduct(mask); // Only backprop through the non-dropped units

        return dInput;
    }
};


class NeuralNetwork {
public:

    vector<std::unique_ptr<Layer>> layers;
    AdamOptimizer optimizer;
    int timestep = 1;


    NeuralNetwork(double lr = 0.01, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
        : optimizer(lr, b1, b2, eps) {}
    
    template<typename LayerType>
    void addLayer(const LayerType& layer) {
        layers.push_back(std::make_unique<LayerType>(layer));
    }

    MatrixXd forward(const MatrixXd& inputs, bool isTraining = false) {
        MatrixXd currentOutput = inputs;
        for (auto& layer : layers) {
            currentOutput = layer->forward(currentOutput, isTraining);
        }
        return currentOutput;
    }

    void backward(const MatrixXd& gradOutput) {
        MatrixXd currentGradient = gradOutput;

        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            if(it != layers.rbegin()){
                if (auto denseLayer = dynamic_cast<DenseLayer*>((it-1)->get())){                    
                    currentGradient = denseLayer->weights.transpose() * currentGradient;
                }                
            }
            //printDims("currentGradient1", currentGradient);

            currentGradient = (*it)->backward(currentGradient);
            //printDims("currentGradient2", currentGradient);

            if(debugging_ && isPrintIteration())
                std::cout << "Current gradient: " << std::endl << currentGradient << std::endl;

            // If it's a DenseLayer, update its weights and biases
            if (auto denseLayer = dynamic_cast<DenseLayer*>(it->get())) {
                // Update the weights
                optimizer.update(denseLayer->weights, denseLayer->dWeights, denseLayer->m, denseLayer->v, timestep);

                // Update the biases
                denseLayer->biases -= optimizer.learningRate * denseLayer->dBias;
            }        
        }

        timestep++;
    }

    void writeWeightsAndBiases()
    {
        int TOTAL_BOTS = 1;
        int numLayers = layers.size() + 1;
        int * layerShapes = new int[numLayers];
        int totalWeights = 0;
        int totalNeurons = 0;
        
        //Account for input layer
        layerShapes[0] = layers[0]->input.size();
        totalNeurons += layerShapes[0];

        for(int i = 1; i < numLayers; i++){
            layerShapes[i] = layers[i - 1]->numNeurons;

            totalNeurons += layers[i - 1]->numNeurons;
            totalWeights += layers[i - 1]->numWeights;
        }

        float * weights = new float[totalWeights];
        int c = 0;
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            if (auto denseLayer = dynamic_cast<DenseLayer*>((it->get()))){   
                for(int j = 0; j < denseLayer->numWeights; j++){
                    weights[c] = denseLayer->weights.transpose().data()[c];
                    c++;
                }
            }
            
        }

        float * biases = new float[totalNeurons];
        for(int i = 0; i < layerShapes[0]; i++)
            biases[i] = 0;

        c = layerShapes[0];
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            if (auto denseLayer = dynamic_cast<DenseLayer*>((it->get()))){   
                for(int j = 0; j < denseLayer->numNeurons; j++){
                    biases[c] = denseLayer->biases.transpose().data()[c];
                    c++;
                }
            }
            
        }

        std::ofstream outfile("RL-bot.data", std::ios::out | std::ios::binary); // this might be more space efficient
        // std::ofstream outfile("allBots.data");
        //  outfile << "all bots:\n";
        //  Write the total number of bots
        outfile.write(reinterpret_cast<const char *>(&TOTAL_BOTS), sizeof(int));

        // Write the total number of weights and neurons
        outfile.write(reinterpret_cast<const char *>(&totalWeights), sizeof(int));
        outfile.write(reinterpret_cast<const char *>(&totalNeurons), sizeof(int));

        // Write the number of layers and their shapes
        outfile.write(reinterpret_cast<const char *>(&numLayers), sizeof(int));
        for (int i = 0; i < numLayers; i++)
        {
            outfile.write(reinterpret_cast<const char *>(&layerShapes[i]), sizeof(int));
        }

        
        // Write the weights for this bot
        for (int i = 0; i < totalWeights; i++)
        {
            float weight = weights[i];
            outfile.write(reinterpret_cast<const char *>(&weight), sizeof(float));
        }

        // Write the biases for this bot
        int biasOffset = 0 * totalNeurons;
        for (int i = 0; i < totalNeurons; i++)
        {
            float bias = biases[biasOffset + i];
            outfile.write(reinterpret_cast<const char *>(&bias), sizeof(float));
        }
        

        outfile.close();

        delete [] layerShapes;
        delete [] weights;
        delete [] biases;
    }
};

// Utility function to compute binary cross-entropy loss
double binaryCrossEntropy(const MatrixXd& Y, const MatrixXd& Y_pred) {
    double eps = 1e-9; // Small constant to prevent log(0)
    return -(Y.array() * (Y_pred.array() + eps).log() + (1 - Y.array()) * (1 - Y_pred.array() + eps).log()).mean();
}




#endif