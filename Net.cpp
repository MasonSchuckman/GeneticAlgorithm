#include "Net.h"



int debugging = 0;
int print_iterval = 500;
int CURRENT_ITERATION = 0;

bool isPrintIteration()
{
    return CURRENT_ITERATION % print_iterval == 0;
}

template <typename MatrixType>
void printDims(const MatrixType &matrix)
{
    std::cout << "Rows: " << matrix.rows() << ", Columns: " << matrix.cols() << std::endl;
}

template <typename MatrixType>
void printDims(const std::string matName, const MatrixType &matrix)
{
    std::cout << matName << "\n";
    std::cout << "Rows: " << matrix.rows() << ", Columns: " << matrix.cols() << std::endl;
}

// Define some activation functions

double linear(double x)
{
    return x;
}

double linearDerivative(double x)
{
    return 1;
}

// Define the ReLU activation function and its derivative
double relu(double x)
{
    return x > 0 ? x : 0; // If x is greater than 0, return x, otherwise return 0
}

double reluDerivative(double x)
{
    return x > 0 ? 1 : 0; // If x is greater than 0, return 1, otherwise return 0
}

double LeakyRelu(double x)
{
    return x > 0 ? x : x / 8.0;
}

double LeakyReluDerivative(double x)
{
    return x > 0 ? 1 : 1.0 / 8.0;
}

// Sigmoid and its derivative
double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x)
{
    double s = sigmoid(x);
    return s * (1 - s);
}

// Tanh and its derivative
double tanh_(double x)
{
    return std::tanh(x);
}

double tanhDerivative_(double x)
{
    double t = tanh_(x);
    return 1.0 - t * t;
}

// Softmax function (useful for multi-class classification)
MatrixXd softmax(const MatrixXd &x)
{
    MatrixXd expX = x.unaryExpr([](double v)
                                { return exp(v); });
    return expX.array().colwise() / expX.rowwise().sum().array();
}

MatrixXd heInitialization(int rows, int cols)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0, sqrt(2.0 / cols)); // Using He-et-al initialization
    MatrixXd m = MatrixXd::Zero(rows, cols);
    for (int i = 0; i < m.rows(); ++i)
        for (int j = 0; j < m.cols(); ++j)
            m(i, j) = distribution(generator);
    return m;
}

AdamOptimizer::AdamOptimizer(double lr, double b1, double b2, double eps, 
                  int decay_steps, double decay_rate)
        : initialLearningRate(lr), learningRate(lr), beta1(b1), beta2(b2), epsilon(eps),
          decaySteps(decay_steps), decayRate(decay_rate) {}


void AdamOptimizer::updateLearningRate(int timestep) {
        if (timestep % decaySteps == 0) {
            learningRate = initialLearningRate * pow(decayRate, timestep / decaySteps);
        }
    }

void AdamOptimizer::update(MatrixXd& params, const MatrixXd& dParams, MatrixXd& m, MatrixXd& v, int t) {
        m = beta1 * m + (1 - beta1) * dParams;
        v = beta2 * v + (1 - beta2) * dParams.array().square().matrix();

        MatrixXd mHat = m / (1.0 - pow(beta1, t));
        MatrixXd vHat = v / (1.0 - pow(beta2, t));

        //std::cout << "initial value :\n" << params.array() << "\nUpdate :\n" << learningRate * mHat.array() / (vHat.array().sqrt() + epsilon) << std::endl;
        params.array() -= learningRate * mHat.array() / (vHat.array().sqrt() + epsilon);
        //std::cout << "post value    :\n" << params.array() << std::endl;

    }

 DenseLayer::DenseLayer(int n_inputs, int n_neurons, ActivationFunction act, ActivationFunction actDeriv)
    : weights(heInitialization(n_neurons, n_inputs)),
    biases(MatrixXd::Zero(n_neurons, 1)),
    activation(act),
    activationDerivative(actDeriv),
    mWeights(MatrixXd::Zero(n_neurons, n_inputs)),
    vWeights(MatrixXd::Zero(n_neurons, n_inputs)),
    mBiases(MatrixXd::Zero(n_neurons, 1)),
    vBiases(MatrixXd::Zero(n_neurons, 1)) {
    numNeurons = n_neurons;
    numWeights = n_neurons * n_inputs;
}

MatrixXd DenseLayer::forward(const MatrixXd &inputs, bool isTraining)
{
    Layer::input = inputs;

    // Compute the matrix product
    MatrixXd z = weights * inputs;
    // Broadcast the bias addition across columns
    unactivated_output = z.colwise() + biases.col(0);

    output = unactivated_output.unaryExpr(activation); // Apply activation function

    // Print layer information
    if (debugging >= 3 && isPrintIteration())
    {
        std::cout << "Layer Information:" << std::endl;
        std::cout << "Inputs:" << std::endl
                  << inputs << std::endl;
        std::cout << "Weights:" << std::endl
                  << weights << std::endl;
        std::cout << "Biases:" << std::endl
                  << biases << std::endl;
        std::cout << "Activations:" << std::endl
                  << output << std::endl;
    }
    return output;
}

MatrixXd DenseLayer::backward(const MatrixXd& error) {
        // Apply derivative of activation function to the error
        MatrixXd output_delta = error.array() * unactivated_output.unaryExpr(activationDerivative).array();

        // Update dWeights and dBias using the outer product of output_delta and input
        dWeights = output_delta * input.transpose();
        dBias = output_delta.rowwise().sum();

        // Propagate the error backwards
        dInput = weights.transpose() * output_delta;

        return dInput;
}

DropoutLayer::DropoutLayer(double rate) : dropoutRate(rate)
{
    generator.seed(static_cast<unsigned int>(time(0))); // Seed for randomness
    numNeurons = 0;
}

MatrixXd DropoutLayer::forward(const MatrixXd &inputs, bool isTraining)
{
    Layer::input = inputs;

    if (isTraining)
    {
        std::bernoulli_distribution distribution(1 - dropoutRate);
        mask = MatrixXd(inputs.rows(), inputs.cols());
        for (int i = 0; i < mask.rows(); ++i)
        {
            for (int j = 0; j < mask.cols(); ++j)
            {
                mask(i, j) = distribution(generator) ? 1.0 : 0.0;
            }
        }
        mask = mask / (1.0 - dropoutRate); // Scale the activations to not reduce the expected sum
        output = inputs.cwiseProduct(mask);
    }
    else
    {
        output = inputs; // Do not apply dropout during testing
    }

    return output;
}

MatrixXd DropoutLayer::backward(const MatrixXd &gradOutput)
{
    dInput = gradOutput.cwiseProduct(mask); // Only backprop through the non-dropped units

    return dInput;
}

NeuralNetwork::NeuralNetwork(double lr, double b1, double b2, double eps)
    : optimizer(lr, b1, b2, eps) {

    timestep = 2;
}


MatrixXd NeuralNetwork::forward(const MatrixXd &inputs, bool isTraining)
{
    MatrixXd currentOutput = inputs;
    for (auto &layer : layers)
    {
        currentOutput = layer->forward(currentOutput, isTraining);
    }
    return currentOutput;
}

void NeuralNetwork::backward(const MatrixXd& gradOutput) {
        MatrixXd currentGradient = gradOutput;

        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {            
            //printf("back1\n");
            currentGradient = (*it)->backward(currentGradient);
            //printf("back2\n");
            if(debugging && isPrintIteration())
                std::cout << "Current gradient: " << std::endl << currentGradient << std::endl;

            // If it's a DenseLayer, update its weights and biases
            if (auto denseLayer = dynamic_cast<DenseLayer*>(it->get())) {
                // Update the weights and biases using Adam optimizer
                //printf("update1\n");
                optimizer.update(denseLayer->weights, denseLayer->dWeights, denseLayer->mWeights, denseLayer->vWeights, timestep);
                //printf("update2\n");
                optimizer.update(denseLayer->biases, denseLayer->dBias, denseLayer->mBiases, denseLayer->vBiases, timestep);
                //printf("update3\n");
            }        
        } 
        //timestep++;
    }

    void NeuralNetwork::updateParameters() {
        for (auto& layer : layers) {
            // Check if layer is a DenseLayer and update its parameters
            if (auto denseLayer = dynamic_cast<DenseLayer*>(layer.get())) {
                optimizer.update(denseLayer->weights, denseLayer->dWeights, denseLayer->mWeights, denseLayer->vWeights, timestep);
                optimizer.update(denseLayer->biases, denseLayer->dBias, denseLayer->mBiases, denseLayer->vBiases, timestep);
            }
        }
        //timestep++;
    }
void NeuralNetwork::writeWeightsAndBiases()
{
    int TOTAL_BOTS = 1;
    int numLayers = layers.size() + 1;
    int *layerShapes = new int[numLayers];
    int totalWeights = 0;
    int totalNeurons = 0;

    // Account for input layer
    layerShapes[0] = layers[0]->input.size();
    totalNeurons += layerShapes[0];

    for (int i = 1; i < numLayers; i++)
    {
        layerShapes[i] = layers[i - 1]->numNeurons;

        totalNeurons += layers[i - 1]->numNeurons;
        totalWeights += layers[i - 1]->numWeights;
    }

    float *weights = new float[totalWeights];
    int c = 0;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it)
    {
        if (auto denseLayer = dynamic_cast<DenseLayer *>((it->get())))
        {
            for (int j = 0; j < denseLayer->numWeights; j++)
            {
                weights[c] = denseLayer->weights.transpose().data()[c];
                c++;
            }
        }
    }

    float *biases = new float[totalNeurons];
    for (int i = 0; i < layerShapes[0]; i++)
        biases[i] = 0;

    c = layerShapes[0];
    for (auto it = layers.rbegin(); it != layers.rend(); ++it)
    {
        if (auto denseLayer = dynamic_cast<DenseLayer *>((it->get())))
        {
            for (int j = 0; j < denseLayer->numNeurons; j++)
            {
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

    delete[] layerShapes;
    delete[] weights;
    delete[] biases;
}

double binaryCrossEntropy(const MatrixXd &Y, const MatrixXd &Y_pred)
{
    double eps = 1e-9; // Small constant to prevent log(0)
    return -(Y.array() * (Y_pred.array() + eps).log() + (1 - Y.array()) * (1 - Y_pred.array() + eps).log()).mean();
}

