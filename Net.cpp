#include "Net.h"



int debugging = 0;
int print_iterval = 1000;
int CURRENT_ITERATION = 0;

const float L2_LAMBDA = 0.0;

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
    std::cout << matName << "   ";
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
    return x > 0 ? x : x / 32.0;
}

double LeakyReluDerivative(double x)
{
    return x > 0 ? 1 : 1.0 / 32.0;
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
    int c = 1;
    for (int i = 0; i < m.cols(); ++i)
        for (int j = 0; j < m.rows(); ++j) {
            m(j, i) =  distribution(generator);
            c++;
        }
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
        //params.array() -= learningRate * mHat.array() / (vHat.array().sqrt() + epsilon);
        //std::cout << "post value    :\n" << params.array() << std::endl;

        
        MatrixXd adjustedGradient = mHat.array() / (vHat.array().sqrt() + epsilon);
        //if(isPrintIteration())
       // std::cout << "Is small ? " << mHat.array() << "   /   " << (vHat.array().sqrt() + epsilon) << std::endl;
        // Update parameters
        //if(isPrintIteration())
        //std::cout << "Max coef : " << adjustedGradient.maxCoeff() << ", total delta : " << (learningRate * adjustedGradient.maxCoeff()) << std::endl;
        //printf("Max coef : %f, prior max coef : %f, total delta : %f\n ", adjustedGradient.maxCoeff(), params.maxCoeff(), (learningRate * adjustedGradient.maxCoeff()));
        
        params.array() -= learningRate * adjustedGradient.array();

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


 MatrixXd DenseLayer::forward(const MatrixXd& inputs, bool isTraining) {
     Layer::input = inputs;

     // Compute the matrix product
     //printDims("Weights", weights);
     //printDims("inputs", inputs);
     MatrixXd z = weights * inputs;
     //           24 x 4    4 x 1
     // Broadcast the bias addition across columns
     unactivated_output = z.colwise() + biases.col(0);

     output = unactivated_output.unaryExpr(activation); // Apply activation function

     // Print layer information
     if (debugging >= 2 && isPrintIteration()) {
         std::cout << "Layer Information:" << std::endl;
         std::cout << "Inputs:" << std::endl << inputs << std::endl;
         std::cout << "Weights:" << std::endl << weights << std::endl;
         std::cout << "Biases:" << std::endl << biases << std::endl;
         std::cout << "Activations:" << std::endl << output << std::endl;
     }
     return output;
 }
//
//MatrixXd DenseLayer::forward(const MatrixXd &inputs, bool isTraining)
//{
//    Layer::input = inputs;
//
//    // Compute the matrix product
//    unactivated_output = (inputs * weights.transpose()).rowwise() + biases.col(0).transpose();
//    
//    
//    output = unactivated_output.unaryExpr(activation); // Apply activation function
//    std::cout << "Activations:" << std::endl
//        << output << std::endl;
//
//
//    // Print layer information
//    if (debugging >= 3 && isPrintIteration())
//    {
//        std::cout << "Layer Information:" << std::endl;
//        std::cout << "Inputs:" << std::endl
//                  << inputs << std::endl;
//        std::cout << "Weights:" << std::endl
//                  << weights << std::endl;
//        std::cout << "Biases:" << std::endl
//                  << biases << std::endl;
//        std::cout << "Activations:" << std::endl
//                  << output << std::endl;
//    }
//    return output;
//}

 //MatrixXd DenseLayer::backward(const MatrixXd& error) {
 //    // Apply derivative of activation function to the error
 //    MatrixXd output_delta = error.array() * unactivated_output.unaryExpr(activationDerivative).array();

 //    // Update dWeights and dBias using the outer product of output_delta and input
 //    dWeights = output_delta * input.transpose();

 //    // Add L2 regularization to the gradient for the weights
 //    dWeights += L2_LAMBDA * weights;

 //    dBias = output_delta.rowwise().sum();

 //    // Propagate the error backwards
 //    dInput = weights.transpose() * output_delta;

 //    return dInput;
 //}
MatrixXd DenseLayer::backward(const MatrixXd& error) {

    MatrixXd output_delta = error.array() * unactivated_output.unaryExpr(activationDerivative).array();
    dWeights = output_delta * input.transpose();
    dWeights += L2_LAMBDA * weights;
    dBias = output_delta.rowwise().sum();

    return output_delta;
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


//BatchNormalizationLayer::BatchNormalizationLayer(int n_inputs)
//    : gamma(MatrixXd::Ones(n_inputs, 1)),
//    beta(MatrixXd::Zero(n_inputs, 1)),
//    runningMean(MatrixXd::Zero(n_inputs, 1)),
//    runningVariance(MatrixXd::Zero(n_inputs, 1)),
//    momentum(0.9) {}

//MatrixXd BatchNormalizationLayer::forward(const MatrixXd& inputs, bool isTraining) {
//    if (isTraining) {
//        // Compute mean and variance for the batch
//        batchMean = inputs.colwise().mean();
//        MatrixXd meanCentered = inputs - batchMean.transpose().replicate(inputs.rows(), 1);
//        batchVariance = ((meanCentered.array().square().colwise().sum()) / inputs.rows()).matrix();
//
//        // Update running mean and variance
//        runningMean = momentum * runningMean + (1 - momentum) * batchMean;
//        runningVariance = momentum * runningVariance + (1 - momentum) * batchVariance;
//
//        // Normalize
//        output = (meanCentered.array() / (batchVariance.array() + 1e-8).sqrt().transpose().replicate(1, inputs.cols())).matrix();
//    }
//    else {
//       // // Use running mean and variance for normalization during inference
//        MatrixXd meanCentered = inputs - runningMean.transpose().replicate(1, inputs.cols());
//        output = (meanCentered.array() / (runningVariance.array() + 1e-8).sqrt().transpose().replicate(1, inputs.cols())).matrix();
//    }
//
//    // Apply scale and shift
//    output.array() *= gamma.transpose().replicate(1, inputs.cols()).array();
//    output.array() += beta.transpose().replicate(1, inputs.cols()).array();
//
//    return output;
//}
//
//
//MatrixXd var(MatrixXd m) {
//    double mean = m.mean();
//    Eigen::MatrixXd variance = (m.array() - mean).matrix().square().mean();
//    return variance;
//}
//
//MatrixXd BatchNormalizationLayer::forward(const MatrixXd& inputs, bool isTraining) {
//    // Calculate the mean and variance of the input data
//    MatrixXd mean = inputs.rowwise().mean();
//    MatrixXd variance = inputs.rowwise().variance();
//
//    // Calculate the scale and shift parameters
//    MatrixXd scale = (1 - momentum) * gamma / sqrt(variance + epsilon);
//    MatrixXd shift = beta - gamma * mean / sqrt(variance + epsilon);
//
//    // Apply the batch normalization transformation
//    MatrixXd normalized = (inputs - mean) / sqrt(variance + epsilon);
//    MatrixXd transformed = scale * normalized + shift;
//
//    // Update the running mean and variance
//    runningMean = momentum * runningMean + (1 - momentum) * mean;
//    runningVariance = momentum * runningVariance + (1 - momentum) * variance;
//
//    return transformed;
//}
//
//MatrixXd BatchNormalizationLayer::backward(const MatrixXd& gradOutput) {
//    //int m = gradOutput.rows();
//
//    // Derivative of loss with respect to scale (gamma) and shift (beta)
//    //MatrixXd dGamma = (gradOutput.array() * output.array()).colwise().sum();
//    //MatrixXd dBeta = gradOutput.colwise().sum();
//
//    //// Intermediate terms for normalization
//    //MatrixXd stdInv = (batchVariance.array() + 1e-8).sqrt().inverse();
//    //MatrixXd xMu = input.rowwise() - batchMean;
//
//    // Derivative of loss with respect to normalized input
//    //MatrixXd dxNorm = gradOutput.array().rowwise() * gamma.transpose().array();
//
//    //// Derivative of loss with respect to variance
//    //MatrixXd dVar = (dxNorm.array() * xMu.array()).colwise().sum() * -0.5 * stdInv.array().cube();
//
//    //// Derivative of loss with respect to mean
//    //MatrixXd dMean = -dxNorm.colwise().sum().array() * stdInv.array() - 2.0 / m * dVar.array() * xMu.colwise().sum().array();
//
//    //// Derivative of loss with respect to input
//    //MatrixXd dInput = dxNorm.array().rowwise() * stdInv.transpose().array() + 2.0 / m * dVar.array() * xMu.array() + dMean.array() / m;
//
//    //// Update scale and shift
//    //gamma -= learningRate * dGamma;
//    //beta -= learningRate * dBeta;
//
//    return gamma;
//}


NeuralNetwork::NeuralNetwork(double lr, double b1, double b2, double eps)
    : optimizer(lr, b1, b2, eps) {
    lambda = 0.01;
    timestep = 1;
}

void NeuralNetwork::polyakUpdate(const NeuralNetwork& primaryNetwork, double polyakCoefficient) {
    for (size_t i = 0; i < layers.size(); ++i) {
        auto& targetLayer = dynamic_cast<DenseLayer&>(*layers[i]);
        const auto& primaryLayer = dynamic_cast<const DenseLayer&>(*primaryNetwork.layers[i]);

        // Update weights and biases with Polyak averaging
        targetLayer.weights = polyakCoefficient * targetLayer.weights +
            (1 - polyakCoefficient) * primaryLayer.weights;
        targetLayer.biases = polyakCoefficient * targetLayer.biases +
            (1 - polyakCoefficient) * primaryLayer.biases;
    }
}

MatrixXd NeuralNetwork::forward(const MatrixXd &inputs, bool isTraining)
{
    int c = 0;
    MatrixXd currentOutput = inputs;
    for (auto &layer : layers)
    {
        //printf("C = %d, ", c);
        //printDims("currentOutput", currentOutput);
        currentOutput = layer->forward(currentOutput, isTraining);
        c++;
    }
    //printf("C = %d, ", c);
    //printDims("currentOutput", currentOutput);
    return currentOutput;
}

//void NeuralNetwork::backward(const MatrixXd& gradOutput) {
//        MatrixXd currentGradient = gradOutput;
//
//        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {            
//            //printf("back1\n");
//
//           /* if (auto normalizer = dynamic_cast<BatchNormalizationLayer*>(it->get()))
//                normalizer->learningRate = optimizer.learningRate;*/
//
//            currentGradient = (*it)->backward(currentGradient);
//            //printf("back2\n");
//            if(debugging && isPrintIteration())
//                std::cout << "Current gradient: " << std::endl << currentGradient << std::endl;
//
//            // If it's a DenseLayer, update its weights and biases
//            if (auto denseLayer = dynamic_cast<DenseLayer*>(it->get())) {
//                // Update the weights and biases using Adam optimizer
//                //printf("update1\n");
//                optimizer.update(denseLayer->weights, denseLayer->dWeights, denseLayer->mWeights, denseLayer->vWeights, timestep);
//                //printf("update2\n");
//                optimizer.update(denseLayer->biases, denseLayer->dBias, denseLayer->mBiases, denseLayer->vBiases, timestep);
//                //printf("update3\n");
//            }        
//        } 
//        //timestep++;
//    }

void NeuralNetwork::backward(const MatrixXd& gradOutput) {
    MatrixXd currentGradient = gradOutput;

    for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
        if (it != layers.rbegin()) {
            if (auto denseLayer = dynamic_cast<DenseLayer*>((it - 1)->get())) {
                currentGradient = denseLayer->weights.transpose() * currentGradient;
            }
        }

        currentGradient = (*it)->backward(currentGradient);

        if (debugging && isPrintIteration())
            std::cout << "Current gradient: " << std::endl << currentGradient << std::endl;

        // If it's a DenseLayer, update its weights and biases
        if (auto denseLayer = dynamic_cast<DenseLayer*>(it->get())) {
            // Update the weights
            optimizer.update(denseLayer->weights, denseLayer->dWeights, denseLayer->mWeights, denseLayer->vWeights, timestep);
            optimizer.update(denseLayer->biases, denseLayer->dBias, denseLayer->mBiases, denseLayer->vBiases, timestep);

            // Update the biases
            //denseLayer->biases -= optimizer.learningRate * denseLayer->dBias;
        }
    }

    timestep++;
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

void NeuralNetwork::printWeightsAndBiases()
{
    for (int i = 0; i < layers.size(); i++) {
        if (auto denseLayer = dynamic_cast<DenseLayer*>((layers[i].get())))
        {
            std::cout << "Layer " << i << " Weights:\n" << denseLayer->weights << std::endl;
            std::cout << "Layer " << i << " Biases:\n" << denseLayer->biases << std::endl;

        }
        printf("\n");
    }
    printf("\n\n");
}
void NeuralNetwork::writeWeightsAndBiases()
{

    printf("\n\nSaving RL-BOT parameters\n\n");

    for (int i = 0; i < layers.size(); i++) {
        if (auto denseLayer = dynamic_cast<DenseLayer*>((layers[i].get())))
        {
            std::cout << "Layer " << i << " Weights:\n" << denseLayer->weights << std::endl;
            std::cout << "Layer " << i << " Biases:\n" << denseLayer->biases << std::endl;

        }
        printf("\n");        
    }
    printf("\n\n");
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
    int layerr = 0;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it)
    {
        //printf("\nLayer %d Weights:\n", layerr);
        if (auto denseLayer = dynamic_cast<DenseLayer *>((it->get())))
        {
            for (int j = 0; j < denseLayer->numWeights; j++)
            {
                
                weights[c] = denseLayer->weights.transpose().data()[c];// printf("%f, ", weights[c]);
                c++;
            }
        }
        layerr++;
    }

    float *biases = new float[totalNeurons];
    for (int i = 0; i < layerShapes[0]; i++)
        biases[i] = 0;

    c = layerShapes[0];
    layerr = 0;
    for (auto it = layers.rbegin(); it != layers.rend(); ++it)
    {
        //printf("\nLayer %d Biases:\n", layerr);
        if (auto denseLayer = dynamic_cast<DenseLayer *>((it->get())))
        {
            for (int j = 0; j < denseLayer->numNeurons; j++)
            {
                
                biases[c] = denseLayer->biases.transpose().data()[c];// printf("%f, ", biases[c]);
                c++;
            }
        }
        layerr++;
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

