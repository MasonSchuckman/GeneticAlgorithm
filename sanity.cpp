//#include <iostream>
//#include <Eigen/Dense>
//#include "Net.h"
//
//using namespace Eigen;
//using namespace std;
//#ifndef M_PI
//#define M_PI 3.14159265358979323846
//#endif
//
//
//using Eigen::VectorXd;
//void generateConcentricSpirals(MatrixXd& X, MatrixXd& Y, int samples, double noise = 0.0) {
//    X = MatrixXd(2, samples);
//    Y = MatrixXd(1, samples);
//
//    VectorXd t = VectorXd::LinSpaced(samples, 0, 4 * M_PI);  // Generate angles for samples
//    VectorXd r1 = VectorXd::LinSpaced(samples, 0, 1);  // Radii for the first spiral
//    VectorXd r2 = VectorXd::LinSpaced(samples, 0, 1);  // Radii for the second spiral
//
//    for (int i = 0; i < samples / 2; ++i) {
//        double x1 = r1(i) * cos(t(i)) + (rand() / (RAND_MAX + 1.0) - 0.5) * noise;
//        double y1 = r1(i) * sin(t(i)) + (rand() / (RAND_MAX + 1.0) - 0.5) * noise;
//
//        // Offset the angle for the second spiral
//        double x2 = r2(i) * cos(t(i) + M_PI) + (rand() / (RAND_MAX + 1.0) - 0.5) * noise;  // Offset by pi (180 degrees)
//        double y2 = r2(i) * sin(t(i) + M_PI) + (rand() / (RAND_MAX + 1.0) - 0.5) * noise;
//
//        X(0, i) = x1;
//        X(1, i) = y1;
//        X(0, i + samples / 2) = x2;
//        X(1, i + samples / 2) = y2;
//
//        Y(0, i) = 0;
//        Y(0, i + samples / 2) = 1;
//    }
//}
//
//
//int main() {
//    // Set up the neural network
//    NeuralNetwork net(0.03);
//    net.addLayer(DenseLayer(2, 8, relu, reluDerivative)); // Hidden layer with 3 neurons
//    net.addLayer(DenseLayer(8, 6, relu, reluDerivative));     // Output layer with 2 neurons
//    net.addLayer(DenseLayer(6, 6, relu, reluDerivative));     // Output layer with 2 neurons
//    net.addLayer(DenseLayer(6, 6, relu, reluDerivative));     // Output layer with 2 neurons
//    net.addLayer(DenseLayer(6, 4, relu, reluDerivative));     // Output layer with 2 neurons
//
//    net.addLayer(DenseLayer(4, 1, sigmoid, sigmoidDerivative));     // Output layer with 2 neurons
//
//
//    net.printWeightsAndBiases();
//
//    // Create a simple dataset
//    /*MatrixXd X(2, 5);
//    X << 0.0, 0, 1, 1, 2,
//         0, 1, 0, 1, -1;
//
//    MatrixXd Y(2, 5);
//    Y << 0.0, 1, 1, 0, 5,
//         1, 0, 1, 0, -8;
//         */
//
//    MatrixXd X, Y;
//    int samples = 200;
//    //generateData(X, Y, samples);
//    //generateDifficultData(X, Y, samples);
//    generateConcentricSpirals(X, Y, samples, 0.00);
//   /* printf("X:\n");
//    std::cout << X << std::endl;
//    printf("\nY:\n");
//
//    std::cout << Y << std::endl;*/
//
//    // Train the network
//    for (int i = 0; i < 200; ++i) { // 100 epochs
//        MatrixXd output = net.forward(X);
//        MatrixXd gradOutput = output - Y; // Assuming binary cross-entropy loss
//        //std::cout << "Gradoutput1 : \n" << gradOutput << std::endl;
//        //gradOutput /= X.cols();
//        //std::cout << "Gradoutput2 : \n" << gradOutput << std::endl;
//
//        net.backward(gradOutput / X.cols());
//        //net.updateParameters();
//        net.timestep++;
//        printf("Epoch %d\tLoss : %f\n", i, gradOutput.array().square().sum());
//    }
//
//    //MatrixXd gradOutputOld;
//    //for (int i = 0; i < 1; ++i) { // 100 epochs
//    //    MatrixXd output = net.forward(X);
//    //    MatrixXd gradOutput = output - Y; // Assuming binary cross-entropy loss
//    //   // gradOutput /= X.cols();
//    //    net.backward(gradOutput);
//    //    //net.updateParameters();
//    //    if (i > 1700) 
//    //    {
//    //        std::cout << std::endl << (gradOutput - gradOutputOld).sum() << "  ";
//    //    }
//    //    printf("Epoch %d\tLoss : %f\n", i, gradOutput.array().square().sum() / (Y.cols() * Y.rows()));
//
//    //    gradOutputOld = gradOutput;
//    //}
//    // Print the model's parameters (weights and biases)
//    net.printWeightsAndBiases();
//
//    // Make predictions
//    MatrixXd predictions = net.forward(X);
//    cout << "\nPredictions:\n" << predictions.leftCols(5).transpose() << endl;
//    cout << "\Labels:\n" << Y.leftCols(5).transpose() << endl;
//
//    return 0;
//}
