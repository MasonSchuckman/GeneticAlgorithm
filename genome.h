#ifndef GENOME_H
#define GENOME_H
#include <iostream>

// string type
#include <string>

// std::transform, std::max, std::min
#include <algorithm> 

// std::set
#include <set>

/**
A single Bot's brain structure

a genome represents the weights of every neuron and bias in a
dense neural network, as well as the shape of the network itself

genomes offer several utility functions, including:
  - mitosis(), allowing a genome to spawn a child with slight mutations
  - meiosis(Genome*), allowing a genome to reproduce with a parent to spawn a child with both their traits
  - shapeString(), yielding a string representing the shape of the neural network
  - bodyString(), yielding a string with the contents of every weight of every connection / bias of the neural network
*/
class Genome {
public:
  int *shape;
  int shapeLen;

  float *biases;
  int numNeurons;

  float *connections;
  int numConnections;

  Genome(int *shape);
  Genome(int *shape, float *biases, float *connections);
  Genome(Genome *other);
  ~Genome();

  Genome *mitosis(float percentage, float staticStepSize, float dynamicStepSize);
  Genome *meiosis(Genome *parent2);

  std::string shapeString();
  std::string bodyString();
};

#endif