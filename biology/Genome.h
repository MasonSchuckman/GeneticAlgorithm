#ifndef GENOME_H
#define GENOME_H
#include <iostream>
#include "math.h"
// string type
#include <string>

// std::transform, std::max, std::min
#include <algorithm> 

// std::set
#include <set>
#include <vector>

// file io
#include <fstream>

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

  std::string activation;

  Genome(std::string gene);
  Genome(int *shape, int shapeLen);
  Genome(int *shape, int shapeLen, float *biases, float *connections, std::string activation);
  Genome(const Genome *other);
  ~Genome();

  Genome *mitosis(float percentage, float staticStepSize, float dynamicStepSize) const;
  Genome *meiosis(const Genome *parent2) const;

  std::string shapeString() const;
  std::string bodyString() const;

  void exportWeights(std::string filepath) const;

  static float distance(const Genome *first, const Genome *second);
};

#endif
