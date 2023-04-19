#include "Taxonomy.h"
#include <vector>

Taxonomy::Taxonomy(Genome **genomes, int genomeCount, float threshold) {

  // initialize distances matrix
  float **distances = new float *[genomeCount];
  for (int i = 0; i < genomeCount; i++)
    distances[i] = new float[genomeCount];

  // calculate euclidean distance between each pair of genomes
  for (int i = 0; i < genomeCount; i++)
    for (int j = 0; j < genomeCount; j++) {
      distances[i][j] = distance(genomes[i], genomes[j]);
      std::cout << "dist " << i << "," << j << ": " << distances[i][j] << std::endl;
    }
}


Taxonomy::Taxonomy(Taxonomy *previous, Genome **genomes, int genomeCount) {

}

// gets the euclidean distance, treating the two genomes as points
// in n-dimensional space (where n = numNeurons + numConnections)
float Taxonomy::distance(Genome *first, Genome *second) {

  // incompatible neural network shapes
  if (first->shapeLen != second->shapeLen)
    return -1;

  // incompatible neural network shapes cont.
  for (int i = 0; i < first->shapeLen; i++)
    if (first->shape[i] != second->shape[i])
      return -1;

  // diff = (a2-a1)^2 + (b2-b1)^2 + ... + (z2-z1)^2
  float diff = 0;
  for (int neuron = 0; neuron < first->numNeurons; neuron++) 
    diff += std::powf(first->biases[neuron] - second->biases[neuron], 2);

  for (int neuron = 0; neuron < first->numConnections; neuron++) 
    diff += std::powf(first->connections[neuron] - second->connections[neuron], 2);


  return std::sqrtf(diff);
}