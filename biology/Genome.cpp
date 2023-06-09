#include "Genome.h"
/**
 deletes all relevant dynamic members of the class
*/
Genome::~Genome() {
  delete shape;
  delete biases;
  delete connections;
}

/**
creates a Genome with a specific shape and random weights

 @param shape  an array describing how many neurons are in each layer of the neural network
 @param shapeLen  the length of the shape parameter
*/
Genome::Genome(int *shape, int shapeLen) {

  this->activation = "sigmoid";
  
  // copying shape data
  this->shapeLen = shapeLen;
  this->shape = new int[shapeLen];

  for (int i = 0; i < shapeLen; i++) 
    this->shape[i] = shape[i];


  // calculating array sizes for weights and biases
  this->numNeurons = 0;
  for (int i = 0; i < shapeLen; i++)
    this->numNeurons += shape[i];

  this->numConnections = 0;
  for (int i = 0; i < shapeLen-1; i++)
    this->numConnections += shape[i] * shape[i + 1];


  // populating arrays with random values from -1.5 to 1.5
  this->biases = new float[numNeurons];
  for(int i = 0; i < numNeurons; i++)
    biases[i] = ((float)rand()/RAND_MAX) * 3 - 1.5;

  this->connections = new float[numConnections];
  for(int i = 0; i < numConnections; i++)
    connections[i] = ((float)rand()/RAND_MAX) * 3 - 1.5;
}


/**
 create a genome with a specific shape and specific weights

 @param shape  an array describing how many neurons are in each layer of the neural network
 @param shapeLen  the length of the shape parameter
 @param biases  an array containing all bias weights
 @param connections  an array containing all connection weights 
*/
Genome::Genome(int *shape, int shapeLen, float *biases, float *connections, std::string activation) : activation(activation) {

  // copying shape data
  this->shapeLen = shapeLen;
  this->shape = new int[shapeLen];

  for (int i = 0; i < shapeLen; i++) 
    this->shape[i] = shape[i];


  // calculating array sizes for weights and biases
  this->numNeurons = 0;
  for (int i = 0; i < shapeLen; i++)
    this->numNeurons += shape[i];

  this->numConnections = 0;
  for (int i = 0; i < shapeLen-1; i++)
    this->numConnections += shape[i] * shape[i + 1];


  // copying array contents
  this->biases = new float[numNeurons];
  for(int i = 0; i < numNeurons; i++)
    this->biases[i] = biases[i];

  this->connections = new float[numConnections];
  for(int i = 0; i < numConnections; i++)
    this->connections[i] = connections[i];
}

/**
 copy constructor which creates a Genome object from an existing Genome

 @param other  the genome to be copied when making this one
*/
Genome::Genome(const Genome *other) {
  
  // copying lengths
  this->shapeLen = other->shapeLen;
  this->numNeurons = other->numNeurons;
  this->numConnections = other->numConnections;
  this->activation = other->activation;

  // copying array contents
  this->shape = new int[other->shapeLen];
  for (int i = 0; i < other->shapeLen; i++) 
    this->shape[i] = other->shape[i];

  this->biases = new float[numNeurons];
  for(int i = 0; i < numNeurons; i++)
    this->biases[i] = other->biases[i];

  this->connections = new float[numConnections];
  for(int i = 0; i < numConnections; i++)
    this->connections[i] = other->connections[i];
}

/**
 produce a child genome based off the parent but with slight mutations

 @param percentage  percentage of neurons, connections which will be mutated
 @param staticStepSize  a consistent mutation step size (range: -staticStepSize < adjust < staticStepSize)
 @param dynamicStepSize  a mutation step size determined based on the weight (range: -dynamicStepSize*weight < adjust < dynamicStepSize*weight)
*/
Genome *Genome::mitosis(float percentage, float staticStepSize, float dynamicStepSize) const {
  Genome *child = new Genome(this);

  // deciding how many neurons / connections we want to mutate
  int numNeuronsToModify = std::min(numNeurons, std::max(0, (int)(numNeurons * percentage) ));
  int numConnectionsToModify = std::min(numConnections, std::max(0, (int)(numConnections * percentage) ));

  // picking which neurons / connections will be mutated
  std::set<int> neuronIndexes;
  while(neuronIndexes.size() < numNeuronsToModify)
    neuronIndexes.insert(rand() % numNeurons);

  std::set<int> connectionIndexes;
  while(connectionIndexes.size() < numConnectionsToModify)
    connectionIndexes.insert(rand() % numConnections);

  // mutating the selected weights by a certain intensity
  // weight = weight + (-staticStepSize..staticStepSize) + weight*(-dynamicStepSize..dynamicStepSize) 
  for(auto i = neuronIndexes.begin(); i != neuronIndexes.end(); i++) {
    float originalWeight = child->biases[*i];
    float newWeight = originalWeight
      + (staticStepSize * 2 * ((float)rand()/RAND_MAX - 0.5))
      + (originalWeight * dynamicStepSize * 2 * ((float)rand()/RAND_MAX - 0.5));
    child->biases[*i] = newWeight;
  }

  for(auto i = connectionIndexes.begin(); i != connectionIndexes.end(); i++) {
    float originalWeight = child->connections[*i];
    float newWeight = originalWeight
      + (staticStepSize * 2 * ((float)rand()/RAND_MAX - 0.5))
      + (originalWeight * dynamicStepSize * 2 * ((float)rand()/RAND_MAX - 0.5));
    child->connections[*i] = newWeight;
  }

  return child;
}

/**
 breed with a second parent to produce a child genome, which inherits traits from both parents

 @param other  the second parent to be used to produce the child
*/
Genome *Genome::meiosis(const Genome *parent2) const {

  // starting with a perfect copy of the current parent
  Genome *child = new Genome(this);

  // picking which neuron / connection weights will be taken from the second parent 
  std::set<int> neuronIndexes;
  while(neuronIndexes.size() < numNeurons / 2)
    neuronIndexes.insert(rand() % numNeurons);

  std::set<int> connectionIndexes;
  while(connectionIndexes.size() < numConnections / 2)
    connectionIndexes.insert(rand() % numConnections);

  // transferring the chosen weights over to the child
  for(auto i = neuronIndexes.begin(); i != neuronIndexes.end(); i++) 
    child->biases[*i] = parent2->biases[*i];

  for(auto i = connectionIndexes.begin(); i != connectionIndexes.end(); i++) 
    child->connections[*i] = parent2->connections[*i];
  
  return child;
}

/**
 returns a string representing the shape of the genome's corresponding Neural Network
*/
std::string Genome::shapeString() const {
  std::string toReturn = "";

  for(int i = 0; i < shapeLen; i++)
    toReturn += std::to_string(shape[i]) + ",";
  
  toReturn = "(" + toReturn.substr(0,toReturn.length()-1) + ")";
  return toReturn;
}

/** 
 returns a string containing all of the connection and bias weights belonging to the genome
 separated + labeled with layer and connection names
*/
std::string Genome::bodyString() const {
  if(shapeLen == 0)
    return "empty network (no layers)";

  std::string toReturn = "";

  // calculate input layer separately to align bias and connection counts
  toReturn += "layer 0 (input): \t";
  int inputLayerSize = shape[0];

  for(int i = 0; i < inputLayerSize; i++) 
      toReturn += std::to_string(biases[i]) + ", ";
  

  // formatting the rest of the layers like so:

  // layer 1:             x.xxx, x.xxx, ...
  // connections 0->1:    x.xxx, x.xxx, x.xxx, x.xxx, ...
  // layer 2:             x.xxx, x.xxx, ...
  // connections 1->2:    x.xxx, x.xxx, x.xxx, x.xxx ...

  int neuronOffset = inputLayerSize;
  int connectionOffset = 0;

  for(int i = 1; i < shapeLen; i++) {
    int layerSize = shape[i];

    toReturn += "\nlayer " + std::to_string(i) + ": \t\t";
    for(int j = 0; j < layerSize; j++) 
      toReturn += std::to_string(biases[neuronOffset + j]) + ", ";

    neuronOffset += layerSize;
    
    toReturn += "\nconnections " + std::to_string(i-1) + "->" + std::to_string(i) + ": \t";
    for(int j = 0; j < layerSize * shape[i-1]; j++) 
      toReturn += std::to_string(connections[connectionOffset + j]) + ", ";

    connectionOffset += layerSize * shape[i-1];
  }
  return toReturn;
}

// gets the euclidean distance, treating the two genomes as points
// in n-dimensional space (where n = numNeurons + numConnections)
float Genome::distance(const Genome *first, const Genome *second) {

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
    diff += std::pow(first->biases[neuron] - second->biases[neuron], 2);

  for (int neuron = 0; neuron < first->numConnections; neuron++) 
    diff += std::pow(first->connections[neuron] - second->connections[neuron], 2);


  return std::sqrt(diff);
}

Genome::Genome(std::string filename) {

  std::ifstream geneFile(filename);

  if (!geneFile.is_open()) 
    throw std::invalid_argument("unable to import genome weights from "+filename);

  std::string garbage;

  // reading in layer size
  geneFile >> garbage >> this->shapeLen;

  // reading in activation func
  geneFile >> garbage >> this->activation;


  // reading in shape
  geneFile >> garbage;
  this->shape = new int[this->shapeLen];
  for(int i = 0; i < this->shapeLen; i++) 
    geneFile >> this->shape[i];


  // calculating array sizes for weights and biases
  this->numConnections = 0;
  for (int i = 0; i < this->shapeLen-1; i++)
    this->numConnections += this->shape[i] * this->shape[i + 1];
  
  this->numNeurons = 0;
  for (int i = 0; i < this->shapeLen; i++)
    this->numNeurons += this->shape[i];


  // reading in connection weights
  geneFile >> garbage;  
  this->connections = new float[numConnections];
  for(int i = 0; i < this->numConnections; i++)
    geneFile >> this->connections[i];


  // reading in bias weights
  geneFile >> garbage;
  this->biases = new float[numNeurons];
  for(int i = 0; i < this->numNeurons; i++) 
    geneFile >> this->biases[i];
  
  geneFile.close();
}


void Genome::exportWeights(std::string filename) const {

  std::ofstream geneFile(filename);

  if (!geneFile.is_open()) 
    throw std::invalid_argument("unable to export genome weights to "+filename);

  geneFile << "layers " << this->shapeLen << std::endl;

  geneFile << "type " << this->activation << std::endl;
  
  geneFile << "shape";
  for(int i = 0; i < this->shapeLen; i++)
    geneFile <<  " " << this->shape[i];
  geneFile << std::endl;

  geneFile << "connections";
  for(int i = 0; i < this->numConnections; i++)
    geneFile <<  " " << this->connections[i];
  geneFile << std::endl;

  geneFile << "biases";
  for(int i = 0; i < this->numNeurons; i++)
    geneFile <<  " " << this->biases[i];
  geneFile << std::endl;

  geneFile.close();
}