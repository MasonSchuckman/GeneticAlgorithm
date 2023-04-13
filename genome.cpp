#include <iostream>

// string type
#include <string>

// std::transform, std::max
#include <algorithm> 

// std::accumulate
#include <numeric>

// std::set
#include <set>

/*
A single agent's brain structure

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

  Genome(int *shape, int shapeLen);
  Genome(int *shape, int shapeLen, float *biases, float *connections);
  Genome(Genome *other);
  ~Genome();

  Genome *mitosis(float percentage, float intensity);
  Genome *meiosis(Genome *parent2);

  std::string shapeString();
  std::string bodyString();
};

/*
deletes all relevant dynamic members of the class
*/
Genome::~Genome() {
  delete shape;
  delete biases;
  delete connections;
}

/*
creates a random Genome with a specific shape
*/
Genome::Genome(int *shape, int shapeLen) {

  // copying shape data
  this->shape = new int[shapeLen];
  this->shapeLen = shapeLen;

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

/*
create a genome with a specific shape and specific weights
*/
Genome::Genome(int *shape, int shapeLen, float *biases, float *connections) {

  // copying shape data
  this->shape = new int[shapeLen];
  this->shapeLen = shapeLen;

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

/*
copy constructor which creates a Genome object from an existing Genome
*/
Genome::Genome(Genome *other) {

  // copying slengths
  this->shapeLen = other->shapeLen;
  this->numNeurons = other->numNeurons;
  this->numConnections = other->numConnections;


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

/*
produce a child genome based off the parent but with slight mutations
*/
Genome *Genome::mitosis(float percentage, float intensity) {
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
  // weight = weight +- weight*(-intensity..intensity)
  for(auto i = neuronIndexes.begin(); i != neuronIndexes.end(); i++) {
    std::cout << "mutating neuron #" << *i << std::endl;
    float originalWeight = child->biases[*i];
    float newWeight = originalWeight + (originalWeight * intensity * 2 * ((float)rand()/RAND_MAX - 0.5));
    child->biases[*i] = newWeight;
  }

  for(auto i = connectionIndexes.begin(); i != connectionIndexes.end(); i++) {
    std::cout << "mutating connection #" << *i << std::endl;
    float originalWeight = child->connections[*i];
    float newWeight = originalWeight + (originalWeight * intensity * 2 * ((float)rand()/RAND_MAX - 0.5));
    child->connections[*i] = newWeight;
  }

  return child;
}

/*
breed with a second parent to produce a child genome,
which inherits traits from both parents
*/
Genome *Genome::meiosis(Genome *parent2) {
  Genome *child = new Genome(this);

  // TODO merge however

  return child;
}

/*
returns a string representing the shape of the genome's corresponding Neural Network
*/
std::string Genome::shapeString() {

  std::string toReturn = "";

  for(int i = 0; i < shapeLen; i++)
    toReturn += std::to_string(shape[i]) + ",";
  
  toReturn = "(" + toReturn.substr(0,toReturn.length()-1) + ")";

  return toReturn;
}

/*
returns a string containing all of the connection and bias weights belonging to the genome
separated + labeled with layer and connection names
*/
std::string Genome::bodyString() {

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

int main() {

   int shape[] = {2,2};
   int len = 2;
  
  Genome* root = new Genome(shape,len);

   for(int i = 0; i < 5; i++) {
    std::cout << "epoch " << i << std::endl;
    std::cout << root->bodyString() << std::endl;
    std::cout << root->shapeString() << std::endl;

    Genome* child = root->mitosis(0.5,1);
    delete root;
    root = child;
   }
}
