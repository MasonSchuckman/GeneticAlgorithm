#ifndef TAXONOMY_H
#define TAXONOMY_H
#include "Genome.h"
#include "Species.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <map>

class Taxonomy {
private:
  std::vector<Specimen*> generation;
  int year;

public:
  Taxonomy(Specimen **genesisGeneration, int genesisCount);

  void incrementGeneration(Specimen **nextGeneration, int generationCount, float progenitorThreshold);
  int getYear();
  
  std::map<Species*, float>* speciesComposition();
};


#endif
