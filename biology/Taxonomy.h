#ifndef TAXONOMY_H
#define TAXONOMY_H
#include "Genome.h"
#include "Species.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <map>

#define Composition std::tuple<Species*, float>
#define CompositionGradient std::vector<Composition>*


class Taxonomy {
private:
  std::vector<Specimen*> generation;
  int year;

public:
  Taxonomy(Specimen **genesisGeneration, int genesisCount);

  void incrementGeneration(Specimen **nextGeneration, int generationCount, float progenitorThreshold);
  int getYear();

  std::vector<std::tuple<Species*, float>>* speciesComposition();
  static std::string compositionString(CompositionGradient composition);
  static std::string compositionGraph(CompositionGradient composition);
};


#endif
