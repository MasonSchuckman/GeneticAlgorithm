#ifndef TAXONOMY_H
#define TAXONOMY_H
#include "Genome.h"
#include "Species.h"
#include "Specimen.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <set>
#include <map>

#define Composition std::tuple<Species*, float>
#define CompositionGradient std::vector<Composition>*

class Taxonomy {
private:
  std::vector<Specimen*> generation;
  int year;

  Species* makeProgenitor(Specimen* specimen);

public:
  Taxonomy(Specimen **genesisGeneration, int genesisCount);

  void incrementGeneration(Specimen **nextGeneration, int generationCount, float progenitorThreshold);
  Species* assignSpecies(Specimen* specimen, float progenitorThreshold);
  void pruneSpecimen(Specimen* toPrune);
  int getYear();

  CompositionGradient speciesComposition();
  static std::string compositionString(const CompositionGradient composition);
  static std::string compositionGraph(const CompositionGradient composition, const int charsWide);

  static void writeCompositionsData(const std::vector<CompositionGradient> compositions, std::string filename);
};


#endif
