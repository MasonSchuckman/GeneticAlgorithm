#ifndef TAXONOMY_H
#define TAXONOMY_H
#include "Genome.h"

#include <cmath>
#include <iostream>


/**

*/
class Taxonomy {

public:
  static float distance(Genome *first, Genome *second);
  Taxonomy(Genome **genomes, int genomeCount, float threshold);
  Taxonomy(Taxonomy *previous, Genome **genomes, int genomeCount);
};

#endif
