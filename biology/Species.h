#ifndef SPECIES_H
#define SPECIES_H

#include "Genome.h"
#include "Specimen.h"
#include <string>

class Species {

public:
  const int speciationYear;
  const Specimen* progenitor;

  Species(Specimen* progenitor, int year);
  ~Species();

};

#endif