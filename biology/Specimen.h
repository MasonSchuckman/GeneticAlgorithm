#ifndef SPECIMEN_H
#define SPECIMEN_H

#include "Genome.h"
class Species; // can't #include else cyclic dependency hell



class Specimen {

public:
  const Genome* genome;
  const Specimen* parent;
  
  Species* species;

  Specimen(Genome* genome, Specimen* parent);
  ~Specimen();
};

#endif