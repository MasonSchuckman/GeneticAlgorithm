#ifndef SPECIES_H
#define SPECIES_H

#include "Genome.h"
#include "Specimen.h"
#include <string>
#include <set>

class Species {
private:
  static int currentID;
public:
  const int id = currentID++;
  const int speciationYear;
  Specimen* progenitor;

  std::set<Species*> descendantSpecies;

  Species(Specimen* progenitor, int year);
  void addDescendantSpecies(Species* descendant);

  ~Species();

};

#endif
