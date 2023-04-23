#include "Species.h"

int Species::currentID = 0;

Species::Species(Specimen* progenitor, int year) : progenitor(progenitor), speciationYear(year) {

}
