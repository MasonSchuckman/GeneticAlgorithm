#include "Species.h"

int Species::currentID = 0;

Species::Species(Specimen* progenitor, int year) : progenitor(progenitor), speciationYear(year) {

}

void Species::addDescendantSpecies(Species* descendant) {
    if(descendant->progenitor->parent->species != this)
        throw std::invalid_argument( "argument species is not a direct descendant of the caller!");
    
    this->descendantSpecies.insert(descendant);
}
