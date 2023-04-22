#include "Specimen.h"

Specimen::Specimen(Genome* genome, Specimen* parent) : genome(genome), parent(parent) {
    if(parent != nullptr)
        this->species = parent->species;
}

Specimen::~Specimen() {
    delete genome;
}

