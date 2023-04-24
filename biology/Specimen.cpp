#include "Specimen.h"

Specimen::Specimen(Genome* genome, Specimen* parent) : genome(genome), parent(parent) {
    this->species = nullptr;
}

Specimen::~Specimen() {
    delete genome;
}

float Specimen::distance(const Specimen *first, const Specimen *second) {
    return Genome::distance(first->genome, second->genome);
}
