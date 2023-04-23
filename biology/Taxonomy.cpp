#include "Taxonomy.h"


Taxonomy::Taxonomy(Specimen **genesisGeneration, int genesisCount) {

    this->year = 0;

    // creating a new species for each new specimen
    for(int i = 0; i < genesisCount; i++) {
        Specimen* nextSpecimen = genesisGeneration[i];
        Species* newSpecies = new Species(nextSpecimen, year);
        nextSpecimen->species = newSpecies;
    }

    // copying specimen to current generation
    for(int i = 0; i < genesisCount; i++) {
        Specimen* nextSpecimen = genesisGeneration[i];
        this->generation.push_back(nextSpecimen);
    }
}

int Taxonomy::getYear() {
    return year;
}

void Taxonomy::incrementGeneration(Specimen **nextGeneration, int generationCount, float progenitorThreshold) {
    
    this->year++;
    this->generation.clear();
    
    // copying specimen to current generation
    for(int i = 0; i < generationCount; i++) {
        Specimen* nextSpecimen = nextGeneration[i];
        this->generation.push_back(nextSpecimen);
    }
}

std::map<Species*, float>* Taxonomy::speciesComposition() {

    auto composition = new std::map<Species*, float>();

    float percentPerSpecimen = 1.0 / this->generation.size();

    for(Specimen* specimen : this->generation) {
        Species* species = specimen->species;

        float density = 0;
        if(composition->find(species) != composition->end()) 
            density = composition->at(species);
        
        composition->insert_or_assign(species, density + percentPerSpecimen);
    }

    return composition;
}

