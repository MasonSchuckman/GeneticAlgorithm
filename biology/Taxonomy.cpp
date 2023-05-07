#include "Taxonomy.h"

Taxonomy::Taxonomy(Specimen **genesisGeneration, int genesisCount) {

    this->year = 0;

    // creating a new species for each new specimen
    for(int i = 0; i < genesisCount; i++) {
        Specimen* nextSpecimen = genesisGeneration[i];
        makeProgenitor(nextSpecimen);
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

void Taxonomy::pruneSpecimen(Specimen* toPrune) {
    if(toPrune->species->progenitor != toPrune)
        delete toPrune;
}


Species* Taxonomy::makeProgenitor(Specimen* specimen) {
    Species* newSpecies = new Species(specimen, this->year);
    specimen->species = newSpecies;

    if(specimen->parent != nullptr) 
        specimen->parent->species->addDescendantSpecies(newSpecies);
    
    return newSpecies;
}

Species* inheritSpecies(Specimen* specimen, Species* species) {
    specimen->species = species;
    return species;
}

Species* Taxonomy::assignSpecies(Specimen* specimen, float progenitorThreshold) {

    if(specimen->parent == nullptr)
        return makeProgenitor(specimen);

    Species* parentSpecies = specimen->parent->species;

    if(Specimen::distance(parentSpecies->progenitor, specimen) <= progenitorThreshold)
        return inheritSpecies(specimen, parentSpecies);
    
    for(Species* existingProgenitor : parentSpecies->descendantSpecies) 
        if(Specimen::distance(existingProgenitor->progenitor, specimen) <= progenitorThreshold)
            return inheritSpecies(specimen, existingProgenitor);

    Species* newSpecies = makeProgenitor(specimen);
    return newSpecies;
}


void Taxonomy::incrementGeneration(Specimen **nextGeneration, int generationCount, float progenitorThreshold) {
    
    this->year++;
    this->generation.clear();
    
    // copying specimen to current generation
    for(int i = 0; i < generationCount; i++) {
        Specimen* nextSpecimen = nextGeneration[i];
        assignSpecies(nextSpecimen, progenitorThreshold);
        this->generation.push_back(nextSpecimen);
    }
}

bool compareComposition(const Composition left, const Composition right) {
    return std::get<1>(left) > std::get<1>(right);
}

CompositionGradient Taxonomy::speciesComposition() {

    std::map<Species*, float> composition;

    float percentPerSpecimen = 1.0 / this->generation.size();

    for(Specimen* specimen : this->generation) {
        Species* species = specimen->species;

        float density = 0;
        if(composition.find(species) != composition.end()) 
            density = composition.at(species);
        
        composition.insert_or_assign(species, density + percentPerSpecimen);
    }

    auto toReturn = new std::vector<std::tuple<Species*, float>>();
    for (auto i = composition.begin(); i != composition.end(); i++) 
        toReturn->push_back(std::make_tuple(i->first, i->second));
    
    std::sort(toReturn->begin(), toReturn->end(), compareComposition);
    return toReturn;
}

std::string Taxonomy::compositionGraph(const CompositionGradient composition, const int charsWide) {


    std::string toReturn = "///|"; 
    for(int i = 1; i <= charsWide; i++)
            toReturn += "-";
    toReturn += "|\n";

    int index = 0;
    float totalPercentage = 0;
    for (auto i : *composition) {

        std::string speciesID = std::to_string(std::get<0>(i)->id);
        int digits = speciesID.length();

        toReturn += ("     " + speciesID).substr(digits+2,digits+5);
        toReturn += "|";

        totalPercentage += std::get<1>(i);
        int charsNeeded = std::get<1>(i)*(charsWide * 1.0);

        for(int tickCount = 0; tickCount < charsNeeded; tickCount++) 
            toReturn += "*";
        toReturn += "\n";
    }
    toReturn += "///|";
    for(int i = 1; i <= charsWide; i++) {
            toReturn += (i % 5 == 0 ? "^" : "-");
    }
    toReturn += "|\n";
    toReturn += "     ^ = " + std::to_string(500.0/charsWide) + "%";
    toReturn += "    showing ~" + std::to_string((int)(totalPercentage*100)) + "% of popl.";


    return toReturn;
}

std::string Taxonomy::compositionString(const CompositionGradient composition) {

    int DECIMAL_DIGITS_PRECISION = 1;
    int OFFSET_FACTOR = pow(10, DECIMAL_DIGITS_PRECISION);

    std::string toReturn = "distribution: ";
    for (auto i : *composition) {
        int integralPercentage = std::get<1>(i)*(100.0 * OFFSET_FACTOR);
        toReturn += std::to_string(std::get<0>(i)->id) + ": " + std::to_string(integralPercentage / OFFSET_FACTOR) + "." + std::to_string(integralPercentage % OFFSET_FACTOR) + "%, ";
    }
    return toReturn;
}