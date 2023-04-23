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

std::string Taxonomy::compositionGraph(CompositionGradient composition) {

    const int CHARS_TILL_100PERCENT = 40;
    std::string toReturn = "///|"; 
    for(int i = 1; i <= CHARS_TILL_100PERCENT; i++)
            toReturn += "-";
    toReturn += "|\n";

    for (auto i : *composition) {
        std::string speciesID = std::to_string(std::get<0>(i)->id);
        int digits = speciesID.length();

        toReturn += ("     " + speciesID).substr(digits+2,digits+5);
        toReturn += "|";

        int integralPercentage = std::get<1>(i)*(CHARS_TILL_100PERCENT * 1.0);

        for(int tickCount = 0; tickCount < integralPercentage; tickCount++) 
            toReturn += "*";
        toReturn += "\n";
    }
    toReturn += "///|";
    for(int i = 1; i <= CHARS_TILL_100PERCENT; i++) {
            toReturn += (i % 5 == 0 ? "^" : "-");
    }
    toReturn += "|\n";
    toReturn += "     ^ = " + std::to_string(500.0/CHARS_TILL_100PERCENT) + "%";

    return toReturn;
}

std::string Taxonomy::compositionString(CompositionGradient composition) {

    int DECIMAL_DIGITS_PRECISION = 1;
    int OFFSET_FACTOR = pow(10, DECIMAL_DIGITS_PRECISION);

    std::string toReturn = "distribution: ";
    for (auto i : *composition) {
        int integralPercentage = std::get<1>(i)*(100.0 * OFFSET_FACTOR);
        toReturn += std::to_string(std::get<0>(i)->id) + ": " + std::to_string(integralPercentage / OFFSET_FACTOR) + "." + std::to_string(integralPercentage % OFFSET_FACTOR) + "%, ";
    }
    return toReturn;
}