#include "Genome.h"
#include "Species.h"
#include "Specimen.h"
#include "Taxonomy.h"

#include <cmath>

using std::cout;
using std::endl;

Specimen** generatePopulation(int count, int *shape, int shapeLen);
void mutatingDistance(Genome* toMutate, int epochs);
std::string prettyDistribution(std::map<Species*, float>* composition);

int main() {

    int SHAPE[] = {2,2,2};
    int SHAPE_LENGTH = sizeof(SHAPE) / sizeof(int);
    int COUNT = 25;
    
    Specimen** genesisPopulation = generatePopulation(COUNT, SHAPE, SHAPE_LENGTH);

    Taxonomy history(genesisPopulation, COUNT);

    auto composition = history.speciesComposition();
    cout << prettyDistribution(composition) << endl;

}

std::string prettyDistribution(std::map<Species*, float>* composition) {

    int DECIMAL_DIGITS_PRECISION = 2;
    int OFFSET_FACTOR = pow(10, DECIMAL_DIGITS_PRECISION);

    std::string toReturn = "distribution: ";
    for (auto i = composition->begin(); i != composition->end(); i++) {
        
        int integralPercentage = i->second*(100.0 * OFFSET_FACTOR);
        toReturn += std::to_string(integralPercentage / OFFSET_FACTOR) + "." + std::to_string(integralPercentage % OFFSET_FACTOR) + "%, ";
    }
    return toReturn;
}


void mutatingDistance(Genome* toMutate, int epochs) {
    Genome* child = new Genome(toMutate);
    cout << "original genome: " << endl << toMutate->bodyString() << endl;

    for(int i = 1; i <= epochs; i++) {
        cout << endl;
        child = child->mitosis(0.5,0.1,0.03);
        cout << "epoch " << i << ":" << endl;
        cout << "genome body: " << endl << child->bodyString() << endl;
        cout << "distance: " << Genome::distance(toMutate, child) << endl;

    }
}

Specimen** generatePopulation(int count, int *shape, int shapeLen) {

    Specimen** population = new Specimen*[count];

    for(int i = 0; i < count; i++) {
        Genome* nextGenome = new Genome(shape, shapeLen);
        population[i] = new Specimen(nextGenome, (Specimen*) nullptr);
    }
    
    return population;
}