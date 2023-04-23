#include "Genome.h"
#include "Species.h"
#include "Specimen.h"
#include "Taxonomy.h"

#include <cmath>

using std::cout;
using std::endl;

Specimen** generatePopulation(int count, int *shape, int shapeLen);
void mutatingDistance(Genome* toMutate, int epochs);
std::string inlineDistributionString(std::map<Species*, float>* composition);
std::string prettyDistributionString(std::map<Species*, float>* composition);
void stepGeneration(Specimen** population);

int SHAPE[] = {2,2,2};
int SHAPE_LENGTH = sizeof(SHAPE) / sizeof(int);
int COUNT = 300;

Genome* idealNetwork = new Genome(SHAPE, SHAPE_LENGTH);

int main() {
    srand(5);

    Specimen** population = generatePopulation(COUNT, SHAPE, SHAPE_LENGTH);

    
    Taxonomy history(population, COUNT);

    for(int i = 0; i < 10; i++) {
        auto composition = history.speciesComposition();
        cout << endl << endl << endl;
        cout << "generation " << history.getYear() << endl;
        cout << prettyDistributionString(composition) << endl;

        stepGeneration(population);
        history.incrementGeneration(population, COUNT, 0);
    } 

    
}

void stepGeneration(Specimen** population) {

    int randOffset = 2 * (rand() % (COUNT/2));

    Specimen** newPopulation = new Specimen*[COUNT];

    for(int i = 0; i < COUNT; i+= 2) {
        auto first = population[i];
        auto second = population[i+1];

        float firstScore = 1 / (Genome::distance(first->genome, idealNetwork));
        float secondScore = 1 / (Genome::distance(second->genome, idealNetwork));

        auto winner = firstScore > secondScore ? first : second;

        auto offspring1 = new Specimen(winner->genome->mitosis(0.5,0.3,0.1), winner);
        auto offspring2 = new Specimen(winner->genome->mitosis(0.5,0.3,0.1), winner);

        newPopulation[i] = offspring1;
        newPopulation[(i+1 + randOffset) % COUNT] = offspring2;
    }
    for(int i = 0; i < COUNT; i++) {
        //delete population[i]; -- don't want to accidentally delete progenitors!
        population[i] = newPopulation[i];
    }
}

std::string prettyDistributionString(std::map<Species*, float>* composition) {

    const int CHARS_TILL_100PERCENT = 50;
    std::string toReturn = "distribution: \n"; 
    for (auto i = composition->begin(); i != composition->end(); i++) {
        std::string speciesID = std::to_string(i->first->id);
        int digits = speciesID.length();

        toReturn += ("00000" + speciesID).substr(digits+2,digits+5);
        toReturn += "|";

        int integralPercentage = i->second*(CHARS_TILL_100PERCENT * 1.0);
        for(int tickCount = 0; tickCount < integralPercentage; tickCount++) 
            toReturn += "*";
        toReturn += "\n";
    }
    toReturn += "    ";
    for(int i = 1; i <= CHARS_TILL_100PERCENT; i++) {
            toReturn += (i % 5 == 0 ? "^" : "-");
    }
    toReturn += "\n";
    toReturn += "     ^ = " + std::to_string((CHARS_TILL_100PERCENT)/10.0) + "%";

    return toReturn;
}

std::string inlineDistributionString(std::map<Species*, float>* composition) {

    int DECIMAL_DIGITS_PRECISION = 1;
    int OFFSET_FACTOR = pow(10, DECIMAL_DIGITS_PRECISION);

    std::string toReturn = "distribution: ";
    for (auto i = composition->begin(); i != composition->end(); i++) {
        
        int integralPercentage = i->second*(100.0 * OFFSET_FACTOR);
        toReturn += std::to_string(i->first->id) + ": " + std::to_string(integralPercentage / OFFSET_FACTOR) + "." + std::to_string(integralPercentage % OFFSET_FACTOR) + "%, ";
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

Specimen** generatePopulation(int count, int* shape, int shapeLen) {

    Specimen** population = new Specimen*[count];

    for(int i = 0; i < count; i++) {
        Genome* nextGenome = new Genome(shape, shapeLen);
        population[i] = new Specimen(nextGenome, (Specimen*) nullptr);
    }
    
    return population;
}