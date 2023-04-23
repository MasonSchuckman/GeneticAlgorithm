#include "Genome.h"
#include "Species.h"
#include "Specimen.h"
#include "Taxonomy.h"

#include <cmath>

using std::cout;
using std::endl;

Specimen** generatePopulation(int count, int *shape, int shapeLen);
void mutatingDistance(Genome* toMutate, int epochs);
std::string compositionString(std::map<Species*, float>* composition);
std::string compositionGraph(std::map<Species*, float>* composition);
void stepGeneration(Specimen** population);

int SHAPE[] = {2,2,2};
int SHAPE_LENGTH = sizeof(SHAPE) / sizeof(int);
int POPULATION_COUNT = 300;
int EPOCHS = 20;

Genome* idealNetwork = new Genome(SHAPE, SHAPE_LENGTH);

int main() {
    srand(5);

    Specimen** population = generatePopulation(POPULATION_COUNT, SHAPE, SHAPE_LENGTH);

    
    Taxonomy history(population, POPULATION_COUNT);

    for(int i = 0; i < EPOCHS; i++) {
        auto composition = history.speciesComposition();
        cout << endl << endl << endl;
        cout << "generation " << history.getYear()+1 << endl;
        cout << Taxonomy::compositionGraph(composition) << endl;
        cout << Taxonomy::compositionString(composition) << endl;

        stepGeneration(population);
        history.incrementGeneration(population, POPULATION_COUNT, 0);
    } 

    
}

void stepGeneration(Specimen** population) {

    int randOffset = 2 * (rand() % (POPULATION_COUNT/2));

    Specimen** newPopulation = new Specimen*[POPULATION_COUNT];

    for(int i = 0; i < POPULATION_COUNT; i+= 2) {
        auto first = population[i];
        auto second = population[i+1];

        float firstScore = 1 / (Genome::distance(first->genome, idealNetwork));
        float secondScore = 1 / (Genome::distance(second->genome, idealNetwork));

        auto winner = firstScore > secondScore ? first : second;

        auto offspring1 = new Specimen(winner->genome->mitosis(0.5,0.3,0.1), winner);
        auto offspring2 = new Specimen(winner->genome->mitosis(0.5,0.3,0.1), winner);

        newPopulation[i] = offspring1;
        newPopulation[(i+1 + randOffset) % POPULATION_COUNT] = offspring2;
    }
    for(int i = 0; i < POPULATION_COUNT; i++) {
        //delete population[i]; -- don't want to accidentally delete progenitors!
        population[i] = newPopulation[i];
    }
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