#include "Genome.h"
#include "Species.h"
#include "Specimen.h"
#include "Taxonomy.h"

#include <cmath>
#include <chrono>
#include <thread>

using std::cout;
using std::endl;

Specimen** generatePopulation(int count, int *shape, int shapeLen);
void mutatingDistance(Genome* toMutate, int epochs);
std::string compositionString(std::map<Species*, float>* composition);
std::string compositionGraph(std::map<Species*, float>* composition);
Specimen** stepGeneration(Specimen** population);
Species* getGenesisAncestor(Species* species);
void printAncestry(Species* species, int offset);

int SHAPE[] = {2,2,2};
int SHAPE_LENGTH = sizeof(SHAPE) / sizeof(int);
int POPULATION_COUNT = 300000;
int EPOCHS = 500;
float PROGENITOR_THRESHOLD = 0.95f;
int GRAPH_NUM_ROWS = 10;

int EFFECTIVENESS = 30000;

Genome* idealNetwork = new Genome(SHAPE, SHAPE_LENGTH);

int main() {
    srand(7);

    cout << idealNetwork->bodyString() << endl << endl;
    idealNetwork->exportWeights("test");
    Genome* newG = new Genome("test");
    cout << newG->bodyString() << endl;
    newG->exportWeights("test2");
    return 0;
    
    Specimen** genesisPopulation = generatePopulation(POPULATION_COUNT, SHAPE, SHAPE_LENGTH);

    Specimen** population = new Specimen*[POPULATION_COUNT];
    for(int i = 0; i < POPULATION_COUNT; i++) 
        population[i] = genesisPopulation[i];

    Taxonomy history(population, POPULATION_COUNT);

    for(int i = 0; i < EPOCHS; i++) {
        auto compositions = history.speciesComposition();
        int lastRow = std::min((float) GRAPH_NUM_ROWS, (float) compositions->size());
        std::vector<std::tuple<Species*, float>> topCompositions(compositions->begin(), compositions->begin() + lastRow);
        
        for(int i = 0; i++ < 30; cout << endl);
        cout << std::flush;

        cout << "generation " << history.getYear()+1 << endl;
        cout << Taxonomy::compositionGraph(&topCompositions, 40) << endl;
        cout << Taxonomy::compositionString(&topCompositions) << endl << std::flush;
        std::this_thread::sleep_for(std::chrono::milliseconds(0));

        Specimen** newPopulation = stepGeneration(population);
        idealNetwork = idealNetwork->mitosis(0.8,0.05,0);
        history.incrementGeneration(newPopulation, POPULATION_COUNT, PROGENITOR_THRESHOLD);

        for(int i = 0; i < POPULATION_COUNT; i++) 
            history.pruneSpecimen(population[i]);
        
        delete population;
        population = newPopulation;
    } 
    
    for(int i = 0; i < POPULATION_COUNT; i++)
        printAncestry(genesisPopulation[i]->species, 0);
}


void printAncestry(Species* species, int offset) {

    if(offset > 0) {
    std::cout << offset << "| ";
    for(int i = 0; i++ < offset; std::cout << "  ");
    std::cout << species->id << std::endl;
    }

    for(Species* subspecies : species->descendantSpecies)
        printAncestry(subspecies, offset+1);
}

Specimen** stepGeneration(Specimen** population) {

    int randOffset = 2 * ((rand() % (POPULATION_COUNT/2)) % EFFECTIVENESS);

    Specimen** newPopulation = new Specimen*[POPULATION_COUNT];

    for(int i = 0; i < POPULATION_COUNT; i+= 2) {
        auto first = population[i];
        auto second = population[i+1];

        float firstScore = 1 / (Genome::distance(first->genome, idealNetwork));
        float secondScore = 1 / (Genome::distance(second->genome, idealNetwork));

        auto winner = firstScore > secondScore ? first : second;

        auto offspring1 = new Specimen(winner->genome->mitosis(0.6,0.09,0), winner);
        auto offspring2 = new Specimen(winner->genome->mitosis(0.6,0.09,0), winner);

        newPopulation[i] = offspring1;
        newPopulation[(i+1 + randOffset) % POPULATION_COUNT] = offspring2;
    }

    return newPopulation;
}

Specimen** generatePopulation(int count, int* shape, int shapeLen) {

    Specimen** population = new Specimen*[count];

    for(int i = 0; i < count; i++) {
        Genome* nextGenome = new Genome(shape, shapeLen);
        population[i] = new Specimen(nextGenome, (Specimen*) nullptr);
    }
    
    return population;
}