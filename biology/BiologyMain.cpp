#include "Genome.h"
#include "Taxonomy.h"

using std::cout;
using std::endl;

Genome** generateGenomes(int count, int *shape, int shapeLen);
void mutatingDistance(Genome* toMutate, int epochs);

int main() {

    int SHAPE[] = {2,2};
    int SHAPE_LENGTH = sizeof(SHAPE) / sizeof(int);
    int COUNT = 5;
    
    Genome** genomes = generateGenomes(COUNT, SHAPE, SHAPE_LENGTH);
    for(int i = 0; i < COUNT; i++) {
        for(int j = 0; j < COUNT; j++) {
            float d = Taxonomy::distance(genomes[i], genomes[j]);
            std::cout << "dist " << i << "," << j << ": " << d << endl;
        }
    }

    //mutatingDistance(genomes[0], 20);

    cout << "original: " << endl << genomes[0]->bodyString() << endl;

}


void mutatingDistance(Genome* toMutate, int epochs) {
    Genome* child = new Genome(toMutate);
    cout << "original genome: " << endl << toMutate->bodyString() << endl;

    for(int i = 1; i <= epochs; i++) {
        cout << endl;
        child = child->mitosis(50,0.1,0.03);
        cout << "epoch " << i << ":" << endl;
        cout << "genome body: " << endl << child->bodyString() << endl;
        cout << "distance: " << Taxonomy::distance(toMutate, child) << endl;

    }
}

Genome** generateGenomes(int count, int *shape, int shapeLen) {
    Genome** genomes = new Genome*[count];

    for(int i = 0; i < count; i++) 
        genomes[i] = new Genome(shape, shapeLen);
    
    return genomes;
}