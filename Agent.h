#include "genome.h"


class Agent{
public:

    Agent(int * layerShapes, int numLayers){
        this->genes = new Genome(layerShapes);
    }

    ~Agent(){
        delete genes;
    }

private:

    Genome * genes = nullptr;


};
