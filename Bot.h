#ifndef BOT_H
#define BOT_H

#include "biology/Genome.h"
//#include "Genome.h"

class Bot{
public:

    Bot(int * layerShapes, int numLayers){
        this->genes = new Genome(layerShapes, numLayers);
    }

    Bot(std::vector<int> layerShapes){
        this->genes = new Genome(layerShapes.data(), layerShapes.size());
    }

    ~Bot(){
        delete genes;
    }

private:

    Genome * genes = nullptr;
};

#endif
