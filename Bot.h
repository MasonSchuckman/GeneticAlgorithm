#ifndef BOT_H
#define BOT_H

#include "Genome.h"


class Bot{
public:

    Bot(int * layerShapes, int numLayers){
        this->genes = new Genome(layerShapes);
    }

    ~Bot(){
        delete genes;
    }

private:

    Genome * genes = nullptr;
};

#endif