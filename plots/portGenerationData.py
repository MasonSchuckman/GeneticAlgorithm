

def importData(filename):
    compositionsFile = open(filename, "r")
    compositions = compositionsFile.read().strip().split('\n')
    
    numGenerations = len(compositions)
    allSpeciesIDs = set()
    allGenerationWeights = []
    for generation in compositions:
        rawPairs = generation.strip(',').split(',')
        mappedPairs = {}
        for p in rawPairs:
            [id,density] = p.split(':')
            id = int(id)
            density = float(density)

            allSpeciesIDs.add(id)
            mappedPairs[id] = density

        allGenerationWeights.append(mappedPairs)


    allSpeciesIDs = list(allSpeciesIDs)
    toReturn = []
    
    for species in allSpeciesIDs:
        speciesComposition = []
        for gen in range(numGenerations):
            speciesComposition.append(allGenerationWeights[gen].get(species, 0))
        toReturn.append(speciesComposition)

    compositionsFile.close()
    return allSpeciesIDs, toReturn

