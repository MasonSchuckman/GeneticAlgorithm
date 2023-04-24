
def importGene(filename):
    geneFile = open(filename, "r")
    contents = geneFile.read().split('\n')
    contents = [l.split() for l in contents]

    shape = list(map(int, contents[1][1:]))
    type = contents[2][1]
    cons  = list(map(float, contents[3][1:]))
    bias  = list(map(float, contents[4][1:]))
    geneFile.close()

    return shape, type, cons, bias

def exportGene(filename, genome):
    shape, type, cons, bias = genome
    geneFile = open(filename, "w")
    geneFile.write(f"layers {len(shape)}\n")
    
    stringify = lambda x: " ".join([str(i) for i in x])

    geneFile.write(f"shape {stringify(shape)} \n" )
    geneFile.write(f"type {type} \n" )
    geneFile.write(f"connections {stringify(cons)} \n" )
    geneFile.write(f"biases {stringify(bias)} \n" )
    geneFile.close()



