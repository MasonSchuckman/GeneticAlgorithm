


def importFromFile(filename):
    geneFile = open(filename+".gene", "r")
    contents = geneFile.read().split('\n')
    contents = [l.split() for l in contents]

    shape = list(map(int, contents[1][1:]))
    cons  = list(map(float, contents[2][1:]))
    bias  = list(map(float, contents[3][1:]))
    geneFile.close()
    
    return shape, cons, bias


def exportToFile(filename, genome):
    shape, cons, bias = genome
    geneFile = open(filename+".gene", "w")
    geneFile.write(f"layers {len(shape)}\n")
    
    stringify = lambda x: " ".join([str(i) for i in x])

    geneFile.write(f"shape {stringify(shape)} \n" )
    geneFile.write(f"connections {stringify(cons)} \n" )
    geneFile.write(f"biases {stringify(bias)} \n" )
    geneFile.close()



res = importFromFile('biology/test')
print(res)

exportToFile("out", res)

