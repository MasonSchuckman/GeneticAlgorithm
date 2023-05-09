import numpy as np
import matplotlib.pyplot as plt
 
import portGenerationData as pg



if __name__ == '__main__':
    # Your x and y axis

    
    labels, data = pg.importData('plots/compositions.txt')
    
    x = range(len(data[0]))
    y = data

    # Basic stacked area chart.
    plt.stackplot(x,y, labels=labels)
    plt.show()
 