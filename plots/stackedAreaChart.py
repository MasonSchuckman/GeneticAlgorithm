import numpy as np
import matplotlib.pyplot as plt
 
# --- FORMAT 1

if __name__ == '__main__':
    # Your x and y axis
    x=range(1,100)
    y=[ [1,4,6,8,9], [2,2,7,10,12], [2,8,5,10,6] ]


    # Basic stacked area chart.
    plt.stackplot(x,y, labels=['A','B','C'])
    plt.legend(loc='upper left')
    plt.show()
 