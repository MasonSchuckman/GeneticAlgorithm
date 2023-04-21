from collections import Counter
import random

size = 2**5
starting = [i for i in range(size)]
print('Iter 0')
print(starting)


iters = 5
blocks = int(size / 2)
next = [0] * size


for iter in range(iters):
    
    shift = int(size * random.uniform(0, 1)) % size
    print('shift = ', shift)
    for block in range(blocks):
        
        offsetBot1 = block * 2;
        
        offsetBot2 = (block * 2 + shift * 2 + 1) % size


        winner = 0
        if starting[offsetBot1] > starting[offsetBot2]:
            winner = offsetBot1
        else:
            winner = offsetBot2
        next[offsetBot1] = starting[winner]
        next[offsetBot2] = starting[winner]
       
    print("\nIter ", (iter + 1))
    print(next)
    nonmax = Counter(next)
    print("num Max = ", nonmax[size - 1])
    temp = starting
    starting = next
    next = temp
   





