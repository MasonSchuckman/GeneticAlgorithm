from collections import Counter
import random

scale = 12
size = 2**scale

starting = [i for i in range(size)]
shiftEffectiveness = .02
print('Iter 0. Num bots = {}, shift effectiveness = {}'.format(size, shiftEffectiveness))
#print(starting)


iters = scale * 4
blocks = int(size / 2)
next = [0] * size



for iter in range(iters):

    if iter % 10 == 0:
        shift =  1
    else:
        shift = int(3*(iter/2))
    shift = int(size * random.uniform(0, 1)) % int(size * shiftEffectiveness)
    #print('shift = ', shift)
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
    #print(next)
    nonmax = Counter(next)
    print("Takeover percent = {0:.00%}".format(nonmax[size - 1] / size))
    print("total = {}".format(nonmax[size - 1]))
    temp = starting
    starting = next
    next = temp
   





