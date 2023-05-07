import pygame
import numpy as np
import math

# Define constants
MAP_WIDTH = 50
MAP_HEIGHT = 50
SCREEN_SCALE = 16
MAX_SPEED = 2
MAX_ACCEL = 0.5
MAX_ROT_SPEED = 30
GOAL_HEIGHT = 5
GOAL_DIST = 20
ACTOR_SIZE = 1


import struct
numLayers = 0
# Read the all bot format
def readWeightsAndBiasesAll():
    with open("allBots.data", "rb") as infile:
        # Read the total number of bots
        TOTAL_BOTS = struct.unpack('i', infile.read(4))[0]

        # Read the total number of weights and neurons
        totalWeights = struct.unpack('i', infile.read(4))[0]
        totalNeurons = struct.unpack('i', infile.read(4))[0]
        print("Total weights: " + str(totalWeights))
        print("Total Neurons: " + str(totalNeurons))
        # Read the number of layers and their shapes
        global numLayers
        numLayers = struct.unpack('i', infile.read(4))[0]
        layerShapes = [struct.unpack('i', infile.read(4))[0] for _ in range(numLayers)]
        print(layerShapes)
        # Allocate memory for the weights and biases
        all_weights = []
        all_biases = []

        # Read the weights and biases for each bot
        for bot in range(TOTAL_BOTS):
            # Read the weights for each layer
            weights = []
            for i in range(numLayers - 1):
                layerWeights = np.zeros(layerShapes[i] * layerShapes[i + 1], dtype=np.float32)
                for j in range(layerShapes[i] * layerShapes[i+1]):                    
                    weight = struct.unpack('f', infile.read(4))[0]
                    layerWeights[j] = weight
                weights.append(layerWeights)

            # Read the biases for each layer
            biases = []
            for i in range(numLayers):
                layerBiases = np.zeros(layerShapes[i], dtype=np.float32)
                for j in range(layerShapes[i]):
                    bias = struct.unpack('f', infile.read(4))[0]
                    layerBiases[j] = bias
                biases.append(layerBiases)

            all_weights.append(weights)
            all_biases.append(biases)

    return layerShapes, all_weights, all_biases


# Reads the one bot format
def read_weights_and_biases():
    with open('weights.data', 'r') as f:
        data = f.read()

    weights_start = data.index("net_weights") + 13
    weights_end = data.index("]", weights_start) + 1
    biases_start = data.index("net_biases") + 12
    biases_end = data.index("]", biases_start) + 1

    weights_data = data[weights_start:weights_end].replace('[','').replace(']','').split(',')
    biases_data = data[biases_start:biases_end].replace('[','').replace(']','').split(',')
    #print(data)
    weights = np.array([float(i.strip()) for i in weights_data])
    biases = np.array([float(i.strip()) for i in biases_data])
    
    #print(biases)
    return weights, biases


# Define neural network parameters
layershapes = []

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((MAP_WIDTH * SCREEN_SCALE, MAP_HEIGHT * SCREEN_SCALE))
clock = pygame.time.Clock()

def forward_propagation(inputs, weights, biases, input_size, output_size, layer):
    output = np.zeros(output_size)
    #print(input_size, " ", output_size, " ", layer)
    weights = np.array(weights).reshape((input_size, output_size))

    # Initialize output to biases
    output[:] = biases

    # Compute dot product of input and weights
    output[:] += np.dot(inputs, weights)

    # Apply activation function (ReLU for non-output layers, sigmoid for output layer)
    if layer != len(layershapes) - 1:
        output[output < 0] = 0
    # else:
    #     #print('sigmoid')
    #     output[:] = 1.0 / (1.0 + np.exp(-output))
    # if layer != len(layershapes) - 1:
    #     output[output < 0] = 0

    return output

def get_actions_air_hockey(state, net_weights, net_biases):
    inputs = np.array(state)
    prevLayer = inputs
    numHiddenLayers = len(layershapes) - 2
    hidden_outputs = [None] * numHiddenLayers
    #forward prop for the hidden layers
    for i in range(numHiddenLayers):
        #print("iter={}, inshape = {}, outshape = {}".format(i, layershapes[i], layershapes[i + 1]))

        hidden_outputs[i] = forward_propagation(prevLayer, net_weights[i], net_biases[i + 1], layershapes[i], layershapes[i + 1],  i)
        prevLayer = hidden_outputs[i]

    output = forward_propagation(prevLayer, net_weights[numLayers - 2], net_biases[numLayers - 1], layershapes[numLayers - 2], layershapes[numLayers - 1], numLayers - 1)

    gamestate = [None] * 2

    # get xAccel
    gamestate[0] = output[0] * MAX_ACCEL
    # get xAccel
    gamestate[1] = output[1] * MAX_ACCEL
    
    #return inputs
    return gamestate

# Read in the bots' networks
layershapes, allWeights, allBiases = readWeightsAndBiasesAll()

NUM_BOTS = 2
bestoffset = 0
# Define initial bot states, ball positions and networks
bots = []
networks = []
ball = {'posx': 0, 'posy': 0, 'velx': 0, 'vely': 0}
gamestatus = {'tick': 0, 'gen': 0}
bot1 = {'posx': -10, 'posy': 0, 'velx': 0, 'vely': 0, 'score': 0}
bots.append(bot1)
bot2 = {'posx': 8, 'posy': 6, 'velx': 0, 'vely': 0, 'score': 0}
bots.append(bot2)

limits = [GOAL_DIST, GOAL_DIST, MAX_SPEED, MAX_SPEED, 1]

for i in range(NUM_BOTS):
    network = {'weights': allWeights[i + bestoffset], 'biases': allBiases[i + bestoffset]}
    networks.append(network)

# Main game loop
while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    

    # Get actions from neural networks
    for i in range(NUM_BOTS):
        bot = bots[i]
        network = networks[i]
        
        #state of the sim on this iter
        state = []

        state.append(bots[i % 2]['posx'] / limits[0])
        state.append(bots[i % 2]['posy'] / limits[1])

        state.append(bots[i % 2]['velx'] / limits[2])
        state.append(bots[i % 2]['vely'] / limits[3])

        #state.append(bots[i % 2]['score'])
        state.append(0)

        otherInfo = True

        if otherInfo:
            state.append(bots[(i + 1) % 2]['posx'] / limits[0])
            state.append(bots[(i + 1) % 2]['posy'] / limits[1])    

            state.append(bots[(i + 1) % 2]['velx'] / limits[2])
            state.append(bots[(i + 1) % 2]['vely'] / limits[3])
            #state.append(bots[(i + 1) % 2]['score'])
            state.append(0)
        else:
            state.append(0) # x pos
            state.append(0) # y pos
            state.append(0) # x vel
            state.append(0) # y vel
            state.append(0) # score

        state.append(ball['posx'] / limits[0])
        state.append(ball['posy'] / limits[1])

        state.append(ball['velx'] / limits[2])
        state.append(ball['vely'] / limits[3])
        #state.append(gamestatus['tick'])
        state.append(0)

        #Reverse x related info for bot B
        if i % 2 == 1:
            for entity in range(3):
                state[0 + entity * 5] *= -1
                state[2 + entity * 5] *= -1

        inputs = get_actions_air_hockey(state, network['weights'], network['biases'])
        
        accelx = inputs[0]
        accely = inputs[1]
        accel = math.hypot(accelx, accely)

        if(accel > MAX_ACCEL):
            f = MAX_ACCEL / accel 
            accelx *= f
            accely *= f

        bot['velx'] += accelx
        bot['vely'] += accely
        speed = math.hypot(bot['velx'], bot['vely'])
        if(speed > MAX_SPEED):
            f = MAX_SPEED / speed 
            bot['velx'] *= f
            bot['vely'] *= f

        bot['posx'] += bot['velx']
        bot['posy'] += bot['vely']

        botDist = [0, 0]
        for i in range(2):
            botDist[i] = math.hypot(
                ball['posx'] - bots[i]['posx'],
                ball['posy'] - bots[i]['posx'])
        # Bot 0 has a slight disadvantage
        closestBot = int(botDist[1] < botDist[0])
        bots[closestBot]['score'] += 1
        for i in range(2):
            dist = math.hypot(ball['posx'] - bots[i]['posx'], ball['posy'] - bots[i]['posy']);
            if (dist < ACTOR_SIZE):
                print("HIT")
                #exit()
                ball['velx'] = bots[i]['velx']
                ball['vely'] = bots[i]['vely']
                bots[i]['score'] += 100


        # Either bounce or score
        if (abs(ball['posx']) > GOAL_DIST):
            # Goal
            if (abs(ball['posy']) < GOAL_HEIGHT):
                # Bot 0 wants to score to the right
                scorer = int(ball['posy'] > 0)
                bots[scorer]['score'] += 10000
            else:
                ball['velx'] *= -1
        if (abs(ball['posy']) > GOAL_DIST):
            ball['vely'] *= -1

        ball['posx'] += ball['velx'];
        ball['posy'] += ball['vely'];

    # Draw bots and targets
    screen.fill((255, 255, 255))
    for i in range(NUM_BOTS):
        bot = bots[i]
        pygame.draw.circle(screen, ((i * 25) % 230, (i * 50) % 256, (i * 33) % 256), 
            (int(bot['posx']) + MAP_WIDTH / 2 * SCREEN_SCALE, 
            int(bot['posy']) + MAP_HEIGHT / 2 * SCREEN_SCALE), ACTOR_SIZE / 2 * SCREEN_SCALE)

    pygame.draw.circle(screen, (125, 125, 125), 
        (int(ball['posx'] + MAP_WIDTH / 2) * SCREEN_SCALE, 
        int(ball['posy'] + MAP_HEIGHT / 2) * SCREEN_SCALE), ACTOR_SIZE / 2 * SCREEN_SCALE)

    wallHeight = (MAP_HEIGHT / 2 - GOAL_HEIGHT) * SCREEN_SCALE
    wallWidth = (MAP_WIDTH / 2 - GOAL_DIST) * SCREEN_SCALE

    # LEFT WALLS
    pygame.draw.rect(screen, (0, 0, 0), 
        pygame.Rect(0, 0, 
        wallWidth, wallHeight
     ))
    pygame.draw.rect(screen, (0, 0, 0), 
        pygame.Rect(0, MAP_HEIGHT * SCREEN_SCALE - wallHeight, 
        wallWidth, wallHeight
    ))
        
    # RIGHT WALLS
    pygame.draw.rect(screen, (0, 0, 0), 
        pygame.Rect(MAP_WIDTH * SCREEN_SCALE - wallWidth, 0, 
        wallWidth, wallHeight
    ))
    pygame.draw.rect(screen, (0, 0, 0), 
        pygame.Rect(MAP_WIDTH * SCREEN_SCALE - wallWidth, MAP_HEIGHT * SCREEN_SCALE - wallHeight, 
        wallWidth, wallHeight
     ))

    # ROOF AND FLOOR
    roofHeight = (MAP_HEIGHT / 2 - GOAL_DIST) * SCREEN_SCALE
    pygame.draw.rect(screen, (0, 0, 0),
        pygame.Rect(0, 0, 
            MAP_WIDTH * SCREEN_SCALE, roofHeight
    ))
    pygame.draw.rect(screen, (0, 0, 0),
        pygame.Rect(0, MAP_HEIGHT * SCREEN_SCALE - roofHeight, 
            MAP_WIDTH * SCREEN_SCALE, roofHeight
    ))

    # Update display
    pygame.display.update()
    clock.tick(60)