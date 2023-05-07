import pygame
import numpy as np
import math

# Define constants
SCREEN_WIDTH = 50
SCREEN_HEIGHT = 50
MAX_SPEED = 1
MAX_ACCEL = 0.5
MAX_ROT_SPEED = 30
GOAL_HEIGHT = 5
GOAL_DIST = 20


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
        print(totalWeights)
        print(totalNeurons)
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
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
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
    # if layer != len(layershapes) - 1:
    #     output[output < 0] = 0
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

    # get acceleration
    gamestate[0] = output[0] * MAX_ACCEL
    # get rot speed
    gamestate[1] = output[1] * MAX_ROT_SPEED
    
    print(gamestate)
    
    #return inputs
    return gamestate

# Read in the bots' networks
layershapes, allWeights, allBiases = readWeightsAndBiasesAll()

NUM_BOTS = 2
bestoffset = 0
# Define initial bot states, ball positions and networks
bots = []
networks = []
ball = {'posx': 0, 'posy': 0, 'vel': 0, 'dir': 0}
for i in range(NUM_BOTS):
    bot = {'posx': 0, 'posy': 0, 'vel': 0, 'dir': 0, 'score': 0}
    bots.append(bot)
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

        state.append(bots[i % 2]['posx'])
        state.append(bots[i % 2]['posy'])

        state.append(bots[i % 2]['vel'])
        state.append(bots[i % 2]['dir'])

        state.append(bots[i % 2]['score'])

        otherInfo = False

        if otherInfo:
            state.append(bots[(i + 1) % 2]['posx'])
            state.append(bots[(i + 1) % 2]['posy'])    

            state.append(bots[(i + 1) % 2]['velx'])
            state.append(bots[(i + 1) % 2]['vely'])
        else:
            state.append(0)
            state.append(0)
            state.append(0)
            state.append(0)

        state.append(ball['posx'])
        state.append(ball['posy'])

        state.append(ball['vel'])
        state.append(ball['dir'])

        inputs = get_actions_air_hockey(state, network['weights'], network['biases'])
        

        bot['vel'] += inputs[0]
        bot['dir'] += inputs[1]

        bot['posx'] += bot['vel'] * math.cos(math.radians(bot['dir']))
        bot['posy'] += bot['vel'] * math.sin(math.radians(bot['dir']))

    # Draw bots and targets
    screen.fill((255, 255, 255))
    for i in range(NUM_BOTS):
        bot = bots[i]
        pygame.draw.circle(screen, ((i * 25) % 230, (i * 50) % 256, (i * 33) % 256), (int(bot['posx']) + SCREEN_WIDTH / 2, int(bot['posy']) + SCREEN_HEIGHT / 2), 8)

    pygame.draw.circle(screen, (0, 0, 0), (int(ball['posx'] + SCREEN_WIDTH / 2), int(ball['posy'] + SCREEN_HEIGHT / 2)), 8)

    # Update display
    pygame.display.update()
    clock.tick(60)