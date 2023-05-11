import pygame
import numpy as np
import math

# Define constants
MAX_SPEED = 10
MAX_ACCEL = 2.0
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
NETWORK_DISPLAY_WIDTH = 400
from network_visualizer import *

# import struct
# # Open the binary file for reading
# with open("allBots.data", "rb") as f:

#     # Read the first 3 integers (12 bytes) to get the total number of bots, weights, and neurons
#     num_bots, num_weights, num_neurons = struct.unpack("iii", f.read(12))

#     # Read the number of layers and their shapes
#     num_layers = struct.unpack("i", f.read(4))[0]
#     layer_shapes = struct.unpack(f"{num_layers}i", f.read(num_layers * 4))

#     # Iterate over each bot
#     for bot in range(num_bots):
#         print(f"Bot {bot}:")
#         # Iterate over the weights for this bot
#         for i in range(num_weights):
#             # Read the weight as a float
#             weight = struct.unpack("f", f.read(4))[0]
#             if i % (64 + 16) == 0:
#                 print(f"  Weight set {i // (64 + 16)}:")
#             if i % 64 == 0:
#                 print("    64 weights: ", end="")
#             print(f"{weight:.6f}", end=", ")
#             if (i + 1) % 64 == 0:
#                 print()
#         print()
#         # Iterate over the biases for this bot
#         bias_offset = bot * num_neurons + layer_shapes[0]
#         for i in range(num_neurons):
#             # Read the bias as a float
#             bias = struct.unpack("f", f.read(4))[0]
#             if i % (64 + 16) == 0:
#                 print(f"  Bias set {i // (64 + 16)}:")
#             if i % 16 == 0:
#                 print("    16 biases: ", end="")
#             print(f"{bias:.6f}", end=", ")
#             if (i + 1) % 16 == 0:
#                 print()
#         print()



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

    return layerShapes, all_weights, all_biases, TOTAL_BOTS


# Reads the one bot format
def read_weights_and_biases(filename):
    with open(filename, 'r') as f:
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
screen = pygame.display.set_mode((SCREEN_WIDTH + NETWORK_DISPLAY_WIDTH * 2, SCREEN_HEIGHT))
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

def get_actions_multibot(state, net_weights, net_biases):
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
    gamestate[1] = output[1] * MAX_ACCEL

    #cap the acceleration
    accel = math.hypot(gamestate[0], gamestate[1])
    if accel > MAX_ACCEL:
        f = MAX_ACCEL / accel
        gamestate[0] *= f
        gamestate[1] *= f
    
    
    print(gamestate)
    
    #return acceleration
    return gamestate

# Read in the bots' networks
layershapes, allWeights, allBiases, numBots = readWeightsAndBiasesAll()
converted_all_weights = convert_weights(allWeights, layershapes)

# how many bots (at most) do you want to show?
NUM_BOTS = 20

NUM_BOTS = min(NUM_BOTS,numBots)
bestoffset = 0
# Define initial bot states, target positions and networks
bots = []
targets = []
networks = []

network_display_left = pygame.Surface((NETWORK_DISPLAY_WIDTH, SCREEN_HEIGHT))
network_display_right = pygame.Surface((NETWORK_DISPLAY_WIDTH, SCREEN_HEIGHT))

net_displays = [network_display_left, network_display_right]
net_locations = [(0,0), (SCREEN_WIDTH + NETWORK_DISPLAY_WIDTH, 0)]

for i in range(NUM_BOTS):
    bot = {'posx': 0, 'posy': 0, 'velx': 0, 'vely': 0}
    bots.append(bot)
    target = {'x': 0, 'y': 0}
    targets.append(target)
    network = {'weights': allWeights[i + bestoffset], 'biases': allBiases[i + bestoffset]}
    networks.append(network)
BLACK = (255,255,255)
# Main game loop
while True:
    screen.fill(BLACK)
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    

    # Get actions from neural networks
    for i in range(NUM_BOTS):
        bot = bots[i]
        target = targets[i]
        network = networks[i]
        
        #state of the sim on this iter
        state = []

        state.append(bots[i % NUM_BOTS]['posx'])
        state.append(bots[i % NUM_BOTS]['posy'])

        state.append(bots[i % NUM_BOTS]['velx'])
        state.append(bots[i % NUM_BOTS]['vely'])

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

        state.append(targets[0]['x'])
        state.append(targets[0]['y'])

        acceleration = get_actions_multibot(state, network['weights'], network['biases'])
        

        bot['velx'] += acceleration[0]
        bot['vely'] += acceleration[1]

        # clamp the velocity
        speed = math.hypot(bot['velx'], bot['vely'])
        if speed > MAX_SPEED:
            normalization = MAX_SPEED / speed
            bot['velx'] *= normalization
            bot['vely'] *= normalization
        


        bot['posx'] += bot['velx']
        bot['posy'] += bot['vely']

        activations_left = calculate_activations(networks[i]['weights'], networks[i]['biases'], state, layershapes, numLayers)
        #display_activations(activations_left, converted_all_weights[i], net_displays[i])
        #display_activations2(activations_left, converted_all_weights[i], net_displays[i], SCREEN_HEIGHT)
        #screen.blit(net_displays[i], net_locations[i])

    # Get current mouse positions and update target positions
    mouse_pos = pygame.mouse.get_pos()
    for i in range(NUM_BOTS):
        target = targets[i]
        target['x'] = mouse_pos[0] - SCREEN_WIDTH / 2 - NETWORK_DISPLAY_WIDTH
        target['y'] = mouse_pos[1] - SCREEN_HEIGHT / 2

    # Draw bots and targets
    
    for i in range(NUM_BOTS):
        bot = bots[i]
        target = targets[i]
        pygame.draw.circle(screen, (255, 0, 0), (int(target['x']) + SCREEN_WIDTH / 2 + NETWORK_DISPLAY_WIDTH, int(target['y']) + SCREEN_HEIGHT / 2), 8)
        pygame.draw.circle(screen, ((i * 25) % 230, (i * 50) % 256, (i * 33) % 256), (int(bot['posx']) + SCREEN_WIDTH / 2 + NETWORK_DISPLAY_WIDTH, int(bot['posy']) + SCREEN_HEIGHT / 2), 8)

    # Update display
    pygame.display.update()
    clock.tick(60)