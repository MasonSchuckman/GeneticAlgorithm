import pygame
import numpy as np
import math

# Define constants
MAX_SPEED = 5
MAX_ACCEL = 1.0
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480


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

# net_weights = np.array([[-0.191083, 0.079841, -0.127740, -0.039663, 0.137477, 0.087965, -0.479188, -0.089636, 0.237234, -0.153124, -0.238110, -0.072986, -0.295682, 0.303986, 0.135147, -0.468184, -0.168100, -0.212201, -0.145062, 0.262820, 0.226249, -0.206955, -0.261188, 0.539418, -0.717928, 0.295275, -0.269030, -0.627511, 0.139830, 0.332173, -0.236078, 0.028958, 0.196836, 0.206497, 0.128855, -0.266095, -0.208614, 0.193854, 0.319410, -0.563935, 0.348045, -0.267320, 0.316727, 0.648191, -0.479277, -0.371274, 0.001850, 0.073868],
# [0.149054, 0.147478, 0.227007, -0.127284, 0.211191, 0.113938, -0.181851, 0.218462, -0.165197, -0.110061, 0.111662, -0.057198, 0.035173, -0.051007, -0.193473, -0.089472]])
# net_biases = np.array([[-0.427300, 0.654512, 0.085583, 0.007135, -0.110449, -0.025819, -0.003751, -0.197740],
# [-0.145017, 0.077235]])

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
layershapes, allWeights, allBiases = readWeightsAndBiasesAll()

NUM_BOTS = 2
bestoffset = 0
# Define initial bot states, target positions and networks
bots = []
targets = []
networks = []
for i in range(NUM_BOTS):
    bot = {'posx': 0, 'posy': 0, 'velx': 0, 'vely': 0}
    bots.append(bot)
    target = {'x': 0, 'y': 0}
    targets.append(target)
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
        target = targets[i]
        network = networks[i]
        
        #state of the sim on this iter
        state = []

        state.append(bots[i % 2]['posx'])
        state.append(bots[i % 2]['posy'])

        state.append(bots[i % 2]['velx'])
        state.append(bots[i % 2]['vely'])

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

    # Get current mouse positions and update target positions
    mouse_pos = pygame.mouse.get_pos()
    for i in range(NUM_BOTS):
        target = targets[i]
        target['x'] = mouse_pos[0] - SCREEN_WIDTH / 2
        target['y'] = mouse_pos[1] - SCREEN_HEIGHT / 2

    # Draw bots and targets
    screen.fill((255, 255, 255))
    for i in range(NUM_BOTS):
        bot = bots[i]
        target = targets[i]
        pygame.draw.circle(screen, (255, 0, 0), (int(target['x']) + SCREEN_WIDTH / 2, int(target['y']) + SCREEN_HEIGHT / 2), 8)
        pygame.draw.circle(screen, ((i * 25) % 230, (i * 50) % 256, (i * 33) % 256), (int(bot['posx']) + SCREEN_WIDTH / 2, int(bot['posy']) + SCREEN_HEIGHT / 2), 8)

    # Update display
    pygame.display.update()
    clock.tick(60)