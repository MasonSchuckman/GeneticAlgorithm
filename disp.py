import pygame
import random
import numpy as np
import math
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


best = 0
layershapes, all_weights, all_biases = readWeightsAndBiasesAll()
networks = [{'weights': all_weights[best], 'biases': all_biases[best]},
            {'weights': all_weights[best + 1], 'biases': all_biases[best + 1]}]


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
    #     

    
    return output

def calculate_activations(net_weights, net_biases, state):
    activs = []
    inputs = np.array(state)
    prevLayer = inputs
    activs.append(inputs)
    numHiddenLayers = len(layershapes) - 2
    hidden_outputs = [None] * numHiddenLayers
    #forward prop for the hidden layers
    for i in range(numHiddenLayers):
        #print("iter={}, inshape = {}, outshape = {}".format(i, layershapes[i], layershapes[i + 1]))

        hidden_outputs[i] = forward_propagation(prevLayer, net_weights[i], net_biases[i + 1], layershapes[i], layershapes[i + 1],  i)
        prevLayer = hidden_outputs[i]
        activs.append(hidden_outputs[i])

    output = forward_propagation(prevLayer, net_weights[numLayers - 2], net_biases[numLayers - 1], layershapes[numLayers - 2], layershapes[numLayers - 1], numLayers - 1)
    activs.append(output)

    
    #print(gamestate)
    
    return activs

# Define constants
width = 800
height = 600
background_color = (50, 50, 50)
neuron_radius = 15
neuron_margin = 20
layer_margin = 80

# Create the window
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Neural Network Activations")

def display_activations(activations, weights):
    def normalize_activations(activations):
        min_activation = min(min(layer) for layer in activations)
        max_activation = max(max(layer) for layer in activations)
        range_activation = max_activation - min_activation

        normalized_activations = []
        for layer in activations:
            normalized_layer = [(a - min_activation) / range_activation for a in layer]
            normalized_activations.append(normalized_layer)

        return normalized_activations

    # Normalize the activations
    normalized_activations = normalize_activations(activations)

    # Constants
    neuron_radius = 15
    neuron_margin = 20
    layer_margin = 80

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return

        # Clear the screen
        screen.fill(background_color)

        # Draw the neurons and connections
        for i, layer in enumerate(normalized_activations):
            layer_height = len(layer) * (2 * neuron_radius + neuron_margin) - neuron_margin
            y_start = (height - layer_height) // 2
            x = layer_margin + i * (2 * neuron_radius + layer_margin)

            for j, neuron_activation in enumerate(layer):
                y = y_start + j * (2 * neuron_radius + neuron_margin)
                color = int(neuron_activation * 255)
                pygame.draw.circle(screen, (color, color, color), (x, y), neuron_radius)

                # Draw connections to the next layer
                if i < len(normalized_activations) - 1:
                    next_layer = normalized_activations[i + 1]
                    next_layer_height = len(next_layer) * (2 * neuron_radius + neuron_margin) - neuron_margin
                    next_layer_y_start = (height - next_layer_height) // 2
                    next_layer_x = layer_margin + (i + 1) * (2 * neuron_radius + layer_margin)

                    for k, next_neuron_activation in enumerate(next_layer):
                        next_neuron_y = next_layer_y_start + k * (2 * neuron_radius + neuron_margin)
                        weight = weights[i][j][k]
                        weight_color = int((weight + 1) / 2 * 255)  # Assuming weights are in the range [-1, 1]
                        pygame.draw.line(screen, (weight_color, weight_color, weight_color), (x, y), (next_layer_x, next_neuron_y))

        # Update the display
        pygame.display.flip()


# Test the function with a sample neural network activation
activations = [
    [0.1, 0.5, 0.9],
    [0.7, 0.3, 0.4, 0.6],
    [0.0, 1.0]
]

display_activations(activations)
