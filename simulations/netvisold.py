import pygame
import random
import numpy as np
import math
import struct


def forward_propagation(inputs, weights, biases, input_size, output_size, layer, layershapes):
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

def calculate_activations(net_weights, net_biases, state, layershapes, numLayers):
    activs = []
    inputs = np.array(state)
    prevLayer = inputs
    activs.append(inputs)
    numHiddenLayers = len(layershapes) - 2
    hidden_outputs = [None] * numHiddenLayers
    #forward prop for the hidden layers
    for i in range(numHiddenLayers):
        #print("iter={}, inshape = {}, outshape = {}".format(i, layershapes[i], layershapes[i + 1]))

        hidden_outputs[i] = forward_propagation(prevLayer, net_weights[i], net_biases[i + 1], layershapes[i], layershapes[i + 1],  i, layershapes)
        prevLayer = hidden_outputs[i]
        activs.append(hidden_outputs[i])

    output = forward_propagation(prevLayer, net_weights[numLayers - 2], net_biases[numLayers - 1], layershapes[numLayers - 2], layershapes[numLayers - 1], numLayers - 1, layershapes)
    activs.append(output)

    
    #print(gamestate)
    
    return activs

def convert_weights(all_weights, layer_shapes):
    # Calculate the number of bots and layers
    num_bots = len(all_weights)
    num_layers = len(all_weights[0]) + 1
    #print("shapes = ", layer_shapes)
    # Create a 4D list to store the converted weights
    #converted_weights = [[[[None] for _ in range(layer_shapes[i + 1])] for i in range(num_layers - 1)] for _ in range(num_bots)]
    converted_weights = [None] * num_bots
    for bot in range(num_bots):
        converted_weights[bot] = [None] * (num_layers - 1)
        for layer in range(num_layers - 1):
            converted_weights[bot][layer] = [None] * layer_shapes[layer]
            for srcNeuron in range(layer_shapes[layer]):
                converted_weights[bot][layer][srcNeuron] = [None] * layer_shapes[layer + 1]

    #print(len(converted_weights), len(converted_weights[0]), len(converted_weights[0][0]), len(converted_weights[0][0][0]))

    # Fill the converted_weights list
    for bot in range(num_bots):
        for layer in range(num_layers - 1):
            for src_neuron in range(layer_shapes[layer]):
                for dst_neuron in range(layer_shapes[layer + 1]):
                    weight = all_weights[bot][layer][src_neuron * layer_shapes[layer + 1] + dst_neuron]
                    #print("{} {} {} {} {}".format(bot,layer,src_neuron,dst_neuron, weight))
                    converted_weights[bot][layer][src_neuron][dst_neuron] = weight

    return converted_weights

def display_activations(activations, weights, loc, SCREEN_WIDTH, SCREEN_HEIGHT, screen):
    def normalize(activations):
        min_activation = min(min(layer) for layer in activations)
        max_activation = max(max(layer) for layer in activations)
        range_activation = max_activation - min_activation

        normalized_activations = []
        for layer in activations:
            normalized_layer = [(a - min_activation) / range_activation for a in layer]
            normalized_activations.append(normalized_layer)

        return normalized_activations

    

    # Normalize the activations
    normalized_activations = normalize(activations)
    
    # Constants
    neuron_radius = 15
    neuron_margin = 20
    layer_margin = 20

    # while True:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             return

        # Clear the screen
        #screen.fill("black")

    # Draw the neurons and connections
    for i, layer in enumerate(normalized_activations):
        layer_height = len(layer) * (2 * neuron_radius + neuron_margin) - neuron_margin
        y_start = (SCREEN_HEIGHT - layer_height) // 2
        x = layer_margin + i * (2 * neuron_radius + layer_margin)

        for j, neuron_activation in enumerate(layer):
            y = y_start + j * (2 * neuron_radius + neuron_margin)
            color = int(neuron_activation * 255)
            pygame.draw.circle(screen, (color, color, color), (loc[0] + x, loc[1] + y), neuron_radius)

            # Draw connections to the next layer
            if i < len(normalized_activations) - 1:
                next_layer = normalized_activations[i + 1]
                next_layer_height = len(next_layer) * (2 * neuron_radius + neuron_margin) - neuron_margin
                next_layer_y_start = (SCREEN_HEIGHT - next_layer_height) // 2
                next_layer_x = layer_margin + (i + 1) * (2 * neuron_radius + layer_margin)
                normalized_weights = weights[i][j] / np.linalg.norm(weights[i][j])
                for k, next_neuron_activation in enumerate(next_layer):
                    next_neuron_y = next_layer_y_start + k * (2 * neuron_radius + neuron_margin)
                    #print("{}, {}, {}".format(i,j,k))
                    weight = normalized_weights[k]
                    weight_color = int((weight + 1) / 2 * 255)  # Assuming weights are in the range [-1, 1]
                    #weight_color = 1
                    pygame.draw.line(screen, (weight_color, weight_color, weight_color), (loc[0] + x, loc[1] + y), (loc[0] + next_layer_x, loc[1] + next_neuron_y))

        # Update the display
        pygame.display.flip()