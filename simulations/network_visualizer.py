import pygame
import numpy as np
import struct
import math

def convert_weights(all_weights, layer_shapes):
    # Calculate the number of bots and layers
    num_bots = len(all_weights)
    num_layers = len(all_weights[0])

    # Create a 3D list to store the converted weights
    converted_weights = [[[None] * (layer_shapes[i] * layer_shapes[i + 1]) for i in range(num_layers - 1)] for _ in range(num_bots)]

    # Fill the converted_weights list
    for bot in range(num_bots):
        for layer in range(num_layers - 1):
            for src_neuron in range(layer_shapes[layer]):
                for dst_neuron in range(layer_shapes[layer + 1]):
                    weight = all_weights[bot][layer][src_neuron * layer_shapes[layer + 1] + dst_neuron]
                    converted_weights[bot][layer][src_neuron][dst_neuron] = weight

    return converted_weights

def draw_connections(surface, weights, activations, positions, layer_spacing, neuron_radius):
    for i in range(len(weights)):
        src_layer = positions[i]
        dst_layer = positions[i + 1]

        for src_idx, src_pos in enumerate(src_layer):
            for dst_idx, dst_pos in enumerate(dst_layer):
                weight = weights[i][src_idx][dst_idx]  # Updated indexing order
                activation = activations[i][src_idx] * activations[i + 1][dst_idx]
                intensity = int(activation * 255)

                if weight > 0:
                    color = (0, intensity, 0)
                else:
                    color = (intensity, 0, 0)

                pygame.draw.line(
                    surface,
                    color,
                    (src_pos[0] + neuron_radius, src_pos[1]),
                    (dst_pos[0] - neuron_radius, dst_pos[1]),
                    1,
                )

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

def display_activations(activations, weights, surface):
    num_layers = len(activations)
    layer_spacing = surface.get_width() / (num_layers + 1)
    neuron_radius = 10
    positions = []

    for i, layer in enumerate(activations):
        positions.append([])
        neuron_spacing = surface.get_height() / (len(layer) + 1)

        for j, activation in enumerate(layer):
            x = int((i + 1) * layer_spacing)
            y = int((j + 1) * neuron_spacing)
            positions[-1].append((x, y))

            color = int(activation * 255)
            pygame.draw.circle(surface, (color, color, color), (x, y), neuron_radius)

    draw_connections(surface, weights, activations, positions, layer_spacing, neuron_radius)
