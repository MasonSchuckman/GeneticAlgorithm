import pygame
import random
import numpy as np
import math
import struct
from network_visualizer import *

data_file = "allBots.data"
data_file = "RL-bot.data"
numLayers = 0
# Read the all bot format
def readWeightsAndBiasesAll():
    with open(data_file, "rb") as infile:
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
            layerBiases1 = np.zeros(0, dtype=np.float32)
            for i in range(numLayers):
                layerBiases = np.zeros(layerShapes[i], dtype=np.float32)
                for j in range(layerShapes[i]):
                    bias = struct.unpack('f', infile.read(4))[0]
                    layerBiases[j] = bias
                biases.append(layerBiases)

            all_weights.append(weights)
            all_biases.append(biases)
    
    return layerShapes, all_weights, all_biases


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
        #output = np.tanh(output)
        output[output < 0] = 0
    # else:
    #     #print('sigmoid')
    #     output[:] = 1.0 / (1.0 + np.exp(-output))
    # if layer != len(layershapes) - 1:
    #     

    
    return output

def get_actions_cart(state, net_weights, net_biases):
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
    
    return output


# Initialize Pygame
pygame.init()

# Set up the game window
NETWORK_DISPLAY_WIDTH = 600
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH + NETWORK_DISPLAY_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("CartPole")

# Define cart and pole dimensions
CART_WIDTH = 60
CART_HEIGHT = 30
POLE_WIDTH = 6
POLE_LENGTH = 150  # Half of the actual pole length

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Define game state variables
cart_x = 0
cart_vx = 0
pole_angle = 0#np.pi  # Pole angle in radians (starting straight up)
pole_angular_velocity = 0

# Physics constants
GRAVITY = 9.8 * 6
CART_MASS = 1.0
POLE_MASS = 1.0
TOTAL_MASS = CART_MASS + POLE_MASS
HALF_POLE_LENGTH = POLE_LENGTH / 2.0
FORCE_MAG = 100.0
TAU = 0.02  # Time interval for updates

# Read weights and biases
best = 0
layershapes, all_weights, all_biases = readWeightsAndBiasesAll()
networks = [{'weights': all_weights[best], 'biases': all_biases[best]}]
converted_all_weights = convert_weights(all_weights, layershapes)

# Function to update the game state
def update_game_state(force):
    
    global cart_x, cart_vx, pole_angle, pole_angular_velocity

    # Physics calculations based on the C++ version
    cosTheta = np.cos(pole_angle)
    sinTheta = np.sin(pole_angle)

    temp = (force + POLE_MASS * HALF_POLE_LENGTH * pole_angular_velocity ** 2 * sinTheta) / TOTAL_MASS
    angular_acceleration = (GRAVITY * sinTheta - cosTheta * temp) / \
                           (HALF_POLE_LENGTH * (4.0 / 3.0 - POLE_MASS * cosTheta ** 2 / TOTAL_MASS))
    linear_acceleration = temp - POLE_MASS * HALF_POLE_LENGTH * angular_acceleration * cosTheta / TOTAL_MASS

    cart_vx += linear_acceleration * TAU
    cart_x += cart_vx * TAU
    pole_angular_velocity += angular_acceleration * TAU
    pole_angle += pole_angular_velocity * TAU


network_display_left = pygame.Surface((NETWORK_DISPLAY_WIDTH, SCREEN_HEIGHT))
net_displays = [network_display_left]
net_locations = [(0,0)]

# Main game loop
running = True

def normalize_state(state):
    normalized_state = np.empty_like(state)
    # Assuming state is [cart_position, cart_velocity, pole_angle, pole_velocity_at_tip]
    # Replace these with the actual ranges for your environment
    min_values = [-12.4, -50, -12, -500]  # Replace with actual min values
    max_values = [12.4, 50, 12, 500]      # Replace with actual max values

    for i in range(len(state)):
        normalized_state[i] = (state[i] - min_values[i]) / (max_values[i] - min_values[i])

    return normalized_state

while running:
    screen.fill(BLACK)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get action from neural network
    state = [cart_x, cart_vx, pole_angle, pole_angular_velocity]
    force = get_actions_cart(normalize_state(state), networks[0]['weights'], networks[0]['biases'])
    #print("Force = ", force)

    #if abs(force) > FORCE_MAG:
    force = FORCE_MAG if force[0] > force[1] else -FORCE_MAG

    print(force)
    print(state)
    # Update game state
    update_game_state(force)

    # Network visualization
    # activations = calculate_activations(networks[0]['weights'], networks[0]['biases'], state, layershapes, numLayers)
    # display_activations(activations, converted_all_weights[0], network_display_left)
    # screen.blit(network_display_left, net_locations[0])

    # Draw the cart
    cart_rect = pygame.Rect(NETWORK_DISPLAY_WIDTH + cart_x - CART_WIDTH // 2 + SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, CART_WIDTH, CART_HEIGHT)
    pygame.draw.rect(screen, WHITE, cart_rect)

    # Calculate and draw the pole
    pole_end_x = SCREEN_WIDTH // 2 + NETWORK_DISPLAY_WIDTH+ cart_x + HALF_POLE_LENGTH * np.sin(pole_angle)
    pole_end_y = SCREEN_HEIGHT // 2 - HALF_POLE_LENGTH * np.cos(pole_angle)
    pygame.draw.line(screen, RED, (SCREEN_WIDTH // 2 + NETWORK_DISPLAY_WIDTH + cart_x, SCREEN_HEIGHT // 2), (pole_end_x, pole_end_y), POLE_WIDTH)

    # Update the display
    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()