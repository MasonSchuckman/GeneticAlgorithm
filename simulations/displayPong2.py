import pygame
import random
import numpy as np
import math
import struct

#nvcc -rdc=true -lineinfo -o runner .\biology\Genome.cpp .\Simulator.cu .\Runner.cu .\Kernels.cu .\simulations\BasicSimulation.cu .\simulations\TargetSimulation.cu .\simulations\MultibotSimulation.cu .\simulations\AirHockeySimulation.cu .\simulations\PongSimulation.cu

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

# Initialize Pygame
pygame.init()

# Set up the game window
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Pong")





def display_activations(activations, weights, loc):
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


# Define paddle and ball dimensions
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 50
BALL_SIZE = 10

# Define paddle and ball speeds
PADDLE_SPEED = 5
BALL_SPEED = 6
SPEED_UP_RATE = 1.00
# Define game colors
BLACK = (50, 50, 0)
WHITE = (255, 255, 255)

# Define game state variables
ball_x = SCREEN_WIDTH // 2
ball_y = SCREEN_HEIGHT // 2
ball_vx = random.choice([-BALL_SPEED, BALL_SPEED])
ball_vy = 0 #random.uniform(-BALL_SPEED, BALL_SPEED)
left_paddle_x = PADDLE_WIDTH / 2
left_paddle_y = SCREEN_HEIGHT // 2
right_paddle_x = PADDLE_WIDTH / 2 + SCREEN_WIDTH - PADDLE_WIDTH
right_paddle_y = SCREEN_HEIGHT // 2



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


best = 0
# Load bot networks
layershapes, all_weights, all_biases = readWeightsAndBiasesAll()
networks = [{'weights': all_weights[best], 'biases': all_biases[best]},
            {'weights': all_weights[best + 1], 'biases': all_biases[best + 1]}]
#print(all_weights)
converted_all_weights = convert_weights(all_weights, layershapes)



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

def get_actions_pong(state, net_weights, net_biases):
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
    #print(output)
    gamestate = [None] * 2

    
    gamestate[0] = min(1, max(-1, output[0])) * PADDLE_SPEED

    
    #print(gamestate)
    
    return gamestate
net_locations = [(200,0), (500,0)]
scores = [0,0]
# Main game loop
running = True
while running:
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Update paddle positions using neural networks
    for i in range(2):
        state = []

        if i == 0:
            state = [abs(ball_x - left_paddle_x) / SCREEN_WIDTH, ball_y / SCREEN_HEIGHT, ball_vx / BALL_SPEED, ball_vy / BALL_SPEED]  # Ball state
        else:
            state = [abs(ball_x - right_paddle_x) / SCREEN_WIDTH, ball_y / SCREEN_HEIGHT, -ball_vx / BALL_SPEED, ball_vy / BALL_SPEED]  # Ball state

            
        if i == 0:
            state += [left_paddle_y / SCREEN_HEIGHT, 0]  # Paddle positions
        else:
            state += [right_paddle_y / SCREEN_HEIGHT, 0]  # Paddle positions
        
        #state = [305.000000, 240.000000, -5.000000, 0.000000, 5.000000, 455.000000, 635.000000, 455.000000]

        #print(state)
        acceleration = get_actions_pong(state, networks[i]['weights'], networks[i]['biases'])
        
        if i == 0:
            left_paddle_y += acceleration[0]
        else:
            right_paddle_y += acceleration[0]

        # Keep paddles within screen boundaries
        if left_paddle_y < 0:
            left_paddle_y = 0
        elif left_paddle_y > SCREEN_HEIGHT - PADDLE_HEIGHT:
            left_paddle_y = SCREEN_HEIGHT - PADDLE_HEIGHT
        if right_paddle_y < 0:
            right_paddle_y = 0
        elif right_paddle_y > SCREEN_HEIGHT - PADDLE_HEIGHT:
            right_paddle_y = SCREEN_HEIGHT - PADDLE_HEIGHT

    

    # update game state
    ball_x += ball_vx
    ball_y += ball_vy

    if ball_x - BALL_SIZE <= left_paddle_x + PADDLE_WIDTH and ball_y >= left_paddle_y and ball_y <= left_paddle_y + PADDLE_HEIGHT and ball_vx < 0:
        ball_vx = -ball_vx * SPEED_UP_RATE
        ball_vy += (ball_y - left_paddle_y - PADDLE_HEIGHT / 2) / (PADDLE_HEIGHT / 2) * BALL_SPEED
        ball_x += ball_vx
        ball_y += ball_vy
    if ball_x + BALL_SIZE >= right_paddle_x and ball_y >= right_paddle_y and ball_y <= right_paddle_y + PADDLE_HEIGHT and ball_vx > 0:
        ball_vx = -ball_vx * SPEED_UP_RATE
        ball_vy += (ball_y - right_paddle_y - PADDLE_HEIGHT / 2) / (PADDLE_HEIGHT / 2) * BALL_SPEED
        ball_x += ball_vx
        ball_y += ball_vy
    if ball_y - BALL_SIZE < 0 or ball_y + BALL_SIZE > SCREEN_HEIGHT:
        ball_vy = -ball_vy
    
    if ball_x < 0 or ball_x > SCREEN_WIDTH:
        if ball_x < 0:
            scores[1] += 1
        else:
            scores[0] += 1
        ball_x = SCREEN_WIDTH // 2
        ball_y = SCREEN_HEIGHT // 2
        ball_vx = random.choice([-BALL_SPEED, BALL_SPEED])
        ball_vy = random.uniform(-BALL_SPEED, BALL_SPEED)
        right_paddle_y = SCREEN_HEIGHT // 2
        left_paddle_y = SCREEN_HEIGHT // 2

    

    # draw game objects
    screen.fill(BLACK)
    
    #draw neural net
    for i in range(2):
        activations_left = calculate_activations(networks[i]['weights'], networks[i]['biases'], state)
        weights_left = converted_all_weights[i]
        display_activations(activations_left, weights_left, net_locations[i])

    pygame.draw.rect(screen, WHITE, (left_paddle_x, left_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.rect(screen, WHITE, (right_paddle_x, right_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.circle(screen, WHITE, (int(ball_x), int(ball_y)), BALL_SIZE)
    
    # Draw the scores
    font = pygame.font.Font(None, 36)
    score1_text = font.render("" + str(scores[0]), True, WHITE)
    score2_text = font.render("" + str(scores[1]), True, WHITE)
    score1_rect = score1_text.get_rect()
    score2_rect = score2_text.get_rect()
    spacing = 20
    score1_rect.midtop = (SCREEN_WIDTH // 2 - spacing, 10)
    score2_rect.midtop = (SCREEN_WIDTH // 2 + spacing, 10)
    screen.blit(score1_text, score1_rect)
    screen.blit(score2_text, score2_rect)
    pygame.display.flip()

    

    



    # update the display
    pygame.display.update()
    clock.tick(72)

# quit Pygamew
pygame.quit()