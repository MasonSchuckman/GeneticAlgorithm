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

# Define paddle and ball dimensions
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 50
BALL_SIZE = 10

# Define paddle and ball speeds
PADDLE_SPEED = 5
BALL_SPEED = 9.5
SPEED_UP_RATE = 1.00
# Define game colors
BLACK = (0, 0, 0)
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


best = 0
# Load bot networks
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
            state = [abs(ball_x - 0) / SCREEN_WIDTH, ball_y / SCREEN_HEIGHT, ball_vx / BALL_SPEED, ball_vy / BALL_SPEED]  # Ball state
        else:
            state = [abs(ball_x - SCREEN_WIDTH) / SCREEN_WIDTH, ball_y / SCREEN_HEIGHT, -ball_vx / BALL_SPEED, ball_vy / BALL_SPEED]  # Ball state

            
        if i == 0:
            state += [0, left_paddle_y / SCREEN_HEIGHT, 1, right_paddle_y / SCREEN_HEIGHT]  # Paddle positions
        else:
            state += [0, right_paddle_y / SCREEN_HEIGHT, 1, left_paddle_y / SCREEN_HEIGHT]  # Paddle positions
        
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
    if ball_x + BALL_SIZE >= right_paddle_x and ball_y >= right_paddle_y and ball_y <= right_paddle_y + PADDLE_HEIGHT and ball_vx > 0:
        ball_vx = -ball_vx * SPEED_UP_RATE
        ball_vy += (ball_y - right_paddle_y - PADDLE_HEIGHT / 2) / (PADDLE_HEIGHT / 2) * BALL_SPEED
    if ball_y - BALL_SIZE < 0 or ball_y + BALL_SIZE > SCREEN_HEIGHT:
        ball_vy = -ball_vy
    
    if ball_x < 0 or ball_x > SCREEN_WIDTH:
        ball_x = SCREEN_WIDTH // 2
        ball_y = SCREEN_HEIGHT // 2
        ball_vx = random.choice([-BALL_SPEED, BALL_SPEED])
        ball_vy = random.uniform(-BALL_SPEED, BALL_SPEED)
        right_paddle_y = SCREEN_HEIGHT // 2
        left_paddle_y = SCREEN_HEIGHT // 2

    # draw game objects
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, (left_paddle_x, left_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.rect(screen, WHITE, (right_paddle_x, right_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.circle(screen, WHITE, (int(ball_x), int(ball_y)), BALL_SIZE)

    # update the display
    pygame.display.update()
    clock.tick(72)

# quit Pygamew
pygame.quit()