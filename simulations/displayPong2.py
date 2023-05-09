import pygame
import random
import numpy as np
import math
import struct
from network_visualizer import *




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
NETWORK_DISPLAY_WIDTH = 400
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
screen = pygame.display.set_mode((SCREEN_WIDTH + NETWORK_DISPLAY_WIDTH * 2, SCREEN_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Pong")



# Define paddle and ball dimensions
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 62
BALL_SIZE = 10

# Define paddle and ball speeds
PADDLE_SPEED = 6
BALL_SPEED = 7
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
    gamestate = [None] * 1

    
    gamestate[0] = min(1, max(-1, output[0])) * PADDLE_SPEED

    
    #print(gamestate)
    
    return gamestate

network_display_left = pygame.Surface((NETWORK_DISPLAY_WIDTH, SCREEN_HEIGHT))
network_display_right = pygame.Surface((NETWORK_DISPLAY_WIDTH, SCREEN_HEIGHT))

net_displays = [network_display_left, network_display_right]
net_locations = [(0,0), (SCREEN_WIDTH + NETWORK_DISPLAY_WIDTH, 0)]
scores = [0,0]
# Main game loop
running = True
while running:
   
    screen.fill(BLACK)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # update game state
    ball_x += ball_vx
    ball_y += ball_vy

    if ball_x - BALL_SIZE <= left_paddle_x + PADDLE_WIDTH and ball_y >= left_paddle_y and ball_y <= left_paddle_y + PADDLE_HEIGHT and ball_vx < 0:
        ball_vx = -ball_vx * SPEED_UP_RATE
        ball_vy += (ball_y - left_paddle_y - PADDLE_HEIGHT / 2) / (PADDLE_HEIGHT / 2) * BALL_SPEED
        ball_x += ball_vx * 2
        ball_y += ball_vy
    if ball_x + BALL_SIZE >= right_paddle_x and ball_y >= right_paddle_y and ball_y <= right_paddle_y + PADDLE_HEIGHT and ball_vx > 0:
        ball_vx = -ball_vx * SPEED_UP_RATE
        ball_vy += (ball_y - right_paddle_y - PADDLE_HEIGHT / 2) / (PADDLE_HEIGHT / 2) * BALL_SPEED
        ball_x += ball_vx * 2
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

        #draw neural net
        
        activations_left = calculate_activations(networks[i]['weights'], networks[i]['biases'], state, layershapes, numLayers)
        display_activations(activations_left, converted_all_weights[i], net_displays[i])
        #display_activations2(activations_left, converted_all_weights[i], net_displays[i], SCREEN_HEIGHT)
        screen.blit(net_displays[i], net_locations[i])

    
    
    
    # draw game objects
    pygame.draw.rect(screen, WHITE, (left_paddle_x + NETWORK_DISPLAY_WIDTH, left_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.rect(screen, WHITE, (right_paddle_x + NETWORK_DISPLAY_WIDTH, right_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.circle(screen, WHITE, (int(ball_x + NETWORK_DISPLAY_WIDTH), int(ball_y)), BALL_SIZE)
    
    # Draw the scores
    font = pygame.font.Font(None, 36)
    score1_text = font.render("" + str(scores[0]), True, WHITE)
    score2_text = font.render("" + str(scores[1]), True, WHITE)
    score1_rect = score1_text.get_rect()
    score2_rect = score2_text.get_rect()
    spacing = 20
    score1_rect.midtop = (SCREEN_WIDTH // 2 - spacing + NETWORK_DISPLAY_WIDTH, 10)
    score2_rect.midtop = (SCREEN_WIDTH // 2 + spacing + NETWORK_DISPLAY_WIDTH, 10)
    screen.blit(score1_text, score1_rect)
    screen.blit(score2_text, score2_rect)
    pygame.display.flip()

    

    



    # update the display
    pygame.display.update()
    clock.tick(72)

# quit Pygamew
pygame.quit()