import pygame
import numpy as np
import math

# Define constants
MAX_SPEED = 10
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

# Define neural network parameters
INPUT_SIZE = 6
HIDDEN_SIZE = 6
OUTPUT_SIZE = 2
net_weights = np.array([[-0.064887, 0.081250, -0.008616, 0.015152, -0.002329, 0.029349, -0.069131, 0.054492, 0.128113, -0.029466, 0.021688, 0.037285, -0.030934, 0.186769, -0.101467, 0.115155, 0.112583, -0.103557, 0.123425, 0.004600, 0.071451, 0.085588, 0.055979, -0.063636, 0.040317, -0.189796, 0.098968, 0.120708, -0.002379, 0.216985, -0.117435, -0.004730, -0.072660, -0.028208, -0.059642, 0.070447],
[-0.011244, -0.109445, -0.083143, 0.058155, 0.035056, -0.081024, -0.023208, 0.006918, -0.095213, -0.021110, 0.141665, 0.013306]])
net_biases = np.array([[0.079239, -0.099714, 0.163010, -0.021708, -0.082598, 0.093382],
[-0.019669, 0.055736]])
# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

def forward_propagation(inputs, weights, biases, input_size, output_size, layer):
    output = np.zeros(output_size)

    weights = np.array(weights).reshape((input_size, output_size))

    # Initialize output to biases
    output[:] = biases

    # Compute dot product of input and weights
    output[:] += np.dot(inputs, weights)

    # Apply activation function (ReLU for non-output layers, sigmoid for output layer)
    if layer != 3 - 2:
        output[output < 0] = 0
    else:
        output[:] = 1.0 / (1.0 + np.exp(-output))
        

    return output

# Define function to query neural network for actions
def get_actions(bot_velx, bot_vely, bot_posx, bot_posy, targetX, targetY):
    inputs = np.array([bot_velx, bot_vely, bot_posx, bot_posy, targetX, targetY])
    
    hidden_outputs = forward_propagation(inputs, net_weights[0], net_biases[0], INPUT_SIZE, HIDDEN_SIZE,  0)
    output = forward_propagation(hidden_outputs, net_weights[1], net_biases[1], HIDDEN_SIZE, OUTPUT_SIZE, 1)

    # Compute bot velocity based on output    
    gamestate = (output - 0.5) * MAX_SPEED * 2

    speed = math.hypot(gamestate[0], gamestate[1]);
    if(speed > MAX_SPEED):
        f = MAX_SPEED / speed;
        gamestate[0] *= f;
        gamestate[1] *= f;
    
    return gamestate

# Define initial bot state

bot_posx = 0
bot_posy = 0
bot_velx = 0
bot_vely = 0

# Define target position

targetX = 10
targetY = 13

# Main game loop
while True:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    # Get actions from neural network
    actions = get_actions(bot_velx, bot_vely, bot_posx, bot_posy, targetX, targetY)

    # Update bot state
    bot_velx = actions[0]
    bot_vely = actions[1]
    #print("bot vel : ", bot_velx, " ", bot_vely, " pos = ", bot_posx, " ", bot_posy)
    bot_posx += bot_velx
    bot_posy += bot_vely

    # Get current mouse position and update target position
    mouse_pos = pygame.mouse.get_pos()
    targetX = mouse_pos[0] - SCREEN_WIDTH / 2
    targetY = mouse_pos[1] - SCREEN_HEIGHT / 2


    # Draw bot and target
    screen.fill((255, 255, 255))
    pygame.draw.circle(screen, (255, 0, 0), (int(targetX) + SCREEN_WIDTH / 2, int(targetY) + SCREEN_HEIGHT / 2), 8)
    pygame.draw.circle(screen, (0, 0, 255), (int(bot_posx) + SCREEN_WIDTH / 2, int(bot_posy) + SCREEN_HEIGHT / 2), 8)

    # Update display
    pygame.display.update()
    clock.tick(30)
