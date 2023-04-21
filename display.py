import pygame
import numpy as np
import math

# Define constants
MAX_SPEED = 10
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

# Define neural network parameters
layershapes = [6,8,2]

net_weights = np.array([[-0.191083, 0.079841, -0.127740, -0.039663, 0.137477, 0.087965, -0.479188, -0.089636, 0.237234, -0.153124, -0.238110, -0.072986, -0.295682, 0.303986, 0.135147, -0.468184, -0.168100, -0.212201, -0.145062, 0.262820, 0.226249, -0.206955, -0.261188, 0.539418, -0.717928, 0.295275, -0.269030, -0.627511, 0.139830, 0.332173, -0.236078, 0.028958, 0.196836, 0.206497, 0.128855, -0.266095, -0.208614, 0.193854, 0.319410, -0.563935, 0.348045, -0.267320, 0.316727, 0.648191, -0.479277, -0.371274, 0.001850, 0.073868],
[0.149054, 0.147478, 0.227007, -0.127284, 0.211191, 0.113938, -0.181851, 0.218462, -0.165197, -0.110061, 0.111662, -0.057198, 0.035173, -0.051007, -0.193473, -0.089472]])
net_biases = np.array([[-0.427300, 0.654512, 0.085583, 0.007135, -0.110449, -0.025819, -0.003751, -0.197740],
[-0.145017, 0.077235]])
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
    if layer != len(layershapes) - 1:
        output[output < 0] = 0
    else:
        #print('sigmoid')
        output[:] = 1.0 / (1.0 + np.exp(-output))
        

    return output

numLayers = len(layershapes)
# Define function to query neural network for actions
def get_actions(bot_velx, bot_vely, bot_posx, bot_posy, targetX, targetY):
    inputs = np.array([bot_velx, bot_vely, bot_posx, bot_posy, targetX, targetY])
    prevLayer = inputs
    numHiddenLayers = len(layershapes) - 2
    hidden_outputs = [None] * numHiddenLayers
    #forward prop for the hidden layers
    for i in range(numHiddenLayers):
        #print("iter: ", i)
        hidden_outputs[i] = forward_propagation(prevLayer, net_weights[i], net_biases[i], layershapes[i], layershapes[i + 1],  i)
        prevLayer = hidden_outputs[i]

    output = forward_propagation(prevLayer, net_weights[numLayers - 2], net_biases[numLayers - 2], layershapes[numLayers - 2], layershapes[numLayers - 1], numLayers - 1)

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

targetX = 0
targetY = 0

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
    clock.tick(40)
