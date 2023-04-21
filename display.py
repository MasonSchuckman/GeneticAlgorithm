import pygame
import numpy as np
import math

# Define constants
MAX_SPEED = 5
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

# Define neural network parameters
INPUT_SIZE = 6
HIDDEN_SIZE = 8
OUTPUT_SIZE = 2
net_weights = np.array([[6.972132, 1.491231, -0.307194, -0.682142, 5.945843, -0.562506, 3.722331, -1.137602, -3.181457, 3.572584, 0.134673, -0.460242, 0.562078, 4.301032, 5.618152, 0.043433, -3.026530, 4.688155, 9.425539, -4.048689, -1.174929, -2.946634, 0.781662, -6.435290, 5.160612, 2.557968, -6.021100, -0.682038, 6.116041, 3.612711, 2.460966, -7.066083, -4.639160, -1.775885, -12.016842, 4.641423, -4.661957, -0.880314, -3.017948, 1.720020, -2.477640, -6.226961, 6.229924, 3.288722, -3.820956, -5.388313, -3.590342, 8.142702],
[-5.170864, -2.590429, 0.889810, -1.210096, -13.036943, 3.966826, 3.865606, 2.390955, 1.138772, 2.244593, 3.235249, -9.008130, -4.107448, -3.294118, 9.326231, 4.560537]])
net_biases = np.array([[-10.929093, -6.429063, 2.436169, 2.211440, 1.705449, -3.328842, -4.424405, -0.403179],
[3.000554, 1.267764]])
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
