import pygame
import numpy as np
import math

# Define constants
MAX_SPEED = 4
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

# Define neural network parameters
INPUT_SIZE = 6
HIDDEN_SIZE = 16
OUTPUT_SIZE = 2

#for positive
net_weights = np.array([[-0.519704, -0.801828, 0.473642, 0.664188, 0.334657, -0.719401, -0.018348, -0.023191, -0.270057, -0.431413, 0.541139, 0.072981, 0.351714, 0.090951, -0.207317, -0.581136, -0.045423, 0.066747, -0.812781, 0.134486, 0.279423, -0.416530, 0.275623, -0.474039, 0.546981, -0.329481, 0.367612, 0.006809, 0.397863, -1.095994, 0.186176, 1.113712, -0.278238, -0.259126, -0.296788, 0.139922, -0.700306, 0.899908, 0.132985, 0.200383, 0.373858, -0.277473, -0.086028, 0.107037, 0.092584, -0.801482, 0.215176, -1.584887, -0.388144, 0.061489, 0.120274, 0.331960, 0.348955, -0.108080, -0.399066, 0.982735, 0.249795, 0.763439, 0.009010, -0.374142, 1.273774, -0.728693, 0.981180, 0.306799, 0.266638, 0.205637, 0.141256, 0.341134, -0.560987, -0.986967, 0.630702, 0.227001, -0.208306, -0.392838, -0.267981, -0.525058, 0.338897, 0.414838, -0.413188, 0.424294, -0.863249, -0.203608, 0.116469, -0.005370, -0.479571, 0.505737, 0.667515, 0.381530, 0.303538, 0.002654, 0.604852, 0.194857, 0.297157, -0.264205, -0.012693, 1.407730],
[-0.619436, -0.646383, -0.343519, -0.446769, 0.693755, -0.455367, 0.063978, -0.199947, 0.385381, 0.138884, -0.204611, -0.533810, 0.468824, 0.912168, -0.115374, 0.352618, -0.250492, 0.424284, 0.265325, 0.192568, 0.404290, -0.467935, 0.201967, -0.446485, -0.552501, -0.173691, 0.033250, -0.634615, 0.633513, -0.732647, 0.346276, -0.287222]])
net_biases = np.array([[-0.154424, 0.133971, 0.116384, -0.969797, -0.208387, 0.026762, -0.284883, -0.272963, -0.029627, -0.050748, -0.197646, -0.180465, -0.153984, 0.184094, -0.115819, -0.648868],
[-0.239948, -0.354952]])

#for negative
# net_weights = np.array([[0.015117, -0.233145, 0.564584, 0.230389, 0.249240, -0.141314, 0.447146, -1.001832, 0.055803, 0.071781, -0.178512, 0.226350, -0.203852, 0.322702, -0.044260, -0.145013, -0.654384, 0.221787, -0.095877, -0.450532, -0.257405, -0.003961, -0.309535, -0.342542, -0.274947, -0.175854, -0.218598, 0.142396, -0.349472, -0.357999, 0.279584, 0.199306, -0.999795, 0.179175, 0.083503, 0.592283, 0.449276, 0.196061, -0.322045, 
# 0.120057, 0.134485, -0.237431, -0.149439, 0.274101, -0.474982, -0.096870, 0.032740, -0.005853, 0.049134, 1.097407, 0.182878, -0.544863, -0.091537, -0.678849, 0.095752, -0.221228, 0.208542, 0.790593, -0.369024, 0.109921, 0.113842, -0.526581, 0.118117, 0.085838, 0.142024, 0.635718, -0.065476, 0.044413, -0.404417, -0.150404, 0.018318, 0.395018, -0.491794, 0.243767, -0.163389, -0.384700, 0.259528, 0.243210, 0.619763, -0.270511, -0.316606, -0.045862, -0.072942, 0.095605, -0.489049, -0.112033, -0.145541, 0.124441, -0.246976, -0.613369, 0.132627, 0.426426, 0.246453, 0.052634, -0.185515, 0.607956],
# [0.089379, -0.222147, 0.951040, -0.391178, 0.153133, 0.521572, -0.155621, 0.558928, -0.697466, -0.500053, 0.091420, 0.478614, 0.650572, -0.604868, 0.455762, 0.505915, -0.626548, 0.163310, 0.833307, -0.291224, 0.288987, 0.236333, 0.287896, 0.143542, 0.642473, 0.122345, 0.064658, 0.157284, 0.092806, -0.129169, 0.555340, -0.071321]])
# net_biases = np.array([[-0.474067, 0.175837, -0.404465, 0.624118, -0.834193, 0.191912, 0.008947, -0.438071, -0.507168, -0.307327, 0.107557, 0.583710, 0.557243, -0.056856, -0.243417, 0.279493],
# [0.142135, 0.243701]])


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()

# Define function to query neural network for actions
def get_actions(bot_velx, bot_vely, bot_posx, bot_posy, targetX, targetY):
    # Normalize inputs
    inputs = np.array([bot_velx, bot_vely, bot_posx, bot_posy, targetX, targetY])
    # Reshape weight and bias arrays for hidden layer
    hidden_weights = np.reshape(net_weights[0], (HIDDEN_SIZE, INPUT_SIZE))
    hidden_biases = np.reshape(net_biases[0], (HIDDEN_SIZE, 1))

    # Compute activations for hidden layer
    hidden_activations = np.dot(hidden_weights, inputs.reshape(INPUT_SIZE, 1)) + hidden_biases
    hidden_activations = hidden_activations.astype(float)
    hidden_outputs = np.maximum(0, hidden_activations)

    # Reshape weight and bias arrays for output layer
    output_weights = np.reshape(net_weights[1], (OUTPUT_SIZE, HIDDEN_SIZE))
    output_biases = np.reshape(net_biases[1], (OUTPUT_SIZE, 1))

    # Compute activations for output layer
    output_activations = np.dot(output_weights, hidden_outputs) + output_biases
    output_activations = output_activations.astype(float)
    output = 1 / (1 + np.exp(-output_activations))


    # Compute bot velocity based on output    
    gamestate = (output - 0.5) * MAX_SPEED * 2

    speed = math.hypot(gamestate[0], gamestate[1]);
    if(speed > MAX_SPEED):
        f = MAX_SPEED / speed;
        gamestate[0] *= f;
        gamestate[1] *= f;
    
    return gamestate

# Define initial bot state
# bot_posx = SCREEN_WIDTH/2
# bot_posy = SCREEN_HEIGHT/2
bot_posx = 0
bot_posy = 0
bot_velx = 0
bot_vely = 0

# Define target position
# targetX = SCREEN_WIDTH/3
# targetY = SCREEN_HEIGHT/3
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
    print("bot vel : ", bot_velx, " ", bot_vely, " pos = ", bot_posx, " ", bot_posy)
    bot_posx += bot_velx
    bot_posy += bot_vely

    # Draw bot and target
    screen.fill((255, 255, 255))
    pygame.draw.circle(screen, (255, 0, 0), (int(targetX) + SCREEN_WIDTH / 2, int(targetY) + SCREEN_WIDTH / 2), 8)
    pygame.draw.circle(screen, (0, 0, 255), (int(bot_posx) + SCREEN_WIDTH / 2, int(bot_posy) + SCREEN_WIDTH / 2), 8)

    # Update display
    pygame.display.update()
    clock.tick(30)
