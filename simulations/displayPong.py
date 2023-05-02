import pygame
import random

# initialize Pygame
pygame.init()

# set up the game window
SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("Pong")

# define paddle and ball dimensions
PADDLE_WIDTH = 10
PADDLE_HEIGHT = 50
BALL_SIZE = 10

# define paddle and ball speeds
PADDLE_SPEED = 5
BALL_SPEED = 5

# define game colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# define game state variables
ball_x = SCREEN_WIDTH // 2
ball_y = SCREEN_HEIGHT // 2
ball_vx = random.choice([-BALL_SPEED, BALL_SPEED])
ball_vy = random.uniform(-BALL_SPEED, BALL_SPEED)
left_paddle_x = 0
left_paddle_y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2
right_paddle_x = SCREEN_WIDTH - PADDLE_WIDTH
right_paddle_y = SCREEN_HEIGHT // 2 - PADDLE_HEIGHT // 2

# main game loop
running = True
while running:
    # handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # handle player input
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w] and left_paddle_y > 0:
        left_paddle_y -= PADDLE_SPEED
    if keys[pygame.K_s] and left_paddle_y < SCREEN_HEIGHT - PADDLE_HEIGHT:
        left_paddle_y += PADDLE_SPEED
    if keys[pygame.K_UP] and right_paddle_y > 0:
        right_paddle_y -= PADDLE_SPEED
    if keys[pygame.K_DOWN] and right_paddle_y < SCREEN_HEIGHT - PADDLE_HEIGHT:
        right_paddle_y += PADDLE_SPEED

    # update game state
    ball_x += ball_vx
    ball_y += ball_vy
    if ball_x - BALL_SIZE <= left_paddle_x + PADDLE_WIDTH and ball_y >= left_paddle_y and ball_y <= left_paddle_y + PADDLE_HEIGHT and ball_vx < 0:
        ball_vx = -ball_vx
        ball_vy += (ball_y - left_paddle_y - PADDLE_HEIGHT / 2) / (PADDLE_HEIGHT / 2) * BALL_SPEED
    if ball_x + BALL_SIZE >= right_paddle_x and ball_y >= right_paddle_y and ball_y <= right_paddle_y + PADDLE_HEIGHT and ball_vx > 0:
        ball_vx = -ball_vx
        ball_vy += (ball_y - right_paddle_y - PADDLE_HEIGHT / 2) / (PADDLE_HEIGHT / 2) * BALL_SPEED
    if ball_y - BALL_SIZE < 0 or ball_y + BALL_SIZE > SCREEN_HEIGHT:
        ball_vy = -ball_vy

    # draw game objects
    screen.fill(BLACK)
    pygame.draw.rect(screen, WHITE, (left_paddle_x, left_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.rect(screen, WHITE, (right_paddle_x, right_paddle_y, PADDLE_WIDTH, PADDLE_HEIGHT))
    pygame.draw.circle(screen, WHITE, (int(ball_x), int(ball_y)), BALL_SIZE)

    # update the display
    pygame.display.update()
    clock.tick(60)

# quit Pygamew
pygame.quit()