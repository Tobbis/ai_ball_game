import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Move the Ball Game")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Ball settings
ball_radius = 20
ball_x = ball_radius
ball_y = HEIGHT - ball_radius
ball_speed = 5

# Starting the main loop
running = True
start_moving = False

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                start_moving = True

    if start_moving:
        ball_x += ball_speed

    if ball_x - ball_radius > WIDTH:  # Ball out of screen
        print("Game Over")
        running = False

    # Clear screen
    screen.fill(BLACK)

    # Draw ball
    pygame.draw.circle(screen, WHITE, (ball_x, ball_y), ball_radius)

    # Update display
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(60)

pygame.quit()
sys.exit()
