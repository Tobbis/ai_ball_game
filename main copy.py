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
GRAY = (100, 100, 100)

# Ball settings
ball_radius = 20
ball_x = ball_radius
ball_y = HEIGHT - ball_radius
ball_speed = 5

# Button settings
button_width = 150
button_height = 50
button_x = WIDTH - button_width - 10
button_y = 10
button_color = GRAY
button_text = "Start Game"

# Setting up fonts
font = pygame.font.Font(None, 36)

# Flag to track if the game should start
start_moving = False

def draw_button(screen, x, y, width, height, color, text):
    pygame.draw.rect(screen, color, (x, y, width, height))
    text_surface = font.render(text, True, WHITE)
    text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
    screen.blit(text_surface, text_rect)

# Starting the main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            if (button_x <= mouse_x <= button_x + button_width) and (button_y <= mouse_y <= button_y + button_height):
                start_moving = True

    if start_moving:
        ball_x += ball_speed

    if ball_x - ball_radius > WIDTH:  # Ball out of screen
        print("Game Over")
        running = False

    # Clear screen
    screen.fill(BLACK)

    # Draw button
    draw_button(screen, button_x, button_y, button_width, button_height, button_color, button_text)

    # Draw ball
    pygame.draw.circle(screen, WHITE, (ball_x, ball_y), ball_radius)

    # Update display
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(60)

pygame.quit()
sys.exit()
