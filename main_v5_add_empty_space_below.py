import pygame
import sys

# Initialize Pygame
pygame.init()

# Screen dimensions
GAME_HEIGHT = 600  # Height of the game area
FIELD_HEIGHT = 200  # Height of the empty field below the game area
WIDTH, HEIGHT = (
    800,
    GAME_HEIGHT + FIELD_HEIGHT,
)  # Total screen height including the field

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Move the Ball Game")

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)  # Color for the empty field

# Ball settings
ball_radius = 20
ball_x = ball_radius
ball_y = GAME_HEIGHT - ball_radius  # Ball will only move in the game area
ball_speed = 5  # Horizontal speed
ball_velocity_y = 0
gravity = 1
jump_strength = -15

# Button settings
button_width = 150
button_height = 50
button_x = WIDTH - button_width - 10
button_y = 10
button_color = BLUE
button_text = "Start Game"

# Setting up fonts
font = pygame.font.Font(None, 36)

# Flags to track game states
start_moving = False
is_jumping = False


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
            if (button_x <= mouse_x <= button_x + button_width) and (
                button_y <= mouse_y <= button_y + button_height
            ):
                start_moving = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if not is_jumping:  # Only allow jump if not already jumping
                    ball_velocity_y = jump_strength
                    is_jumping = True

    if start_moving:
        ball_x += ball_speed

        # Change direction if ball hits the walls
        if ball_x + ball_radius >= WIDTH or ball_x - ball_radius <= 0:
            ball_speed = -ball_speed  # Reverse direction

    # Apply gravity if ball is jumping
    if is_jumping:
        ball_velocity_y += gravity
        ball_y += ball_velocity_y

        # Check if the ball has landed in the game area
        if ball_y >= GAME_HEIGHT - ball_radius:
            ball_y = GAME_HEIGHT - ball_radius
            ball_velocity_y = 0
            is_jumping = False

    # Clear screen
    screen.fill(BLACK)

    # Draw the empty field at the bottom
    pygame.draw.rect(screen, GREEN, (0, GAME_HEIGHT, WIDTH, FIELD_HEIGHT))

    # Draw button
    draw_button(
        screen,
        button_x,
        button_y,
        button_width,
        button_height,
        button_color,
        button_text,
    )

    # Draw ball (within the game area)
    pygame.draw.circle(screen, WHITE, (ball_x, ball_y), ball_radius)

    # Update display
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(60)

pygame.quit()
sys.exit()
