import pygame
import sys
import time

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
RED = (255, 0, 0)  # Color for the draggable square

# Ball settings
ball_radius = 20
ball_x = ball_radius
ball_y = GAME_HEIGHT - ball_radius  # Ball will only move in the game area
ball_speed = 5  # Horizontal speed
ball_velocity_y = 0
gravity = 1
jump_strength = -15

# Square settings
square_size = 50
square_x = 100  # Initial position of the square
square_y = (
    GAME_HEIGHT + (FIELD_HEIGHT - square_size) // 2
)  # Position within the empty field
square_velocity_y = 0  # Vertical velocity for the square
falling = False  # To track if the square should fall due to gravity
dragging = False  # Flag to check if the square is being dragged

# Button settings
button_width = 150
button_height = 50
button_x = WIDTH - button_width - 10
button_y = 10
button_color = BLUE
button_text = "Start Game"

# Setting up fonts
font = pygame.font.Font(None, 36)
score_font = pygame.font.Font(None, 48)  # Larger font for the score

# Flags to track game states
start_moving = False
is_jumping = False

# Score settings
score = 0
start_time = None  # To store the start time of the game


def draw_button(screen, x, y, width, height, color, text):
    pygame.draw.rect(screen, color, (x, y, width, height))
    text_surface = font.render(text, True, WHITE)
    text_rect = text_surface.get_rect(center=(x + width // 2, y + height // 2))
    screen.blit(text_surface, text_rect)


def check_collision(ball_x, ball_y, ball_radius, square_x, square_y, square_size):
    """Check if the ball has collided with the square."""
    # Simple bounding box collision detection
    if (square_x <= ball_x <= square_x + square_size) and (
        square_y <= ball_y <= square_y + square_size
    ):
        return True
    return False


def display_score(screen, score):
    """Display the score at the top center of the screen."""
    score_surface = score_font.render(f"Score: {score}", True, WHITE)
    score_rect = score_surface.get_rect(
        center=(WIDTH // 2, 30)
    )  # Top center of the screen
    screen.blit(score_surface, score_rect)


# Starting the main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            # Only allow dragging if the game hasn't started yet
            if not start_moving:
                # Check if the mouse is over the square when clicked
                if (
                    square_x <= mouse_x <= square_x + square_size
                    and square_y <= mouse_y <= square_y + square_size
                ):
                    dragging = True
                    offset_x = (
                        mouse_x - square_x
                    )  # Store the offset of where the square was clicked
                    offset_y = mouse_y - square_y

            # Start the game when the start button is clicked
            if (button_x <= mouse_x <= button_x + button_width) and (
                button_y <= mouse_y <= button_y + button_height
            ):
                start_moving = True
                falling = True  # Start applying gravity to the square
                start_time = time.time()  # Record the time when the game starts

        elif event.type == pygame.MOUSEBUTTONUP:
            dragging = False  # Stop dragging when the mouse button is released

        elif event.type == pygame.MOUSEMOTION:
            if (
                dragging and not start_moving
            ):  # Only allow dragging if the game hasn't started
                # Update the square's position based on the mouse movement
                mouse_x, mouse_y = event.pos
                square_x = mouse_x - offset_x
                square_y = mouse_y - offset_y

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

        # Update the score based on elapsed time
        if start_time is not None:
            elapsed_time = time.time() - start_time
            score = int(elapsed_time)  # Score is the elapsed time in seconds

    # Apply gravity if ball is jumping
    if is_jumping:
        ball_velocity_y += gravity
        ball_y += ball_velocity_y

        # Check if the ball has landed in the game area
        if ball_y >= GAME_HEIGHT - ball_radius:
            ball_y = GAME_HEIGHT - ball_radius
            ball_velocity_y = 0
            is_jumping = False

    # Apply gravity to the square if it should be falling
    if falling:
        square_velocity_y += gravity
        square_y += square_velocity_y

        # Check if the square has landed on the "ground" (bottom of the game area)
        if square_y + square_size >= GAME_HEIGHT:
            square_y = GAME_HEIGHT - square_size
            square_velocity_y = 0
            falling = False  # Stop applying gravity after it lands

    # Check for collision between the ball and the square
    if check_collision(ball_x, ball_y, ball_radius, square_x, square_y, square_size):
        print("Game Over: Ball hit the square!")
        running = False

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

    # Draw the draggable square in the empty field
    pygame.draw.rect(screen, RED, (square_x, square_y, square_size, square_size))

    # Display score
    display_score(screen, score)

    # Update display
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(60)

pygame.quit()
sys.exit()
