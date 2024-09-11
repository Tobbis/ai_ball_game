import pygame
import sys
import time

from config import (
    GAME_HEIGHT,
    FIELD_HEIGHT,
    WIDTH,
    HEIGHT,
    WHITE,
    BLACK,
    GREEN,
    RED,
    BLUE,
    GRAY,
    Button,
    Ball,
    Rectangle,
    check_collision,
    draw_button,
)

# Initialize Pygame
pygame.init()

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Move the Ball Game")

gravity = 1

# Setting up fonts
font = pygame.font.Font(None, 36)
score_font = pygame.font.Font(None, 48)  # Larger font for the score

# Flags to track game states
start_moving = False
is_jumping = False

# Score settings
score = 0
start_time = None  # To store the start time of the game


def display_score(screen, score):
    """Display the score at the top center of the screen."""
    score_surface = score_font.render(f"Score: {score}", True, WHITE)
    score_rect = score_surface.get_rect(
        center=(WIDTH // 2, 30)
    )  # Top center of the screen
    screen.blit(score_surface, score_rect)


# Starting the main loop
ball = Ball(
    radius=20,
    speed=5,
    gravity=1,
    jump_strength=-15,
    start_x=20,
    start_y=GAME_HEIGHT - 20,
    ground_y=GAME_HEIGHT,
)
dragging = False
square = Rectangle(size=50, x=100, y=(GAME_HEIGHT + (FIELD_HEIGHT - 50) // 2))
button = Button(
    width=150, height=50, x=WIDTH - 150 - 10, y=10, color=BLUE, text="Start Game"
)
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
                    square.x <= mouse_x <= square.x + square.size
                    and square.y <= mouse_y <= square.y + square.size
                ):
                    dragging = True
                    offset_x = (
                        mouse_x - square.x
                    )  # Store the offset of where the square was clicked
                    offset_y = mouse_y - square.y

            # Start the game when the start button is clicked
            if (button.x <= mouse_x <= button.x + button.width) and (
                button.y <= mouse_y <= button.y + button.height
            ):
                start_moving = True
                # Only apply gravity if the square is within the game area
                if square.y < GAME_HEIGHT:
                    square.falling = True  # Start applying gravity to the square
                start_time = time.time()  # Record the time when the game starts

        elif event.type == pygame.MOUSEBUTTONUP:
            dragging = False  # Stop dragging when the mouse button is released

        elif event.type == pygame.MOUSEMOTION:
            if (
                dragging and not start_moving
            ):  # Only allow dragging if the game hasn't started
                # Update the square's position based on the mouse movement
                mouse_x, mouse_y = event.pos
                square.x = mouse_x - offset_x
                square.y = mouse_y - offset_y

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if not ball.is_jumping:  # Only allow jump if not already jumping
                    ball.velocity_y = ball.jump_strength
                    ball.is_jumping = True

    if start_moving:
        ball.updatePosition()

        # Change direction if ball hits the walls
        if ball.x + ball.radius >= WIDTH or ball.x - ball.radius <= 0:
            ball.changeDirection()

        # Update the score based on elapsed time
        if start_time is not None:
            elapsed_time = time.time() - start_time
            score = int(elapsed_time)  # Score is the elapsed time in seconds

        # Check if the ball has landed in the game area
        ball.checkCollisionWithGround()

    # Apply gravity to the square if it should be falling
    square.applyGravity(gravity, GAME_HEIGHT)

    # Check for collision between the ball and the square
    if check_collision(ball, square):
        print("Game Over: Ball hit the square!, score: ", score)
        running = False

    # Clear screen
    screen.fill(BLACK)

    # Draw the empty field at the bottom
    pygame.draw.rect(screen, GREEN, (0, GAME_HEIGHT, WIDTH, FIELD_HEIGHT))

    # Draw button
    draw_button(screen, button, font, pygame)

    # Draw ball (within the game area)
    pygame.draw.circle(screen, WHITE, (ball.x, ball.y), ball.radius)

    # Draw the draggable square in the empty field
    pygame.draw.rect(screen, RED, (square.x, square.y, square.size, square.size))

    # Display score
    display_score(screen, score)

    # Update display
    pygame.display.flip()

    # Cap the frame rate
    pygame.time.Clock().tick(60)

pygame.quit()
sys.exit()
