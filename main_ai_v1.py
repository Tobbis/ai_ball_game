import pygame
import sys
import time
import numpy as np
from agent import DQNAgent
import cProfile
import pstats
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
    Button,
    Ball,
    Rectangle,
    check_collision,
    ACTION_NOT_JUMP,
    ACTION_JUMP,
)


# Initialize Pygame
pygame.init()

# Setting up fonts
font = pygame.font.Font(None, 36)
score_font = pygame.font.Font(None, 48)  # Larger font for the score

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Move the Ball Game")


def display_score(screen, score):
    """Display the score at the top center of the screen."""
    score_surface = score_font.render(f"Score: {score}", True, WHITE)
    score_rect = score_surface.get_rect(
        center=(WIDTH // 2, 30)
    )  # Top center of the screen
    screen.blit(score_surface, score_rect)


def getDistanceToObstacle(ball, square):
    return abs(square.x - ball.x)


# Square settings (Obstacle)
square = Rectangle(size=50, x=300, y=GAME_HEIGHT - 50)
# Button settings
button = Button(
    width=150, height=50, x=WIDTH - 150 - 10, y=10, color=BLUE, text="Start Game"
)

# Setting up fonts
font = pygame.font.Font(None, 36)

# Agent setup
# state_size = 3  # Ball's horizontal position, vertical position, and obstacle's horizontal position
state_size = 3  # ball.x, ball.speed, square.x
action_size = 2  # Jump or no jump
agent = DQNAgent(state_size, action_size)


# Game loop
def game_loop():

    def restart_game():
        ball.reset()
        ball.start(speed=5)
        # later we could random the place of the rectangle (X between 100 to 500)
        return

    # Ball settings
    ball = Ball(
        radius=20,
        speed=5,
        gravity=1,
        jump_strength=-15,
        start_x=20,
        start_y=GAME_HEIGHT - 20,
        ground_y=GAME_HEIGHT,
    )
    start_time = time.time()

    done = False
    reward = 0
    score = 0
    showScreen = False  # only shown sometimes during training
    num_games_played = 0
    distanceToObject = getDistanceToObstacle(ball, square)
    num_jumps = 0
    while True:
        reward = 0
        # State includes the ball's horizontal and vertical positions, and the obstacle's horizontal position
        # state = np.array([ball.x, ball.y, square.x])
        # state = np.array([distanceToObject])
        state = np.array([ball.x, ball.speed, square.x])

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # check if it is valid to jump
        valid_actions = [ACTION_NOT_JUMP, ACTION_JUMP]
        if ball.is_jumping:
            valid_actions = [ACTION_NOT_JUMP]

        # Agent decides action (0: no jump, 1: jump)
        action = agent.act(state, valid_actions)

        if action == 1:
            num_jumps += 1
            reward -= 1  # Small negative reward for jumping

        ball.jump(action)
        ball.updatePosition()
        ball.checkCollisionWithGround()

        # change direction of the ball if it hits the wall
        if ball.x + ball.radius >= WIDTH or ball.x - ball.radius <= 0:
            ball.changeDirection()
            # gets reward for reaching a wall
            reward += 20

        # Collision with square (obstacle)
        if check_collision(ball, square):
            reward = -100  # Large negative reward for hitting the obstacle
            done = True

        # Give a reward for surviving one step
        reward += 1  # Reward for each frame survived
        score += 1

        # Agent learns by storing the state, action, reward, next state, and whether the episode is done
        # distanceToObject = getDistanceToObstacle(ball, square)
        # next_state = np.array([ball.x, ball.y, square.x])  # Next state
        # next_state = np.array([distanceToObject])  # Next state
        next_state = np.array([ball.x, ball.speed, square.x])

        agent.remember(state, action, reward, next_state, done)
        agent.replay()  # Train the agent with the experience

        # Reset if the game is over
        if done:
            restart_game()
            epsilon = agent.getEpsilon()
            print(
                f"Game {num_games_played}: Score: {score}, Jumps: {num_jumps}, Epsilon: {epsilon}, time: {time.time() - start_time}"
            )
            score = 0
            done = False
            num_games_played += 1
            num_jumps = 0
            # print("Time taken: ", time.time() - start_time)
            start_time = time.time()

        # print("score: ", score)

        if num_games_played % 10 == 0:
            showScreen = True
        else:
            showScreen = False

        # Save the model after a certain number of games
        if num_games_played % 10 == 0:  # Save every 10 games
            file_name = "training/dqn_model_game" + str(num_games_played) + ".pth"
            agent.save(file_name)

        if num_games_played == 151:
            break

        if showScreen:
            # Clear screen
            screen.fill(BLACK)

            # Draw the empty field at the bottom
            pygame.draw.rect(screen, GREEN, (0, GAME_HEIGHT, WIDTH, FIELD_HEIGHT))

            # Draw ball (within the game area)
            pygame.draw.circle(screen, WHITE, (ball.x, ball.y), ball.radius)

            # Draw the square obstacle in the game area
            pygame.draw.rect(
                screen, RED, (square.x, square.y, square.size, square.size)
            )

            # Display score
            display_score(screen, score)

            # Update display
            pygame.display.flip()

            # Cap the frame rate
            pygame.time.Clock().tick(60)


if __name__ == "__main__":
    #   cProfile.run("game_loop()")
    # cProfile.run("game_loop()", "profile_output.prof")

    # Load the profiling data from the file
    #    p = pstats.Stats('profile_output.prof')

    game_loop()
