import pygame
import sys
import time
import numpy as np
from simple_evolutionary_agent import EvolvableAgent  # New EvolvableAgent class
import random

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
score_font = pygame.font.Font(None, 48)

# Create screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Move the Ball Game")


def display_score(screen, score):
    score_surface = score_font.render(f"Score: {score}", True, WHITE)
    score_rect = score_surface.get_rect(center=(WIDTH // 2, 30))
    screen.blit(score_surface, score_rect)


def getDistanceToObstacle(ball, square):
    return abs(square.x - ball.x)


# Square settings (Obstacle)
square = Rectangle(size=50, x=300, y=GAME_HEIGHT - 50)
button = Button(
    width=150, height=50, x=WIDTH - 150 - 10, y=10, color=BLUE, text="Start Game"
)

state_size = 3  # ball.x, ball.speed, square.x
action_size = 2  # Jump or no jump

population_size = 20
generations = 50
mutation_rate = 0.01


def evaluate_agent(agent):
    ball = Ball(
        radius=20,
        speed=5,
        gravity=1,
        jump_strength=-15,
        start_x=20,
        start_y=GAME_HEIGHT - 20,
        ground_y=GAME_HEIGHT,
    )
    score = 0
    done = False
    while not done:
        state = np.array([ball.x, ball.speed, square.x])
        action = agent.act(state)

        if action == 1:
            ball.jump(action)
            score -= 1  # Penalize for jumping
        ball.updatePosition()
        ball.checkCollisionWithGround()

        if ball.x + ball.radius >= WIDTH or ball.x - ball.radius <= 0:
            ball.changeDirection()
            score += 20

        if check_collision(ball, square):
            score -= 100
            done = True

        score += 2
        if done:
            break

        if score > 1000:  # Arbitrary large number to stop the game if it takes too long
            break

    return score


def evolve():
    population = [
        EvolvableAgent(state_size, action_size) for _ in range(population_size)
    ]

    for generation in range(generations):
        fitness_scores = [evaluate_agent(agent) for agent in population]
        sorted_population = [
            agent
            for _, agent in sorted(
                zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True
            )
        ]

        next_generation = sorted_population[: population_size // 2]  # Selection

        for i in range(population_size // 2, population_size, 2):
            parent1 = random.choice(sorted_population[: population_size // 4])
            parent2 = random.choice(sorted_population[: population_size // 4])
            child1 = parent1.crossover(parent2)
            child2 = parent2.crossover(parent1)
            child1.mutate(mutation_rate)
            child2.mutate(mutation_rate)
            next_generation += [child1, child2]

        population = next_generation

        best_agent = sorted_population[0]
        best_score = fitness_scores[0]
        if generation % 5 == 0:
            print(f"Generation {generation}: Best Score = {best_score}")

    # At the end, you may want to save the best agent's weights
    return best_agent


if __name__ == "__main__":
    best_agent = evolve()

    # use the best agent to play the game
    ball = Ball(
        radius=20,
        speed=5,
        gravity=1,
        jump_strength=-15,
        start_x=20,
        start_y=GAME_HEIGHT - 20,
        ground_y=GAME_HEIGHT,
    )
    done = False
    score = 0
    while not done:
        state = np.array([ball.x, ball.speed, square.x])
        action = best_agent.act(state)

        if action == 1:
            ball.jump(action)
        ball.updatePosition()
        ball.checkCollisionWithGround()

        if ball.x + ball.radius >= WIDTH or ball.x - ball.radius <= 0:
            ball.changeDirection()
            score += 20

        if check_collision(ball, square):
            score -= 100
            done = True

        score += 1
        if done:
            break

        if score > 1000:  # Arbitrary large number to stop the game if it takes too long
            break

        # display the game
        # Clear screen
        screen.fill(BLACK)

        # Draw the empty field at the bottom
        pygame.draw.rect(screen, GREEN, (0, GAME_HEIGHT, WIDTH, FIELD_HEIGHT))

        # Draw ball (within the game area)
        pygame.draw.circle(screen, WHITE, (ball.x, ball.y), ball.radius)

        # Draw the square obstacle in the game area
        pygame.draw.rect(screen, RED, (square.x, square.y, square.size, square.size))

        # Display score
        display_score(screen, score)

        # Update display
        pygame.display.flip()

        # Cap the frame rate
        pygame.time.Clock().tick(60)
