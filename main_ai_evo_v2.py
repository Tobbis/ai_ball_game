import pygame
import sys
import time
import numpy as np
from nn_evolutionary_agent import EvolvableAgent  # New EvolvableAgent class
import random
import torch
import os
import glob
import argparse

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

# Advanced EA Parameters
population_size = 50
generations = 100
mutation_rate = 0.05
max_score = 2000


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

        if (
            score > max_score
        ):  # Arbitrary large number to stop the game if it takes too long
            break

    return score


def select_best_agents(population, fitness_scores, num_best):
    sorted_population = [
        agent
        for _, agent in sorted(
            zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True
        )
    ]
    return sorted_population[:num_best]


def evolve():
    population = [
        EvolvableAgent(state_size, action_size) for _ in range(population_size)
    ]

    for generation in range(generations):
        fitness_scores = [evaluate_agent(agent) for agent in population]
        best_agents = select_best_agents(
            population, fitness_scores, population_size // 4
        )

        # Elitism: Keep the best agents for the next generation
        next_generation = best_agents

        # Fill the rest of the next generation with children
        while len(next_generation) < population_size:
            parent1, parent2 = random.sample(best_agents, 2)  # Parent selection
            child = parent1.crossover(parent2)
            child.mutate(mutation_rate)
            next_generation.append(child)

        population = next_generation

        best_agent = best_agents[0]
        best_score = max(fitness_scores)
        if generation % 5 == 0:
            print(f"Generation {generation}: Best Score = {best_score}")
            # store the best agent in a file in evo_training directory
            torch.save(
                best_agent.model, f"evo_training/best_agent_gen_{generation}.pth"
            )

        if best_score >= max_score:
            print(f"Max score {best_score} reached in generation {generation}")
            torch.save(
                best_agent.model, f"evo_training/best_agent_gen_{generation}_final.pth"
            )
            break

    return best_agent


def find_best_agent_from_file(folder_path):
    """
    Find all files in the folder that contains the given prefix.

    Parameters:
    folder_path (str): The path to the folder containing the files.
    file_prefix (str): The prefix of the files to be found.

    Returns:
    list: A list of file paths.
    """
    # Create a pattern to match all files with the specified prefix
    file_pattern = os.path.join(folder_path, f"*final.pth")
    file_paths = glob.glob(file_pattern)

    return file_paths


def load_best_agent(file_path):
    """
    Load the best agent from the file.

    Parameters:
    file_path (str): The path to the file containing the best agent.

    Returns:
    EvolvableAgent: The best agent loaded from the file.
    """
    return torch.load(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Choose whether to load model from file or not."
    )
    parser.add_argument(
        "--load", "-l", action="store_true", help="Load model from file if set"
    )
    args = parser.parse_args()
    loadFromFile = args.load
    print(f"Load best model from file: {loadFromFile}")

    if loadFromFile:
        best_model_file_paths = find_best_agent_from_file("evo_training")
        print(f"Best model file path: {best_model_file_paths[0]}")
        best_model = load_best_agent(best_model_file_paths[0])
        best_agent = EvolvableAgent(state_size, action_size)
        best_agent.model = best_model
    else:
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

        if score > 2000:  # Arbitrary large number to stop the game if it takes too long
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
