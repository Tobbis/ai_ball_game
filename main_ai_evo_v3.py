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
import json

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


def draw_button(screen, btn):  # x, y, width, height, color, text):
    pygame.draw.rect(screen, btn.color, (btn.x, btn.y, btn.width, btn.height))
    text_surface = font.render(btn.text, True, WHITE)
    text_rect = text_surface.get_rect(
        center=(btn.x + btn.width // 2, btn.y + btn.height // 2)
    )
    screen.blit(text_surface, text_rect)


def getDistanceToObstacle(ball, square):
    return abs(square.x - ball.x)


state_size = 3  # ball.x, ball.speed, square.x
action_size = 2  # Jump or no jump

# Advanced EA Parameters
population_size = 50
generations = 100
mutation_rate = 0.05
max_score = 2000
gravity = 1


def CheckIfSquareIsFalling(square):
    if square.y < GAME_HEIGHT - square.size:
        return True
    return False


def evaluate_agent(agent, lvl_config):
    ball = Ball(
        radius=20,
        speed=5,
        gravity=1,
        jump_strength=-15,
        start_x=20,
        start_y=GAME_HEIGHT - 20,
        ground_y=GAME_HEIGHT,
    )
    square = Rectangle(size=50, x=lvl_config["square_x"], y=lvl_config["square_y"])
    # square could be above ground level
    square.falling = CheckIfSquareIsFalling(square)
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

        # apply gravity to the square if it is not on the ground
        square.applyGravity(gravity, GAME_HEIGHT)

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


def evolve(lvl_config):
    print("Starting evolution... max generation:", generations)
    population = [
        EvolvableAgent(state_size, action_size) for _ in range(population_size)
    ]

    # remove all old files in evo_training directory
    files = glob.glob("evo_training/*")
    for f in files:
        os.remove(f)

    dict_score = {}

    for generation in range(generations):
        fitness_scores = [evaluate_agent(agent, lvl_config) for agent in population]
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
        if best_score not in dict_score:
            dict_score[best_score] = 1
            print(
                f"Generation {generation}: Best Score = {best_score} (new high score)"
            )
            # store the best agent in a file in evo_training directory
            torch.save(
                best_agent.model.state_dict(),
                f"evo_training/best_agent_gen_{generation}_{best_score}.pth",
            )
        else:
            if generation % 5 == 0:
                print(f"Generation {generation}: Best Score = {best_score}")

        if best_score >= max_score:
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
    # find the last file in the folder
    files = glob.glob(f"{folder_path}/*")
    if len(files) == 0:
        return None
    files.sort(key=os.path.getmtime)
    file_path = [files[-1]][0]

    # Create a pattern to match all files with the specified prefix

    # file_pattern = os.path.join(folder_path, f"*final.pth")
    # file_paths = glob.glob(file_pattern)
    return file_path


def load_best_agent(file_path):
    """
    Load the best agent from the file.

    Parameters:
    file_path (str): The path to the file containing the best agent.

    Returns:
    EvolvableAgent: The best agent loaded from the file.
    """

    agent = EvolvableAgent(state_size, action_size)
    agent.model.load_state_dict(torch.load(file_path, weights_only=True))
    return agent


def load_level_config():
    # check if file exists
    if not os.path.exists("level_config.json"):
        return None
    with open("level_config.json", "r") as file:
        level_config = json.load(file)
    return level_config


def createLevel(lvlConfig):
    # user can move the obstacle to the game screen and then press a button to store the lvl config
    # the lvl config will be stored in a file

    # Square settings (Obstacle)
    # square = Rectangle(size=50, x=300, y=GAME_HEIGHT - 50)
    if lvlConfig is None:
        square = Rectangle(size=50, x=100, y=(GAME_HEIGHT + (FIELD_HEIGHT - 50) // 2))
    else:
        square = Rectangle(size=50, x=lvlConfig["square_x"], y=lvlConfig["square_y"])

    # Display text to instruct user to move the rectangle
    font = pygame.font.Font(None, 36)
    text = font.render("Move the rectangle to place obstacle.", True, WHITE)
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))

    # Button settings
    button = Button(
        width=150, height=50, x=WIDTH - 150 - 10, y=10, color=BLUE, text="Start Game"
    )
    ball = Ball(
        radius=20,
        speed=5,
        gravity=1,
        jump_strength=-15,
        start_x=20,
        start_y=GAME_HEIGHT - 20,
        ground_y=GAME_HEIGHT,
    )
    start_training = False
    dragging = False
    # Starting the main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                # Only allow dragging if the game hasn't started yet
                if not start_training:
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
                    running = False
                    break
                    # start_training = True
                    # # Only apply gravity if the square is within the game area
                    # if square_y < GAME_HEIGHT:
                    #     falling = True  # Start applying gravity to the square
                    # start_time = time.time()  # Record the time when the game starts

            elif event.type == pygame.MOUSEBUTTONUP:
                dragging = False  # Stop dragging when the mouse button is released

            elif event.type == pygame.MOUSEMOTION:
                if (
                    dragging and not start_training
                ):  # Only allow dragging if the game hasn't started
                    # Update the square's position based on the mouse movement
                    mouse_x, mouse_y = event.pos
                    square.x = mouse_x - offset_x
                    square.y = mouse_y - offset_y

        # Clear screen
        screen.fill(BLACK)

        # Draw the empty field at the bottom
        pygame.draw.rect(screen, GREEN, (0, GAME_HEIGHT, WIDTH, FIELD_HEIGHT))

        # Draw button
        draw_button(screen, button)

        # Draw ball (within the game area)
        pygame.draw.circle(screen, WHITE, (ball.x, ball.y), ball.radius)

        # Draw the draggable square in the empty field
        pygame.draw.rect(screen, RED, (square.x, square.y, square.size, square.size))

        # display the text
        screen.blit(text, text_rect)

        # Update display
        pygame.display.flip()

        # Cap the frame rate
        pygame.time.Clock().tick(60)

    # Save the level configuration to a file using json
    level_config = {"square_x": square.x, "square_y": square.y}
    with open("level_config.json", "w") as file:
        json.dump(level_config, file)

    return level_config


# def showLevel(lvlConfig):
#     # just display how the level will look like with the ball and square
#     square = Rectangle(size=50, x=lvlConfig["square_x"], y=lvlConfig["square_y"])
#     ball = Ball(
#         radius=20,
#         speed=5,
#         gravity=1,
#         jump_strength=-15,
#         start_x=20,
#         start_y=GAME_HEIGHT - 20,
#         ground_y=GAME_HEIGHT,
#     )
#     # Draw the empty field at the bottom
#     pygame.draw.rect(screen, GREEN, (0, GAME_HEIGHT, WIDTH, FIELD_HEIGHT))

#     # Draw ball (within the game area)
#     pygame.draw.circle(screen, WHITE, (ball.x, ball.y), ball.radius)

#     # Draw the square obstacle in the game area
#     pygame.draw.rect(screen, RED, (square.x, square.y, square.size, square.size))

#     # Update display
#     pygame.display.flip()


def getAllDifferentAgents():
    # get all agents from the evo_training directory
    files = glob.glob("evo_training/*")
    uniqueAgents = []
    dict_scores = {}

    for f in files:
        # get the file name, not path
        name = os.path.basename(f)
        # split the file name by "_"
        parts = name.split("_")
        # the last part is the score (remove the .pth)
        score = int(parts[-1].split(".")[0])
        if score in dict_scores:
            continue
        dict_scores[score] = 1
        # the second last part is the generation
        generation = int(parts[-2])
        agent = EvolvableAgent(state_size, action_size)
        agent.model.load_state_dict(torch.load(f, weights_only=True))

        # add the agent to the list
        uniqueAgents.append((score, generation, agent))
    # sort the agents by score, with the lowest number first
    uniqueAgents.sort(key=lambda x: x[0])
    return uniqueAgents


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Choose whether to load model from file or not."
    )
    parser.add_argument(
        "--load", "-l", action="store_true", help="Load model from file if set"
    )
    args = parser.parse_args()
    loadFromFile = args.load

    lvl_config = load_level_config()
    lvl_config = createLevel(lvl_config)

    # if lvl_config is None:
    #     print("no level config found, creating a new level")
    #     createLevel()
    #     lvl_config = load_level_config()

    # showLevel(lvl_config)

    if loadFromFile:
        best_model_file_path = find_best_agent_from_file("evo_training")
        if best_model_file_path is None:
            print(
                "No best model found in evo_training directory. Will evolve a new one."
            )
            best_agent = evolve(lvl_config)
    else:
        best_agent = evolve(lvl_config)

    allUniqueAgents = getAllDifferentAgents()
    for agentTuple in allUniqueAgents:
        score, generation, agent = agentTuple

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

        square = Rectangle(size=50, x=lvl_config["square_x"], y=lvl_config["square_y"])
        # square could be above ground level
        square.falling = CheckIfSquareIsFalling(square)
        endButton = Button(
            width=150, height=50, x=10, y=10, color=GRAY, text="End Game"
        )

        # display text about the score and generation
        font = pygame.font.Font(None, 36)
        text = font.render(f"Score: {score}, Generation: {generation}", True, WHITE)
        text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))

        done = False
        score = 0
        while not done:
            state = np.array([ball.x, ball.speed, square.x])
            action = agent.act(state)

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

            # apply gravity to the square if it is not on the ground
            square.applyGravity(gravity, GAME_HEIGHT)

            score += 1
            if done:
                break

            if (
                score > 2000
            ):  # Arbitrary large number to stop the game if it takes too long
                break

            # check if endbutton is pressed
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_x, mouse_y = event.pos
                    if (endButton.x <= mouse_x <= endButton.x + endButton.width) and (
                        endButton.y <= mouse_y <= endButton.y + endButton.height
                    ):
                        done = True
                        break

            # display the game
            # Clear screen
            screen.fill(BLACK)

            draw_button(screen, endButton)

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

            screen.blit(text, text_rect)

            # Update display
            pygame.display.flip()

            # Cap the frame rate
            pygame.time.Clock().tick(60)
