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


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Move the Ball Game")
        self.font = pygame.font.Font(None, 36)
        self.score_font = pygame.font.Font(None, 48)
        self.clock = pygame.time.Clock()
        self.gravity = 1
        self.reset_game()

    def reset_game(self):
        self.start_moving = False
        self.is_jumping = False
        self.game_over = False
        self.score = 0
        self.start_time = None
        self.dragging = False

        self.ball = Ball(
            radius=20,
            speed=5,
            gravity=1,
            jump_strength=-15,
            start_x=20,
            start_y=GAME_HEIGHT - 20,
            ground_y=GAME_HEIGHT,
        )
        self.square = Rectangle(
            size=50, x=100, y=(GAME_HEIGHT + (FIELD_HEIGHT - 50) // 2)
        )
        self.start_button = Button(
            width=150,
            height=50,
            x=WIDTH - 150 - 10,
            y=10,
            color=BLUE,
            text="Start Game",
        )
        self.final_button = Button(
            width=200,
            height=60,
            x=(WIDTH - 200) // 2,
            y=(HEIGHT - 60) // 2 + 50,
            color=RED,
            text="Restart Game",
        )

    def display_score(self):
        score_surface = self.score_font.render(f"Score: {self.score}", True, WHITE)
        score_rect = score_surface.get_rect(center=(WIDTH // 2, 30))
        self.screen.blit(score_surface, score_rect)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_button_down(event)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self.dragging = False
                elif (
                    event.type == pygame.MOUSEMOTION
                    and self.dragging
                    and not self.start_moving
                ):
                    mouse_x, mouse_y = event.pos
                    self.square.x = mouse_x - self.offset_x
                    self.square.y = mouse_y - self.offset_y
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    if not self.ball.is_jumping:
                        self.ball.velocity_y = self.ball.jump_strength
                        self.ball.is_jumping = True

            # handle ball and rectangle movement
            if self.start_moving:
                self.ball.updatePosition()
                if (
                    self.ball.x + self.ball.radius >= WIDTH
                    or self.ball.x - self.ball.radius <= 0
                ):
                    self.ball.changeDirection()
                if self.game_over == False:
                    elapsed_time = time.time() - self.start_time
                    self.score = int(elapsed_time)
                self.ball.checkCollisionWithGround()

            self.square.applyGravity(self.gravity, GAME_HEIGHT)
            if check_collision(self.ball, self.square):
                self.game_over = True

            self.screen.fill(BLACK)
            pygame.draw.rect(self.screen, GREEN, (0, GAME_HEIGHT, WIDTH, FIELD_HEIGHT))

            if self.game_over:
                self.display_game_over_screen()
            else:
                draw_button(self.screen, self.start_button, self.font, pygame)
                pygame.draw.circle(
                    self.screen, WHITE, (self.ball.x, self.ball.y), self.ball.radius
                )
                pygame.draw.rect(
                    self.screen,
                    RED,
                    (self.square.x, self.square.y, self.square.size, self.square.size),
                )
                self.display_score()

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()
        sys.exit()

    def handle_mouse_button_down(self, event):
        mouse_x, mouse_y = event.pos
        if self.game_over:
            if (
                self.final_button.x
                <= mouse_x
                <= self.final_button.x + self.final_button.width
            ) and (
                self.final_button.y
                <= mouse_y
                <= self.final_button.y + self.final_button.height
            ):
                self.reset_game()
        else:
            if not self.start_moving:
                if (
                    self.square.x <= mouse_x <= self.square.x + self.square.size
                    and self.square.y <= mouse_y <= self.square.y + self.square.size
                ):
                    self.dragging = True
                    self.offset_x = mouse_x - self.square.x
                    self.offset_y = mouse_y - self.square.y
            if (
                self.start_button.x
                <= mouse_x
                <= self.start_button.x + self.start_button.width
            ) and (
                self.start_button.y
                <= mouse_y
                <= self.start_button.y + self.start_button.height
            ):
                print("start button clicked")
                self.start_moving = True
                if self.square.y < GAME_HEIGHT:
                    self.square.falling = True
                self.start_time = time.time()

    def display_game_over_screen(self):
        game_over_text = self.score_font.render("Game Over!", True, WHITE)
        game_over_rect = game_over_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
        self.screen.blit(game_over_text, game_over_rect)

        final_score_text = self.score_font.render(
            f"Final Score: {self.score}", True, WHITE
        )
        final_score_rect = final_score_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        self.screen.blit(final_score_text, final_score_rect)

        draw_button(self.screen, self.final_button, self.font, pygame)


if __name__ == "__main__":
    game = Game()
    game.run()
