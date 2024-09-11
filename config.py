# Screen dimensions
GAME_HEIGHT = 600  # Height of the game area
FIELD_HEIGHT = 200  # Height of the empty field below the game area
WIDTH, HEIGHT = (
    800,
    GAME_HEIGHT + FIELD_HEIGHT,
)  # Total screen height including the field

# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)  # Color for the empty field
RED = (255, 0, 0)  # Color for the draggable square

ACTION_NOT_JUMP = 0
ACTION_JUMP = 1


class Ball:
    def __init__(
        self, radius, speed, gravity, jump_strength, start_x, start_y, ground_y
    ):
        self.radius = radius
        self.speed = speed
        self.gravity = gravity
        self.jump_strength = jump_strength
        self.x = start_x
        self.y = start_y
        self.velocity_y = 0
        self.ground_y = ground_y
        self.is_jumping = False

    def reset(self):
        self.x = 20
        self.y = self.ground_y - 20
        self.velocity_y = 0
        self.is_jumping = False

    def start(self, speed):
        self.speed = speed

    def updatePosition(self):
        self.x += self.speed
        # Apply gravity if ball is jumping
        if self.is_jumping:
            self.velocity_y += self.gravity
            self.y += self.velocity_y

    def checkCollisionWithGround(self):
        if self.y >= self.ground_y - self.radius:
            self.y = self.ground_y - self.radius
            self.velocity_y = 0
            self.is_jumping = False

    def jump(self, action):
        if action == 1 and not self.is_jumping:
            self.velocity_y = self.jump_strength
            self.is_jumping = True

    def changeDirection(self):
        self.speed = -self.speed


class Rectangle:
    def __init__(self, size, x, y):
        self.size = size
        self.x = x
        self.y = y
        self.falling = False
        self.velocity_y = 0

    def applyGravity(self, gravity, ground_y):
        if self.falling:
            self.velocity_y += gravity
            self.y += self.velocity_y

            if self.y + self.size >= ground_y:
                self.y = ground_y - self.size
                self.velocity_y = 0
                self.falling = False


class Button:
    def __init__(self, width, height, x, y, color, text):
        self.width = width
        self.height = height
        self.x = x
        self.y = y
        self.color = color
        self.text = text


# Helper function to check collision between the ball and the obstacle
def check_collision(ball: Ball, square: Rectangle):
    # Calculate the distance between the ball and the square
    ball_center_x = ball.x
    ball_center_y = ball.y
    square_center_x = square.x + square.size / 2
    square_center_y = square.y + square.size / 2

    distance_x = abs(ball_center_x - square_center_x)
    distance_y = abs(ball_center_y - square_center_y)

    # Check for collision
    if distance_x < (square.size / 2 + ball.radius) and distance_y < (
        square.size / 2 + ball.radius
    ):
        return True
    return False

def draw_button(screen, btn, font, game):  # x, y, width, height, color, text):
    game.draw.rect(screen, btn.color, (btn.x, btn.y, btn.width, btn.height))
    text_surface = font.render(btn.text, True, WHITE)
    text_rect = text_surface.get_rect(
        center=(btn.x + btn.width // 2, btn.y + btn.height // 2)
    )
    screen.blit(text_surface, text_rect)
