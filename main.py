import pygame
from displays import Display
from sprite import Sprite

pygame.init()

FPS = 40
WIN_WIDTH = 500
WIN_HEIGHT = 600

SPRITE = Sprite(["Sprites/bird_down.png",
                 "Sprites/bird_straight.png",
                 "Sprites/bird_up.png"], WIN_WIDTH / 2 - 25, WIN_HEIGHT / 2, 50, 35)
BG = pygame.image.load("Sprites/bg.png")
OBSTACLES = [pygame.image.load("Sprites/pipe_down.png"), pygame.image.load("Sprites/pipe_up.png")]

BOOLS = {"dead": False,
         "started": False,
         "paused": False,
         "running": True}

screen = Display(WIN_WIDTH, WIN_HEIGHT, BG, SPRITE, BOOLS, OBSTACLES, FPS)
pygame.display.set_caption("Flappy Bird")

# Run Game
screen.main_loop()

pygame.quit()
