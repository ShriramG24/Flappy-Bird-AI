import pygame
import gym
from display import Display
from sprite import Sprite

pygame.init()
fps = 45
win_width = 500
win_height = 600

sprite = Sprite(["Sprites/bird_down.png",
                 "Sprites/bird_straight.png",
                 "Sprites/bird_up.png"], win_width / 2 - 25, win_height / 2, 50, 35)
bg = pygame.image.load("Sprites/bg.png")
obstacles = [pygame.image.load("Sprites/pipe_down.png"), pygame.image.load("Sprites/pipe_up.png")]

bools = {"jumping": False,
         "dead": False,
         "started": False,
         "paused": False,
         "running": True}

screen = Display(win_width, win_height, bg, sprite, bools, obstacles)
pygame.display.set_caption("Flappy Bird")

# Run Game
screen.main_loop(fps)

pygame.quit()
