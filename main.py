import pygame
import neat
from display import Display
from sprite import Sprite

pygame.init()
fps = 40
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

screen = Display(win_width, win_height, bg, sprite, bools, obstacles, fps)
pygame.display.set_caption("Flappy Bird")

# Run Game
#screen.main_loop()


# NEAT Algorithm
def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    survivor = pop.run(screen.fitness_eval, 50)
    print('\nBest genome:\n{!s}'.format(survivor))



# Run AI Training
config_pth = "net_config.txt"
run(config_pth)

pygame.quit()
