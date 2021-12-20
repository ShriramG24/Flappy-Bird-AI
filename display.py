import pygame
import random
import math
from pipes import Pipes
from sprite import Sprite

pygame.init()

x_bg = 0
pipe_x = 1000
pipes = []
pipes_hb = []
angle = math.pi / 6


class Display:
    def __init__(self, w, h, bg, sprites, bools, obstacles):
        self.width, self.height = w, h
        self.window = pygame.display.set_mode((w, h))
        self.bg = bg
        self.sprites = sprites  # sprite
        self.bools = bools  # dict of game's global variables
        self.obstacles = obstacles  # list of obstacle sprites
        self.score = 0
        self.scores = [0]

    def restart(self):
        global x_bg, pipe_x, pipes, pipes_hb
        self.bools = {key: False for (key, value) in self.bools.items()}
        self.bools["running"] = True
        self.sprites = Sprite(["Sprites/bird_down.png",
                               "Sprites/bird_straight.png",
                               "Sprites/bird_up.png"], self.width / 2 - 25, self.height / 2, 50, 35)
        x_bg = 0
        pipe_x = 600
        pipes = []
        pipes_hb = []
        self.score = 0
        self.main_loop(50)

    def event_loop(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.bools["running"] = False
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if not self.bools["started"]:
                    self.bools["started"] = True
                elif self.bools["dead"]:
                    self.restart()
                else:
                    self.sprites.jump(20, 2)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not self.bools["started"]:
                        self.bools["started"] = True
                    elif self.bools["paused"]:
                        self.bools["paused"] = False
                    elif self.bools["dead"]:
                        self.restart()
                    else:
                        self.sprites.jump(20, 2)
                if event.key == pygame.K_RETURN and self.bools["dead"]:
                    self.restart()
                if event.key == pygame.K_ESCAPE:
                    if self.bools["paused"]:
                        self.bools["paused"] = False
                    else:
                        self.bools["paused"] = True

    def main_loop(self, fps):
        clock = pygame.time.Clock()
        global x_bg
        while self.bools["running"]:
            clock.tick(fps)
            self.event_loop()

            if self.sprites.get_loc()[1] >= self.height - 35 or \
                    self.sprites.get_hb().collidelist(pipes_hb) != -1:
                self.bools["dead"] = True

            self.window.blit(self.bg, (x_bg, 0))
            self.redraw_window()
            self.window.blit(self.sprites.get_img(), self.sprites.get_loc())

            if not self.bools["dead"]:
                self.title_screen()
                self.pause()

                if self.bools["started"]:
                    self.sprites.fall(1.6)

                for pipe in pipes:
                    pipe.move_back(5)
                    if pipe.get_loc()[0] < -300:
                        pipes.pop(pipes.index(pipe))
                    if self.sprites.get_loc()[0] >= pipe.get_loc()[0] + 82 and not pipe._passed():
                        self.score += 1

            else:
                self.sprites.set_img(["Sprites/bird_dead.png"])
                self.sprites.fall(3.2)
                if self.sprites.get_loc()[1] >= self.height - 35:
                    dark_scr = pygame.Surface((self.width, self.height))
                    dark_scr.set_alpha(128)
                    dark_scr.fill((0, 0, 0))
                    self.window.blit(dark_scr, (0, 0))
                    self.game_over()

            pygame.display.update()

    def title_screen(self):
        global x_bg, angle
        title = pygame.image.load("Sprites/Title.png")
        while not self.bools["started"]:
            self.event_loop()

            angle += math.pi / 6
            self.sprites.bounce(angle, 3)
            self.window.blit(self.bg, (0, 0))
            self.window.blit(self.sprites.get_img(), self.sprites.get_loc())
            self.window.blit(title, (54, 100))
            self.display_text("Press Space to Start", (65, 400), 30)
            self.display_text(f"High Score: {max(self.scores)}", (135, 500), 30)
            pygame.display.update()

    def redraw_window(self):
        global x_bg, pipe_x, pipes, pipes_hb
        if not self.bools["dead"]:
            self.window.blit(self.bg, (self.width + x_bg, 0))
            self.window.blit(self.sprites.get_img(), self.sprites.get_loc())
            if x_bg <= -1 * self.width:
                x_bg = 0
            x_bg -= 5

        else:
            self.window.blit(self.bg, (self.width + x_bg, 0))
            self.window.blit(self.sprites.get_img(), self.sprites.get_loc())

        if self.bools["started"]:
            pipe_y = random.randint(-175, -50)
            pipe_set = Pipes(self.obstacles[0], pipe_x, pipe_y, self.obstacles[1], pipe_y + 475)
            pipes, pipes_hb = pipe_set.store_pipes(pipes, pipes_hb)
            for pipe in pipes:
                pipe.draw(self.window)
            pipe_x += 300
            if not self.bools["dead"]:
                self.display_text(f"Score: {self.score}", (300, 25), 35)

        pygame.display.update()

    def pause(self):
        while self.bools["paused"]:
            self.event_loop()
            self.display_text("Paused", (168, 270), 40)
            pygame.display.update()

    def game_over(self):
        while self.bools["dead"]:
            self.event_loop()

            self.display_text("Game Over", (110, 180), 50)
            self.display_text(f"Score: {self.score}", (150, 290), 40)
            self.scores.append(self.score)
            if self.score == 0:
                pass
            elif self.score == max(self.scores):
                self.display_text("New High Score!", (70, 350), 40)
            else:
                self.display_text(f"High Score: {max(self.scores)}", (100, 350), 40)

            pygame.display.update()

    def display_text(self, text, pos, size):
        pygame.font.init()
        font = pygame.font.Font("Sprites/EightBitDragon-anqx.ttf", size)
        font = font.render(text, True, (0, 0, 0))
        self.window.blit(font, pos)
