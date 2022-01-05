import pygame
import random
import math
import neat
from pipes import Pipes
from sprite import Sprite

pygame.init()

# Global Variables
x_bg = 0
pipe_x = 600
pipes = []
pipes_hb = []
score = 0
scores = [0]
angle = math.pi / 6

# Fitness Function Variables
birds, ges, nets = [], [], []
gen = 1
config_path = "net_config.txt"


class Display:
    def __init__(self, w, h, bg, sprites, bools, obstacles, fps):
        self.width, self.height = w, h
        self.window = pygame.display.set_mode((w, h))
        self.bg = bg
        self.sprites = sprites
        self.bools = bools
        self.obstacles = obstacles
        self.clock = pygame.time.Clock()
        self.fps = fps

    def reset(self):
        global x_bg, pipe_x, pipes, pipes_hb, score
        x_bg = 0
        pipe_x = 600
        pipes = []
        pipes_hb = []
        score = 0
        self.bools = {key: False for (key, value) in self.bools.items()}
        self.bools["running"] = True
        self.sprites = Sprite(["Sprites/bird_down.png",
                               "Sprites/bird_straight.png",
                               "Sprites/bird_up.png"], self.width / 2 - 25, self.height / 2, 50, 35)

    def event_loop(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.bools["running"] = False
                pygame.quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if not self.bools["started"]:
                    self.bools["started"] = True
                elif self.bools["dead"]:
                    self.reset()
                    self.main_loop()
                else:
                    self.sprites.jump(20, 2)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not self.bools["started"]:
                        self.bools["started"] = True
                    elif self.bools["paused"]:
                        self.bools["paused"] = False
                    elif self.bools["dead"]:
                        self.reset()
                        self.main_loop()
                    else:
                        self.sprites.jump(20, 2)
                if event.key == pygame.K_RETURN:
                    if self.bools["dead"]:
                        self.reset()
                        self.main_loop()
                    elif not self.bools["started"]:
                        screen = TrainDisplay(self.width, self.height, self.bg, self.sprites,
                                              self.bools, self.obstacles, self.fps)
                        screen.run(config_path)
                if event.key == pygame.K_ESCAPE:
                    if self.bools["paused"]:
                        self.bools["paused"] = False
                    else:
                        self.bools["paused"] = True

    def main_loop(self):
        global x_bg, score
        while self.bools["running"]:
            self.clock.tick(self.fps)
            self.event_loop()

            if self.sprites.get_loc()[1] >= self.height - 35 or self.sprites.get_loc()[1] <= 0 or \
               self.sprites.get_hb().collidelist([pipe_hb[0] for pipe_hb in pipes_hb]) != -1 or \
               self.sprites.get_hb().collidelist([pipe_hb[1] for pipe_hb in pipes_hb]) != -1:
                self.bools["dead"] = True

            self.window.blit(self.bg, (x_bg, 0))
            self.redraw_window()
            self.window.blit(self.sprites.get_img(), self.sprites.get_loc())

            if not self.bools["dead"]:
                self.title_screen()
                self.pause()

                if self.bools["started"]:
                    self.sprites.fall(2)

                for pipe in pipes:
                    pipe.move_back(5)
                    if pipe.get_loc()[0] < -300:
                        pipes.pop(pipes.index(pipe))
                    if self.sprites.get_loc()[0] >= pipe.get_loc()[0] + 82 and not pipe.passed():
                        score += 1

            else:
                self.sprites.set_img(["Sprites/bird_dead.png"])
                self.sprites.fall(4)
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
            self.display_text("Space -> Start", (120, 390), 30)
            self.display_text("Enter -> Train AI", (108, 455), 30)
            self.display_text(f"High Score: {max(scores)}", (140, 520), 30)
            pygame.display.update()

    def redraw_window(self):
        global x_bg, pipe_x, pipes, pipes_hb, score
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
            pipe_y = random.randint(-150, -30)
            pipe_set = Pipes(self.obstacles[0], pipe_x, pipe_y, self.obstacles[1], pipe_y + 450)
            pipes, pipes_hb = pipe_set.store_pipes(pipes, pipes_hb)
            for pipe in pipes:
                pipe.draw(self.window)
            pipe_x += 300
            if not self.bools["dead"]:
                self.display_text(f"Score: {score}", (25, 25), 35)

        pygame.display.update()

    def pause(self):
        while self.bools["paused"]:
            self.event_loop()
            self.display_text("Paused", (168, 270), 40)
            pygame.display.update()

    def game_over(self):
        global score, scores
        while self.bools["dead"]:
            self.event_loop()

            self.display_text("Game Over", (110, 180), 50)
            self.display_text(f"Score: {score}", (150, 290), 40)
            scores.append(score)
            if score == 0:
                pass
            elif score == max(scores):
                self.display_text("New High Score!", (70, 350), 40)
            else:
                self.display_text(f"High Score: {max(scores)}", (100, 350), 40)

            pygame.display.update()

    def display_text(self, text, pos, size):
        pygame.font.init()
        font = pygame.font.Font("Sprites/EightBitDragon-anqx.ttf", size)
        font = font.render(text, True, (0, 0, 0))
        self.window.blit(font, pos)


class TrainDisplay(Display):
    def __init__(self, w, h, bg, sprites, bools, obstacles, fps):
        super().__init__(w, h, bg, sprites, bools, obstacles, fps)

    def pause(self):
        while self.bools["paused"]:
            self.event_loop()
            self.display_text("Paused", (168, 270), 40)
            pygame.display.update()

    def redraw_window(self):
        global x_bg, pipe_x, pipes, pipes_hb, birds, score, gen
        if not self.bools["dead"]:
            self.window.blit(self.bg, (self.width + x_bg, 0))
            for bird in birds:
                self.window.blit(bird.get_img(), bird.get_loc())
            if x_bg <= -1 * self.width:
                x_bg = 0
            x_bg -= 5

        else:
            self.window.blit(self.bg, (self.width + x_bg, 0))
            self.window.blit(self.sprites.get_img(), self.sprites.get_loc())

        if self.bools["started"]:
            pipe_y = random.randint(-150, -30)
            pipe_set = Pipes(self.obstacles[0], pipe_x, pipe_y, self.obstacles[1], pipe_y + 450)
            pipes, pipes_hb = pipe_set.store_pipes(pipes, pipes_hb)
            for pipe in pipes:
                pipe.draw(self.window)
            pipe_x += 300
            if not self.bools["dead"]:
                self.display_text(f"Score: {score}", (25, 25), 35)
                self.display_text(f"Gen:  {gen}", (335, 25), 35)

        pygame.display.update()

    def fitness_eval(self, genomes, config_pth):
        global birds, ges, nets, pipes, pipes_hb, x_bg, score, gen
        self.bools["started"] = True

        for genome_id, genome in genomes:
            birds.append(Sprite(["Sprites/bird_down.png",
                                 "Sprites/bird_straight.png",
                                 "Sprites/bird_up.png"], self.width / 2 - 25, self.height / 2, 50, 35))
            ges.append(genome)
            net = neat.nn.FeedForwardNetwork.create(genome, config_pth)
            nets.append(net)
            genome.fitness = 0

        while self.bools["started"]:
            self.clock.tick(self.fps)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

            self.window.blit(self.bg, (x_bg, 0))
            self.redraw_window()

            for pipe in pipes:
                pipe.move_back(5)
                if pipe.get_loc()[0] < -300:
                    pipes.pop(pipes.index(pipe))
                if self.sprites.get_loc()[0] >= pipe.get_loc()[0] + 82 and not pipe.passed():
                    score += 1

            for i, bird in enumerate(birds):
                self.window.blit(bird.get_img(), bird.get_loc())
                output = nets[i].activate((bird.get_hb().y, self.get_dist(bird.get_hb(), pipes_hb, True),
                                           self.get_dist(bird.get_hb(), pipes_hb, False)))
                if output[0] > 0.5 and bird.del_vel > 0:
                    bird.jump(20, 2)
                bird.fall(2)
                ges[i].fitness += bird.get_loc()[0] / 1000 + bird.get_score(pipes)
                if bird.get_loc()[1] >= self.height - 35 or bird.get_loc()[1] <= 0 or \
                        bird.get_hb().collidelist([pipe_hb[0] for pipe_hb in pipes_hb]) != -1 or \
                        bird.get_hb().collidelist([pipe_hb[1] for pipe_hb in pipes_hb]) != -1:
                    ges[i].fitness -= 1
                    [birds, ges, nets] = self.remove([birds, ges, nets], i)

            if len(birds) == 0:
                break

            pygame.display.update()

        gen += 1
        self.reset()

    def get_dist(self, sprite_hb, pipe_hbs, above):
        curr_pipe = 0
        while curr_pipe < len(pipe_hbs) and self.extract(pipe_hbs, curr_pipe, 0).x + 82 < sprite_hb.x:
            curr_pipe += 1
        x_dist = self.extract(pipe_hbs, curr_pipe, 0).x + 82 - sprite_hb.x
        if above:
            y_dist = self.extract(pipe_hbs, curr_pipe, 0).y + self.extract(pipe_hbs, curr_pipe, 1).h - sprite_hb.y
        else:
            y_dist = self.extract(pipe_hbs, curr_pipe, 1).y - sprite_hb.y + sprite_hb.h / 2
        return math.sqrt(x_dist ** 2 + y_dist ** 2)

    @staticmethod
    def remove(lists, index):
        for each_list in lists:
            each_list.pop(index)
        return lists

    @staticmethod
    def extract(_list, index, sub_index):
        sublist = _list[index]
        return sublist[sub_index]

    # NEAT Algorithm - Run Fitness Function
    def run(self, config_pth):
        config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                    neat.DefaultSpeciesSet, neat.DefaultStagnation, config_pth)

        pop = neat.Population(config)
        pop.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        pop.add_reporter(stats)
        survivor = pop.run(self.fitness_eval, 50)
        print('\nBest genome:\n{!s}'.format(survivor))
