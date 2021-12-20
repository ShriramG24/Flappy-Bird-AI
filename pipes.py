import pygame
pygame.init()


class Pipes:
    def __init__(self, pipe_above, x, y1, pipe_below, y2):
        self.pipe_above, self.x, self.y1 = pipe_above, x, y1
        self.pipe_below, self.y2 = pipe_below, y2
        self.above_hitbox = pygame.Rect(self.x, self.y1 - 5000, 82, 5300)
        self.below_hitbox = pygame.Rect(self.x, self.y2, 82, 300)
        self.passed = False

    def _passed(self):
        if not self.passed:
            self.passed = True
            return False
        else:
            return True

    def move_back(self, steps):
        self.x -= steps
        self.above_hitbox.update(self.x, self.y1 - 5000, 82, 5300)
        self.below_hitbox.update(self.x, self.y2, 82, 300)

    def get_above_hb(self):
        return self.above_hitbox

    def get_below_hb(self):
        return self.below_hitbox

    def get_loc(self):
        return self.x, self.y1, self.y2

    def draw(self, surface):
        surface.blit(self.pipe_above, (self.x, self.y1))
        # pygame.draw.rect(surface, (255, 0, 0), self.above_hitbox, 2)
        surface.blit(self.pipe_below, (self.x, self.y2))
        # pygame.draw.rect(surface, (255, 0, 0), self.below_hitbox, 2)

    def store_pipes(self, pipes, pipes_hb):
        pipes.append(self)
        pipes_hb.append(self.get_above_hb())
        pipes_hb.append(self.get_below_hb())
        return pipes, pipes_hb



