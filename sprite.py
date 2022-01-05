import pygame
import math
pygame.init()


class Sprite(pygame.sprite.Sprite):
    def __init__(self, filenames, x, y, w, h):
        super(Sprite, self).__init__()
        self.images = [pygame.image.load(filename) for filename in filenames]
        self.curr_img = 1
        self.x, self.y, self.w, self.h = x, y, w, h
        self.del_vel = 0
        self.hitbox = pygame.Rect(self.x, self.y, self.w, self.h)
        self.score = 0

    def set_img(self, imgs):
        self.images = [pygame.image.load(img) for img in imgs]

    def get_img(self):
        self.curr_img += 0.3
        if self.curr_img >= len(self.images):
            self.curr_img = 0
        return self.images[int(self.curr_img)]

    def get_score(self, pipes):
        for pipe in pipes:
            if self.get_loc()[0] >= pipe.get_loc()[0] + 82 and not pipe.passed():
                self.score += 1
        return self.score

    def get_loc(self):
        return self.x, self.y

    def get_hb(self):
        return self.hitbox

    def jump(self, vel, grav):
        self.del_vel = 0
        if self.y >= 565:
            self.y = 565
        elif self.del_vel >= -1 * grav:
            self.del_vel -= vel
        self.y -= self.del_vel
        self.hitbox.update(self.x, self.y, self.w, self.h)

    def fall(self, grav):
        if self.del_vel < 20:
            self.del_vel += grav
        self.y += self.del_vel
        self.hitbox.update(self.x, self.y, self.w, self.h)

    def bounce(self, angle, amp):
        self.y += amp * math.sin(angle)
        pygame.time.wait(15)
        self.hitbox.update(self.x, self.y, self.w, self.h)



