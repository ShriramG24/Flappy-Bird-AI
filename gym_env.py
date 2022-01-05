from gym import Env, spaces
import pygame
import torch
import torchvision.transforms as T
from display import Display
from sprite import Sprite
import numpy as np
import os
import cv2

pygame.init()

FPS = 30
WIN_WIDTH = 500
WIN_HEIGHT = 600

X_BG = 0
PIPE_X = 400
PIPES = []
PIPES_HB = []

BG = pygame.image.load("Sprites/bg.png")
SPRITE = Sprite(["Sprites/bird_down.png",
                 "Sprites/bird_straight.png",
                 "Sprites/bird_up.png"], WIN_WIDTH / 2 - 25, WIN_HEIGHT / 2, 50, 35)
OBSTACLES = [pygame.image.load("Sprites/pipe_down.png"), pygame.image.load("Sprites/pipe_up.png")]
BOOLS = {"jumping": False,
         "dead": False,
         "started": False,
         "paused": False,
         "running": True}


class FlappyBirdEnv(Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(500, 600))
        self.display = Display(WIN_WIDTH, WIN_HEIGHT, BG, SPRITE, BOOLS, OBSTACLES, FPS)
        self.reward = 0
        pygame.display.set_caption("Flappy Bird")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clock = pygame.time.Clock()

    def step(self, action):
        if action == 0:
            self.reward += 1
        else:
            initial_y = self.display.sprites.get_loc()[1]
            self.display.sprites.jump(20, 2)
            while self.display.sprites.get_loc()[1] < initial_y:
                pass
            self.reward -= 1
        self.reward = self.display.calc_reward(self.reward)
        done = self.render()
        obs = self.process_image()
        if done:
            self.reset()

        return obs, torch.tensor([self.reward], device=self.device), done, {}

    def reset(self):
        self.display.reset()
        self.reward = 0
        return self.process_image()

    def process_image(self):
        pygame.image.save(self.display.window, "frame.jpg")
        frame = cv2.imread('frame.jpg')
        g_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (thresh, bnw_frame) = cv2.threshold(g_frame, 127, 255, cv2.THRESH_BINARY)
        os.remove("frame.jpg")
        bnw_frame = np.ascontiguousarray(bnw_frame, dtype=np.float32)
        bnw_frame = torch.from_numpy(bnw_frame)

        resize = T.Compose([T.ToPILImage(), T.ToTensor()])

        return resize(bnw_frame).unsqueeze(0).to(self.device)

    def render(self, mode="human", close=False):
        self.clock.tick(FPS)
        return self.display.draw_frame()



