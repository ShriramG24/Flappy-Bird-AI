from gym import Env, spaces
import pygame
from display import Display
from sprite import Sprite
import os
import cv2

pygame.init()

FPS = 30
WIN_WIDTH = 500
WIN_HEIGHT = 600

X_BG = 0
PIPE_X = 1000
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
        self.display = Display(WIN_WIDTH, WIN_HEIGHT, BG, SPRITE, BOOLS, OBSTACLES)
        self.reward = 0
        pygame.display.set_caption("Flappy Bird")

    def step(self, action):
        if action == 0:
            self.reward += 2
        else:
            self.display.sprites.jump(20, 2)
            self.reward -= 1
        done = self.render()
        obs = self.process_obs()
        if done:
            self.reward -= 20
            self.reset()

        return obs, self.reward, done, {}

    def reset(self):
        self.display.reset()
        return self.process_obs()

    def process_obs(self):
        pygame.image.save(self.display.window, "frame.jpg")
        frame = cv2.imread('frame.jpg')
        g_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        (thresh, bnw_frame) = cv2.threshold(g_frame, 127, 255, cv2.THRESH_BINARY)
        os.remove("frame.jpg")
        return bnw_frame

    def render(self, mode="human", close=False):
        return self.display.draw_frame(FPS)


env = FlappyBirdEnv()
for episode in range(10):
    observation = env.reset()
    for t in range(100):
        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps.".format(t + 1))
            break
env.close()


