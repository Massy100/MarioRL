import pygame as pg
import sys
sys.path.insert(0, ".")
from mario_env import MarioEnv

env = MarioEnv(render=True)
obs, _ = env.reset()
for i in range(200):
    obs, reward, done, truncated, info = env.step(4)
    print(f"Frame {i} | X: {info['mario_x']} | action: 4 (correr+saltar)")
    for event in pg.event.get():
        if event.type == pg.QUIT:
            env.close()
            sys.exit()
    if done:
        print("Murió o ganó")
        break

env.close()