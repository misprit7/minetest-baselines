import gymnasium as gym
import numpy as np
import time

import minetest_baselines.tasks
import minetest_baselines.algos.train_muax as train_muax

env = train_muax.make_env('minetester-treechop_shaped-v0', 0, 0, False, 'test', render=True)()

env.reset()
env.step(4)
env.render()

while True:
    s = input()
    c = 4
    if s == 'exit' or s == 'q':
        break
    elif 'r' in s:
        env.reset()

    # Left/right
    if 'a' in s:
        c = 1
    elif 'd' in s:
        c = 7
    elif 'w' in s:
        c = 3
    elif 's' in s:
        c = 5

    # Jump
    if ' ' in s:
        c += 9

    # Forward
    if 'e' in s:
        c += 18

    obs, r, done, truncated, info = env.step(c)
    print('r: ', r)
    env.render()

env.close()


