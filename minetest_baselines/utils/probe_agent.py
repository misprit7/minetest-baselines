#!/usr/bin/env python3
import gymnasium as gym
import minetest_baselines.tasks  # noqa

render = True
max_steps = 100

env_id = "minetester-treechop_shaped-v0"
seed = 0

env = gym.make(
            env_id,
            base_seed=seed,
            headless=False,
            start_xvfb=True,
            env_port=5555,
            server_port=30000,
        )

env.reset()
done = False
step = 0

print("""
      Welcome to the Minetest Probe Agent!
      The probe agent is a debugging tool to test the reward system and the action space.
      For the default environment (specified above), the action space is as follows:

      Camera Movement:
      0  3  6
      1  4  7
      2  5  8

      Jump: +9

      Forward: +18
      """)

while True:
    try:
        action = int(input("Enter action (0-{}): ".format(env.action_space.n - 1)))
        number = int(input("Enter repetitions (>0): "))

        for i in range(number):
            _, rew, done, truncated, info = env.step(action)
            print(step, rew, done or truncated, info)
            if render:
                env.render()
            if done or truncated:
                env.reset()
            step += 1

    except ValueError:
        print("Please enter a valid integer.")
    except KeyboardInterrupt:
        break
env.close()