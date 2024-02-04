import gymnasium as gym
from gymnasium import spaces
import numpy as np


class OneActionOneReward(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, **kwargs):
        self.observation_shape = (128, 128, 1)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape) * 255,)
        self.action_space = spaces.Discrete(1)
        self.terminated = False

    def step(self, action):
        observation = self._get_obs()
        info = self._get_info()

        reward = 1
        self.terminated = True

        return observation, reward, self.terminated, False, info

    def _get_obs(self):
        return np.ones(self.observation_shape, dtype = np.float32) * 255

    def _get_info(self):
        if self.terminated:
            return {"terminal_observation": self._get_obs}
        else:
            return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def close(self):
        return


class OneActionTwoObservationTwoReward(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, **kwargs):
        self.observation_shape = (128, 128, 1)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape) * 255,)
        self.action_space = spaces.Discrete(1)
        self.terminated = False

    def step(self, action):
        if np.allclose(self.curr_observation, np.zeros(self.observation_shape)):
            reward = -1
        else:
            reward = 1
        self.curr_observation = self._get_obs()
        self.terminated = True
        info = self._get_info()

        return self.curr_observation, reward, self.terminated, False, info

    def _get_obs(self):
        self.curr_observation = np.zeros(self.observation_shape, dtype = np.float32) if np.random.random() < 0.5 else np.ones(self.observation_shape,  dtype = np.float32) * 255
        return self.curr_observation

    def _get_info(self):
        if self.terminated:
            return {"terminal_observation": self.curr_observation}
        else:
            return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.terminated = False
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def close(self):
        return


class OneActionOneRewardTwoStep(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, **kwargs):
        self.observation_shape = (128, 128, 1)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape) * 255,)
        self.action_space = spaces.Discrete(1)
        self.terminated = False
        self.stepnum = 0

    def step(self, action):
        observation = self._get_obs()
        info = self._get_info()

        reward = 1
        self.stepnum += 1

        if self.stepnum >= 2:
            self.terminated = True
        else:
            self.terminated = False

        return observation, reward, self.terminated, False, info

    def _get_obs(self):
        return np.ones(self.observation_shape, dtype = np.float32) * 255

    def _get_info(self):
        if self.terminated:
            return {"terminal_observation": self._get_obs}
        else:
            return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()
        self.stepnum = 0
        self.terminated = False

        return observation, info

    def close(self):
        return



class TwoActionTwoReward(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, **kwargs):
        self.observation_shape = (128, 128, 1)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape) * 255,)
        self.action_space = spaces.Discrete(2)
        self.terminated = False

    def step(self, action):
        observation = self._get_obs()
        info = self._get_info()

        reward = 1 if action == 1 else -1
        self.terminated = True

        return observation, reward, self.terminated, False, info

    def _get_obs(self):
        return np.ones(self.observation_shape, dtype = np.float32) * 255

    def _get_info(self):
        if self.terminated:
            return {"terminal_observation": self._get_obs}
        else:
            return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()
        self.terminated = False

        return observation, info

    def close(self):
        return


class TwoActionTwoObservationTwoReward(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    def __init__(self, **kwargs):
        self.observation_shape = (128, 128, 1)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape) * 255)
        self.action_space = spaces.Discrete(2)
        self.terminated = False

    def step(self, action):
        if action == 1 and np.allclose(self.curr_observation, np.ones(self.observation_shape) * 255):
            reward = 1
        elif action == 0  and np.allclose(self.curr_observation, np.zeros(self.observation_shape)):
            reward = 1
        else:
            reward = -1
        self.terminated = True

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, self.terminated, False, info

    def _get_obs(self):
        self.curr_observation = np.zeros(self.observation_shape, dtype = np.float32) if np.random.random() < 0.5 else np.ones(self.observation_shape,  dtype = np.float32) * 255
        return self.curr_observation

    def _get_info(self):
        if self.terminated:
            return {"terminal_observation": self.curr_observation}
        else:
            return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()
        self.terminated = False

        return observation, info

    def close(self):
        return

class TwoActionTwoRewardHundredSteps(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, **kwargs):
        self.observation_shape = (128, 128, 1)
        self.observation_space = spaces.Box(low = np.zeros(self.observation_shape), 
                                            high = np.ones(self.observation_shape) * 255,)
        self.action_space = spaces.Discrete(2)
        self.terminated = False
        self.stepnum = 0

    def step(self, action):
        observation = self._get_obs()
        info = self._get_info()

        reward = 1 if action == 1 else -1
        self.stepnum += 1

        if self.stepnum >= 100:
            self.terminated = True

        return observation, reward, self.terminated, False, info

    def _get_obs(self):
        return np.ones(self.observation_shape, dtype = np.float32) * 255

    def _get_info(self):
        if self.terminated:
            return {"terminal_observation": self._get_obs}
        else:
            return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()
        self.terminated = False
        self.stepnum = 0

        return observation, info

    def close(self):
        return
