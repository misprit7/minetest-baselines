
import gymnasium as gym
from gymnasium.envs.registration import register
from minetest_baselines.utils.test_envs.probe_envs import OneActionOneReward
from minetest_baselines.utils.test_envs.probe_envs import OneActionOneRewardTwoStep
from minetest_baselines.utils.test_envs.probe_envs import TwoActionTwoReward
from minetest_baselines.utils.test_envs.probe_envs import TwoActionTwoObservationTwoReward

register(
    id='TwoActionTwoReward-v0',
    entry_point='minetest_baselines.utils.test_envs:TwoActionTwoReward',
    max_episode_steps=1,
)

register(
    id='TwoActionTwoObservationTwoReward-v0',
    entry_point='minetest_baselines.utils.test_envs:TwoActionTwoObservationTwoReward',
    max_episode_steps=1,
)

register(
    id='OneActionOneReward-v0',
    entry_point='minetest_baselines.utils.test_envs:OneActionOneReward',
    max_episode_steps=1,
)

register(
    id='OneActionOneRewardTwoStep-v0',
    entry_point='minetest_baselines.utils.test_envs:OneActionOneRewardTwoStep',
    max_episode_steps=2,
)