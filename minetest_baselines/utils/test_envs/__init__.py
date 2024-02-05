import gymnasium as gym
from gymnasium.envs.registration import register
from minetest_baselines.utils.test_envs.probe_envs import OneActionOneReward
from minetest_baselines.utils.test_envs.probe_envs import OneActionOneRewardTwoStep
from minetest_baselines.utils.test_envs.probe_envs import TwoActionTwoReward
from minetest_baselines.utils.test_envs.probe_envs import TwoActionTwoObservationTwoReward
from minetest_baselines.utils.test_envs.probe_envs import TwoActionTwoRewardThirtyTwoSteps
from minetest_baselines.utils.test_envs.probe_envs import OneActionOneRewardThirtyTwoSteps
from minetest_baselines.utils.test_envs.probe_envs import TwoActionTwoObservationTwoRewardThirtyTwoSteps


## For DQN, PPO, etc.
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

## For Muax, EfficientZero, etc
register(
    id='OneActionOneRewardThirtyTwoSteps-v0',
    entry_point='minetest_baselines.utils.test_envs:OneActionOneRewardThirtyTwoSteps',
    max_episode_steps=32,
)

register(
    id='TwoActionTwoRewardThirtyTwoSteps-v0',
    entry_point='minetest_baselines.utils.test_envs:TwoActionTwoRewardThirtyTwoSteps',
    max_episode_steps=32,
)


register(
    id='TwoActionTwoObservationTwoRewardThirtyTwoSteps-v0',
    entry_point='minetest_baselines.utils.test_envs:TwoActionTwoObservationTwoRewardThirtyTwoSteps',
    max_episode_steps=32,
)
