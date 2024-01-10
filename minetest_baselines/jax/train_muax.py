import argparse

import jax

import muax
from muax import nn

import numpy as np
import gymnasium as gym
import minetest_baselines.tasks  # noqa

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")

    # Algorithm specific
    parser.add_argument(
        "--env-id",
        type=str,
        default="minetester-treechop_shaped-v0",
        help="the id of the environment",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--training-steps",
        type=int,
        default=10000,
        help="total timesteps of the experiments",
    )

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    return args

class LazyWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        # observation = np.array(observation)
        observation = observation.__array__()
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        # observation = np.array(observation)
        observation = observation.__array__()
        return observation, info

def train(args=None):
    if args is None:
        args = parse_args()
    else:
        args = parse_args(args)

    # Set up minetest
    env = LazyWrapper(
        gym.make(
            args.env_id,
            world_seed=args.seed,
            start_xvfb=False,
            headless=True,
            # env_port=5555,
            # server_port=30000,
            # x_display=4,
        )
    )

    print("start")
    obs, _ = env.reset()
    print(type(obs))
    print("end")

    # Set up muax
    # support_size = 10 
    # embedding_size = 8
    # num_actions = 36
    # full_support_size = int(support_size * 2 + 1)

    # # repr_fn = nn._init_representation_func(nn.Representation, embedding_size)
    # repr_fn = nn._init_representation_func(nn.ResNetRepresentation, embedding_size)
    # pred_fn = nn._init_prediction_func(nn.ResNetPrediction, num_actions, full_support_size)
    # dy_fn = nn._init_dynamic_func(nn.ResNetDynamic, embedding_size, num_actions, full_support_size)

    num_actions = 36
    support_size = 10
    pred_channels = input_channels = 32
    output_channels = input_channels * 2
    full_support_size = int(support_size * 2 + 1)
    repr_fn = nn._init_resnet_representation_func(nn.ResNetRepresentation, input_channels)
    pred_fn = nn._init_resnet_prediction_func(nn.ResNetPrediction, num_actions, full_support_size, pred_channels)
    dy_fn = nn._init_resnet_dynamic_func(nn.ResNetDynamic, num_actions, full_support_size, output_channels)

    discount = 0.99
    tracer = muax.PNStep(10, discount, 0.5)
    buffer = muax.TrajectoryReplayBuffer(5)

    gradient_transform = muax.model.optimizer(init_value=1e-3, peak_value=2e-3, end_value=1e-3, warmup_steps=5000, transition_steps=5000)

    model = muax.MuZero(repr_fn, pred_fn, dy_fn, policy='muzero', discount=discount,
                        optimizer=gradient_transform, support_size=support_size)

    print("Starting fit")
    model_path = muax.fit(model, 
                        env=env,
                        test_env=env,
                        max_episodes=args.episodes,
                        max_training_steps=args.training_steps,
                        tracer=tracer,
                        buffer=buffer,
                        k_steps=10,
                        sample_per_trajectory=1,
                        num_trajectory=32,
                        buffer_warm_up=1,
                        # tensorboard_dir='/content/tensorboard/cartpole',
                        # model_save_path='/content/models/cartpole',
                        # save_name='cartpole_model_params',
                        random_seed=0,
                        log_all_metrics=True)
    print("Finished fit")

