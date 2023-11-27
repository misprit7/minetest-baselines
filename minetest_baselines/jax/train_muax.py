import argparse

import jax

import muax
from muax import nn

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

def train(args=None):
    if args is None:
        args = parse_args()
    else:
        args = parse_args(args)

    # Set up minetest
    env = gym.make(
        args.env_id,
        world_seed=args.seed,
        start_xvfb=False,
        headless=True,
        env_port=5555,
        server_port=30000,
        x_display=4,
    )

    # Set up muax
    support_size = 10 
    embedding_size = 8
    print(env.action_space.shape)
    print(env.action_space)
    num_actions = 36
    full_support_size = int(support_size * 2 + 1)

    repr_fn = nn._init_representation_func(nn.Representation, embedding_size)
    pred_fn = nn._init_prediction_func(nn.Prediction, num_actions, full_support_size)
    dy_fn = nn._init_dynamic_func(nn.Dynamic, embedding_size, num_actions, full_support_size)

    discount = 0.99
    tracer = muax.PNStep(10, discount, 0.5)
    buffer = muax.TrajectoryReplayBuffer(500)

    gradient_transform = muax.model.optimizer(init_value=0.02, peak_value=0.02, end_value=0.002, warmup_steps=5000, transition_steps=5000)

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
                        # tensorboard_dir='/content/tensorboard/cartpole',
                        # model_save_path='/content/models/cartpole',
                        # save_name='cartpole_model_params',
                        random_seed=0,
                        log_all_metrics=True)
    print("Finished fit")

