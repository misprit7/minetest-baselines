
import argparse
import os
import time
from distutils.util import strtobool

import jax
from jax import numpy as jnp

import muax
from muax import nn

from tensorboardX import SummaryWriter

import numpy as np
import gymnasium as gym
import minetest_baselines.tasks  # noqa

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument(
        "--track",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases",
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default="minetest-baselines",
        help="the wandb's project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="the entity (team) of wandb's project",
    )
    parser.add_argument(
        "--capture-video",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to capture videos of the agent behavior",
    )

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
    """
    Wrapper to deal with weird return type from minetest env.step
    """
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

def temperature_fn(max_training_steps, training_steps):
  r"""Determines the randomness for the action taken by the model"""
  if training_steps < 0.5 * max_training_steps:
      return 1.0
  elif training_steps < 0.75 * max_training_steps:
      return 0.5
  else:
      return 0.25

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
            env_port=5555,
            server_port=30000,
            x_display=4,
            render_mode="rgb_array",
        )
    )

    env = gym.wrappers.RecordEpisodeStatistics(env)



    # Not strictly neccessary, but it's a good canary to see if background unkilled minetest instances make this hang
    print("start")
    obs, _ = env.reset()
    print(type(obs))
    print("end")

    # Set up muax

    # Action space size
    num_actions = 36
    # Something to do with the range of possible rewards? Not 100% sure
    support_size = 10

    # Size of the internal representation
    # A bit unclear about it's exact form
    pred_channels = input_channels = 32
    # Number of channels coming from dynamics function
    output_channels = input_channels * 2
    # Support size including negative and 0
    full_support_size = int(support_size * 2 + 1)

    repr_fn = nn._init_resnet_representation_func(nn.ResNetRepresentation, input_channels)
    pred_fn = nn._init_resnet_prediction_func(nn.ResNetPrediction, num_actions, full_support_size, pred_channels)
    dy_fn = nn._init_resnet_dynamic_func(nn.ResNetDynamic, num_actions, full_support_size, output_channels)

    discount = 0.99
    tracer = muax.PNStep(10, discount, 0.5)
    buffer = muax.TrajectoryReplayBuffer(500)

    gradient_transform = muax.model.optimizer(init_value=1e-3, peak_value=2e-3, end_value=1e-3, warmup_steps=5000, transition_steps=5000)

    model = muax.MuZero(repr_fn, pred_fn, dy_fn, policy='muzero', discount=discount,
                        optimizer=gradient_transform, support_size=support_size)

    # Set up logging
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.capture_video:
        env = gym.wrappers.RecordVideo(
            env,
            f"videos/{run_name}",
            lambda x: x % 100 == 0,
        )


    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    print("Starting fit")
    # model_path = muax.fit(model, 
    #                     env=env,
    #                     test_env=env,
    #                     max_episodes=args.episodes,
    #                     max_training_steps=args.training_steps,
    #                     tracer=tracer,
    #                     buffer=buffer,
    #                     k_steps=10,
    #                     sample_per_trajectory=1,
    #                     num_trajectory=32,
    #                     buffer_warm_up=1,
    #                     # tensorboard_dir='/content/tensorboard/cartpole',
    #                     # model_save_path='/content/models/cartpole',
    #                     # save_name='cartpole_model_params',
    #                     random_seed=0,
    #                     log_all_metrics=True)

    # Custom training loop to incorporate wandb
    # Based mostly on train.pn in muax

    # Params
    # Many of these would normally be passed to muax.fit
    random_seed = 0
    max_training_steps = args.training_steps
    num_simulations = 50
    k_steps = 10
    max_episodes = args.episodes
    num_update_per_episode = 50
    num_trajectory = 32
    sample_per_trajectory = 1
    save_every_n_epochs = 1
    model_save_path = None
    save_name = None
    test_interval = 10
    num_test_episodes = 10
    test_env = env

    buffer_warm_up = 128
    # buffer_warm_up = 1


    # Setup

    if save_name is None:
        save_name = 'model_params'

    if model_save_path is None:
        timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
        model_dir = os.path.join('models', timestr) 
    else:
        model_dir = model_save_path 

    sample_input = jnp.expand_dims(env.observation_space.sample(), axis=0).astype(float)
    key = jax.random.PRNGKey(random_seed)
    key, test_key, subkey = jax.random.split(key, num=3)
    model.init(subkey, sample_input) 

    training_step = 0
    best_test_G = -float('inf')
    model_path = None


    print('buffer warm up stage...')
    while len(buffer) < buffer_warm_up:
        # print('buffer run')
        print("new buffer warmup episode")
        obs, info = env.reset()    
        tracer.reset()
        trajectory = muax.Trajectory()
        temperature = temperature_fn(max_training_steps=max_training_steps, training_steps=training_step)
        for t in range(env.spec.max_episode_steps):
            if t%30 == 0: print('buffer step')
            key, subkey = jax.random.split(key)
            a, pi, v = model.act(subkey, obs, 
                           with_pi=True, 
                           with_value=True, 
                           obs_from_batch=False,
                           num_simulations=num_simulations,
                           temperature=temperature)
            obs_next, r, done, truncated, info = env.step(a)
            #       if truncated:
            #         r = 1 / (1 - tracer.gamma)
            tracer.add(obs, a, r, done or truncated, v=v, pi=pi)
            while tracer:
                trans = tracer.pop()
                trajectory.add(trans)
            if done or truncated:
                break 
            obs = obs_next 
        trajectory.finalize()
        if len(trajectory) >= k_steps:
          buffer.add(trajectory, trajectory.batched_transitions.w.mean())


    print('start training...')
    # env = TrainMonitor(env, tensorboard_dir=os.path.join(tensorboard_dir, name), log_all_metrics=log_all_metrics)
    
    start_time = time.time()
    global_step = 0
    for ep in range(max_episodes):
        print("New episode")
        obs, info = env.reset(seed=random_seed)   
        tracer.reset() 
        trajectory = muax.Trajectory()
        temperature = temperature_fn(max_training_steps=max_training_steps, training_steps=training_step)
        for t in range(env.spec.max_episode_steps):
            if t%30 == 0: print('train step')
            key, subkey = jax.random.split(key)
            a, pi, v = model.act(subkey, obs, 
                                 with_pi=True, 
                                 with_value=True, 
                                 obs_from_batch=False,
                                 num_simulations=num_simulations,
                                 temperature=temperature)
            obs_next, r, done, truncated, info = env.step(a)
  #           if truncated:
  #             r = 1 / (1 - tracer.gamma)
            tracer.add(obs, a, r, done or truncated, v=v, pi=pi)
            while tracer:
                trans = tracer.pop()
                trajectory.add(trans)
                # env.record_metrics({'v': trans.v, 'Rn': trans.Rn})
                writer.add_scalar(
                    "v",
                    trans.v,
                    global_step
                )
                writer.add_scalar(
                    "Rn",
                    trans.Rn,
                    global_step
                )
                writer.add_scalar(
                    "SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
            if done or truncated:
                break 
            obs = obs_next 
            global_step += 1

        trajectory.finalize()
        if len(trajectory) >= k_steps:
            buffer.add(trajectory, trajectory.batched_transitions.w.mean())
  
        train_loss = 0
        for _ in range(num_update_per_episode):
            transition_batch = buffer.sample(num_trajectory=num_trajectory,
                                              sample_per_trajectory=sample_per_trajectory,
                                              k_steps=k_steps)
            loss_metric = model.update(transition_batch)
            train_loss += loss_metric['loss']
            training_step += 1

        train_loss /= num_update_per_episode
        # env.record_metrics({'loss': train_loss})
        if ep % save_every_n_epochs == 0:
            model_folder_name = f'epoch_{ep:04d}_loss_{train_loss:.8f}'
            if not os.path.exists(os.path.join(model_dir, model_folder_name)):
                os.makedirs(os.path.join(model_dir, model_folder_name))
            cur_path = os.path.join(model_dir, model_folder_name, save_name) 
            model.save(cur_path)
            if not model_path:
                model_path = cur_path
        if training_step >= max_training_steps:
            return model_path
        # env.record_metrics({'training_step': training_step})
  
        # Periodically test the model
        if ep % test_interval == 0:
            test_G = muax.test(model, test_env, test_key, num_simulations=num_simulations, num_test_episodes=num_test_episodes)
            writer.add_scalar(
                "test_G",
                test_G,
                global_step
            )
            # test_env.record_metrics({'test_G': test_G})
            # env.record_metrics({'test_G': test_G})
            if test_G >= best_test_G:
                best_test_G = test_G
                model_folder_name = f'epoch_{ep:04d}_test_G_{test_G:.8f}'
                if not os.path.exists(os.path.join(model_dir, model_folder_name)):
                  os.makedirs(os.path.join(model_dir, model_folder_name))
                model_path = os.path.join(model_dir, model_folder_name, save_name)
                model.save(model_path)


    writer.close()
    print("Finished fit")

