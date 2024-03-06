
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

from minetester.utils import start_xserver

#Probe environment import
import subprocess
# Debugging imports
import minetest_baselines.utils.test_envs
import minetest_baselines.utils.logging as logger

# Note on nototation: the example muax/dqn/ppo code for muax is confusing about episode vs epoch.
# Here I refer to a complete session of a minetest game as an episode (i.e. until done=True is returned)
# An epoch is a combination of many episodes and a training update afterwards

###############################################################################
# Arguments
###############################################################################
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
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="the number of environments to sample from",
    )
    parser.add_argument(
        "--headless",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether minetest is executing on a headless machine",
    )

    # Algorithm specific
    parser.add_argument(
        "--env-id",
        type=str,
        default="minetester-treechop_shaped-v0",
        help="the id of the environment",
    )

    parser.add_argument(
        "--episodes-per-epoch",
        type=int,
        default=30,
        help="Number of environment runs in a single episode",
    )
    parser.add_argument(
        "--updates-per-epoch",
        type=int,
        default=50,
        help="Number of training updates per episode",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1000,
        help="total epochs of the experiments",
    )
    parser.add_argument(
        "--world-dir",
        type=str,
        default=None,
        help="the path to an existing world directory",
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="the path to an existing minetest.conf file",
    )

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    return args

###############################################################################
# Env Wrappers
###############################################################################
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


###############################################################################
# Utils
###############################################################################
def temperature_fn(max_epochs, training_epochs):
  r"""Determines the randomness for the action taken by the model"""
  if training_epochs < 0.5 * max_epochs:
      return 1.0
  elif training_epochs < 0.75 * max_epochs:
      return 0.5
  else:
      return 0.25

# This is similar to dqn, not entirely sure but I think this awkward helper function
# is to prevent a lambda from capturing an indexing variables whne making the 
# SyncVectorEnv
def make_env(env_id, seed, idx, capture_video, run_name, max_env_steps = 500, headless=False, world_dir = None, config_path = None):
    def thunk():
        env = gym.make(
            env_id,
            max_env_steps=max_env_steps,
            world_seed=seed,
            start_xvfb=headless, #True for remote, false for local
            headless=True,
            env_port=5555+idx,
            server_port=30000+idx,
            x_display=4,
            render_mode="rgb_array",
            world_dir=world_dir,
            config_path=config_path
        )
        env = LazyWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        if capture_video:
            env = gym.wrappers.RecordVideo(
                env,
                f"videos/{run_name}",
                step_trigger = None,
                episode_trigger=lambda x: x % 1 == 0,
            )


        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

# Test function, taken from Muax and modified for sync vector
# For simplicity also changed so that each test is run on num_envs episodes
def test(model, envs, key, num_simulations, max_env_steps, random_seed=None):
    num_envs = 1#len(envs.envs)
    total_reward = np.zeros(num_envs)

    obs, info = envs.reset(seed=random_seed)
    for t in range(max_env_steps):
        key, subkey = jax.random.split(key)
        a = model.act(subkey, obs, 
                      with_pi=False, 
                      with_value=False, 
                      obs_from_batch=True,
                      num_simulations=num_simulations,
                      temperature=0.) # Use deterministic actions during testing
        obs_next, r, done, truncated, info = envs.step(a)
        total_reward += r
        if done.any() or truncated.any():
            # TODO: If we ever get good enough to finish runs before max_env_steps we should be smarter about this
            break 
        obs = obs_next 
        
    average_test_reward = np.mean(total_reward)
    return average_test_reward


###############################################################################
# Main Training Loop
###############################################################################
def train(args=None):
    if args is None:
        args = parse_args()
    else:
        args = parse_args(args)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    subprocess.call("./worlds/CopyWorld.sh")

    # Set up minetest
    if args.headless:
        start_xserver(4)
    envs = gym.vector.AsyncVectorEnv([
        make_env(args.env_id, args.seed, i, args.capture_video, run_name, headless=args.headless, world_dir = args.world_dir + '/' + str(i), config_path = args.config_path)
        for i in range(args.num_envs)
    ])

    # Not strictly neccessary, but it's a good canary to see if background unkilled minetest instances make this hang
    print("Start envs sanity check")
    obs, _ = envs.reset()
    print("Envs reset")
    test_obs = obs[0]
    print(type(obs))
    print("envs working")


    ###########################################################################
    # Muax Set Up
    ###########################################################################

    # Action space size
    num_actions = envs.single_action_space.n
    print(f"Number of actions: {num_actions}")
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
    tracers = [muax.PNStep(10, discount, 0.5) for _ in range(args.num_envs)]
    buffer = muax.TrajectoryReplayBuffer(500)

    gradient_transform = muax.model.optimizer(init_value=1e-3, peak_value=2e-3, end_value=1e-3, warmup_steps=5000, transition_steps=5000)

    model = muax.MuZero(repr_fn, pred_fn, dy_fn, policy='muzero', discount=discount,
                        optimizer=gradient_transform, support_size=support_size)

    ###########################################################################
    # Logging Set Up
    ###########################################################################
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

    ###########################################################################
    # Training Loop Params
    ###########################################################################

    # model_path = muax.fit(model, 
    #                     env=env,
    #                     test_env=env,
    #                     max_episodes=args.max_episodes,
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
    num_simulations = 50
    k_steps = 10
    max_epochs = args.max_epochs
    num_trajectory = 32
    sample_per_trajectory = 1
    save_every_n_epochs = 50
    model_save_path = None
    save_name = None
    test_interval = 10
    num_test_episodes = 10
    test_env = envs
    # buffer_warm_up = 32
    buffer_warm_up = 1

    num_update_per_epoch = args.updates_per_epoch
    episodes_per_epoch = args.episodes_per_epoch
    max_env_steps = 500 # Steps per episode


    ###########################################################################
    # Training Setup
    ###########################################################################

    if save_name is None:
        save_name = 'model_params'

    if model_save_path is None:
        timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
        model_dir = os.path.join('models', timestr) 
    else:
        model_dir = model_save_path 

    sample_input = jnp.expand_dims(envs.single_observation_space.sample(), axis=0).astype(float)
    key = jax.random.PRNGKey(random_seed)
    key, test_key, subkey = jax.random.split(key, num=3)
    model.init(subkey, sample_input) 

    training_step = 0
    best_test_G = -float('inf')
    model_path = None


    print('buffer warm up stage...')
    while len(buffer) < buffer_warm_up:
        print(f"New buffer warmup episode: {len(buffer)}")
        obs, _ = envs.reset()    
        for tracer in tracers: tracer.reset()
        trajectories = [muax.Trajectory() for _ in range(args.num_envs)]
        temperature = temperature_fn(max_epochs=max_epochs, training_epochs=0)
        
        episodes_finished = 0
        for t in range(max_env_steps):
            # if t%30 == 0: print('buffer step')
            key, subkey = jax.random.split(key)
            a, pi, v = model.act(subkey, obs, 
                           with_pi=True, 
                           with_value=True, 
                           obs_from_batch=True,
                           num_simulations=num_simulations,
                           temperature=temperature)
            obs_next, r, done, truncated, info = envs.step(a)
            print(f"Value: {v}, reward: {r}")
            #       if truncated:
            #         r = 1 / (1 - tracer.gamma)
            for i, (tracer, trajectory) in enumerate(zip(tracers, trajectories)):
                tracer.add(obs[i], a[i], r[i], done[i] or truncated[i], v=v[i], pi=pi[i])
                while tracer:
                    trans = tracer.pop()
                    trajectory.add(trans)
                if done[i] or truncated[i]:
                    # Note: sync vector is automatically reset, so no need to do it manually
                    if len(trajectory) >= k_steps:
                        trajectory.finalize()
                        buffer.add(trajectory, trajectory.batched_transitions.w.mean())
                    trajectories[i] = muax.Trajectory()
                    episodes_finished += 1
                    print('finished early:', t)

            if episodes_finished >= episodes_per_epoch:
                break

            obs = obs_next 
        for trajectory in trajectories:
            print(len(trajectory))
            if len(trajectory) >= k_steps:
                trajectory.finalize()
                buffer.add(trajectory, trajectory.batched_transitions.w.mean())
            episodes_finished += 1


    ###########################################################################
    # Main Training Loop
    ###########################################################################
    print('Start training...')
    
    start_time = time.time()
    global_step = 0

    _, old_test_policy, _ = model.act(subkey, test_obs, 
                            with_pi=True, 
                            with_value=True, 
                            obs_from_batch=False,
                            num_simulations=num_simulations,
                            temperature=temperature,
                            max_depth = None)
    
    for ep in range(max_epochs):
        print(f"New epoch: {ep}")
        obs, info = envs.reset(seed=random_seed)   
        for tracer in tracers: tracer.reset()
        trajectories = muax.Trajectory()
        temperature = temperature_fn(max_epochs=max_epochs, training_epochs=ep)

        # Logging metrics
        total_r = 0
        local_step = 0
        action_log = []
        action_count = np.zeros(num_actions)


        #######################################################################
        # Run Envs
        #######################################################################
        episodes_finished = 0
        t_stepping_start = time.time()
        for t in range(max_env_steps):
            # if t%30 == 0: print('buffer step')
            key, subkey = jax.random.split(key)
            a, pi, v = model.act(subkey, obs, 
                           with_pi=True, 
                           with_value=True, 
                           obs_from_batch=True,
                           num_simulations=num_simulations,
                           temperature=temperature)

            obs_next, r, done, truncated, info = envs.step(a)
            global_step += args.num_envs
            local_step += args.num_envs

            # Update logging metrics
            for i in range(len(obs_next)):
                total_r += r[i]
                action_count[a[i]] += 1
                action_log += [a[i]]
                if done[i] or truncated[i]:
                    print(f"number of steps in episode {t}")
                    writer.add_scalar("number of steps in episode", t, global_step)
  #           if truncated:
  #             r = 1 / (1 - tracer.gamma)
            for i, (tracer, trajectory) in enumerate(zip(tracers, trajectories)):
                tracer.add(obs[i], a[i], r[i], done[i] or truncated[i], v=v[i], pi=pi[i])

                writer.add_scalar(
                    "relative entropy",
                    logger.relative_entropy(pi[0]),
                    global_step
                )
                while tracer:
                    trans = tracer.pop()
                    trajectory.add(trans)
                    # env.record_metrics({'v': trans.v, 'Rn': trans.Rn})
                    # writer.add_scalar(
                    #     "v",
                    #     trans.v,
                    #     global_step
                    # )
                    # writer.add_scalar(
                    #     "Rn",
                    #     trans.Rn,
                    #     global_step
                    # )
                if done[i] or truncated[i]:
                    # Note: sync vector is automatically reset, so no need to do it manually
                    if len(trajectory) >= k_steps:
                        trajectory.finalize()
                        buffer.add(trajectory, trajectory.batched_transitions.w.mean())
                    episodes_finished += 1

            if episodes_finished >= episodes_per_epoch:
                break;
            obs = obs_next 

        for trajectory in trajectories:
            if len(trajectory) >= k_steps:
                trajectory.finalize()
                buffer.add(trajectory, trajectory.batched_transitions.w.mean())
            episodes_finished += 1

        #######################################################################
        # Training
        #######################################################################
        print(f"Time stepping: {time.time() - t_stepping_start}")
        print("Action counts: ", action_count)

        writer.add_scalar("mean_r", total_r / local_step, global_step);
        writer.add_scalar("episode", ep, global_step);
        writer.add_scalar("temperature", temperature, global_step);
        writer.add_scalar("SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_histogram("actions", np.array(action_log), global_step)
        writer.add_scalar("number of episodes", local_step, global_step)

        percent_forward, percent_jump, percent_look = logger.action_types(action_log)
        writer.add_scalar("percent time moving forward", percent_forward, global_step)
        writer.add_scalar("percent time jumping", percent_jump, global_step)
        writer.add_scalars("percent time looking looking", percent_look, global_step)


        if args.track:
            wandb.log({"total episode reward": total_r})
        train_loss = 0
        t_training_start = time.time()
        for _ in range(num_update_per_epoch):
            transition_batch = buffer.sample(num_trajectory=num_trajectory,
                                              sample_per_trajectory=sample_per_trajectory,
                                              k_steps=k_steps)
            loss_metric = model.update(transition_batch)
            train_loss += loss_metric['loss']
        print(f"Time training: {time.time() - t_training_start}")

        print("model updated")

        _, new_test_policy, _ = model.act(subkey, test_obs, 
                                    with_pi=True, 
                                    with_value=True, 
                                    obs_from_batch=False,
                                    num_simulations=num_simulations,
                                    temperature=temperature,
                                    max_depth = None)
        
        writer.add_scalar("KL divergence", logger.kl_divergence(old_test_policy[0], new_test_policy[0]), global_step)

        print("1")
        old_test_policy = new_test_policy
        print("2")

        train_loss /= num_update_per_epoch
        writer.add_scalar("train_loss", train_loss, training_step)
        print("3")

        #######################################################################
        # Model Saving
        #######################################################################
        if ep % save_every_n_epochs == 0 and ep > 0:
            model_folder_name = f'epoch_{ep:04d}_loss_{train_loss:.8f}'
            if not os.path.exists(os.path.join(model_dir, model_folder_name)):
                os.makedirs(os.path.join(model_dir, model_folder_name))
            cur_path = os.path.join(model_dir, model_folder_name, save_name) 
            model.save(cur_path)
            if not model_path:
                model_path = cur_path
  
        #######################################################################
        # Evaluation
        #######################################################################
        t_testing_start = time.time()
        if ep % test_interval == 0:
            test_G = test(model, test_env, test_key, num_simulations=num_simulations, max_env_steps=max_env_steps)
            writer.add_scalar("test_G", test_G, global_step)
            print(f"TEST RESULT {test_G}")
            if test_G >= best_test_G:
                best_test_G = test_G
                model_folder_name = f'epoch_{ep:04d}_test_G_{test_G:.8f}'
                if not os.path.exists(os.path.join(model_dir, model_folder_name)):
                  os.makedirs(os.path.join(model_dir, model_folder_name))
                model_path = os.path.join(model_dir, model_folder_name, save_name)
                model.save(model_path)
        print(f"Time testing: {time.time() - t_testing_start}")

    # Cleans the worlds after done running each one
    # In DQN This went after envs.close(), but I think this just needs to be done after training sometime. 
    subprocess.call("./worlds/CleanWorlds.sh")


    print(model_path)
    writer.close()
    print("Finished fit")

if __name__ == "__main__":
    train()
