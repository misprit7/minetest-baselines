# adapted from cleanRL: https://github.com/vwxyzjn/cleanrl
import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Callable

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import psutil
from flax.training.train_state import TrainState
from jax_smi import initialise_tracking
from minetester.utils import start_xserver
from stable_baselines3.common.buffers import ReplayBuffer
from tensorboardX import SummaryWriter

import minetest_baselines.tasks  # noqa


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")
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
        "--save-model",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to save model into the `runs/{run_name}` folder",
    )
    parser.add_argument(
        "--upload-model",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=True,
        help="whether to upload the saved model to huggingface",
    )
    parser.add_argument(
        "--hf-entity",
        type=str,
        default="",
        help="the user or org name of the model repository from the Hugging Face Hub",
    )

    # Algorithm specific arguments
    parser.add_argument(
        "--env-id",
        type=str,
        default="minetester-treechop_shaped-v0",
        help="the id of the environment",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1000000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=5000, # usually 500000
        help="the replay memory buffer size",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="the discount factor gamma",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="the target network update rate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="the batch size of sample from the reply memory",
    )
    parser.add_argument(
    	"--exploration-noise", 
    	type=float, 
    	default=0.1,
        help="the scale of exploration noise",
    )
    parser.add_argument(
    	"--learning-starts", 
    	type=int, 
    	default=25e3,
        help="timestep to start learning"
    )
    parser.add_argument(
    	"--policy-frequency",
    	type=int, 
    	default=2,
        help="the frequency of training policy (delayed)"
    )
    parser.add_argument(
    	"--noise-clip", 
    	type=float, 
    	default=0.5,
        help="noise clip parameter of the Target Policy Smoothing Regularization"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="the number of environments to sample from",
    )
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        # TODO train agent on diverse seeds / biomes / conditions
        env = gym.make(
            env_id,
            world_seed=seed,
            start_xvfb=False,
            headless=True,
            env_port=5555 + idx,
            server_port=30000 + idx,
            x_display=4,
        )
        env = gym.wrappers.RecordEpisodeStatistics(env)
        # if capture_video:
        #     if idx == 0 or idx < 0:
        #         env = gym.wrappers.RecordVideo(
        #             env,
        #             f"videos/{run_name}",
        #             lambda x: x % 100 == 0,
        #         )
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk




def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: nn.Module,
    capture_video: bool = True,
    exploration_noise: float = 0.1,
    seed=1,
):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, -1, capture_video, run_name)],
    )
    obs, _ = envs.reset()

    Actor, QNetwork = Model
    action_scale = np.array((envs.action_space.high - envs.action_space.low) / 2.0)
    action_bias = np.array((envs.action_space.high + envs.action_space.low) / 2.0)
    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=action_scale,
        action_bias=action_bias,
    )
    qf = QNetwork()
    key = jax.random.PRNGKey(seed)
    key, actor_key, qf_key = jax.random.split(key, 3)
    actor_params = actor.init(actor_key, obs)
    print(envs.action_space.sample())
    qf_params = qf.init(qf_key, obs, envs.action_space.sample())
    # note: qf_params is not used in this script
    with open(model_path, "rb") as f:
        (actor_params, qf_params) = flax.serialization.from_bytes((actor_params, qf_params), f.read())
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        actions = actor.apply(actor_params, obs)
        actions = np.array(
            [
                (jax.device_get(actions)[0] + np.random.normal(0, action_scale * exploration_noise)[0]).clip(
                    envs.single_action_space.low, envs.single_action_space.high
                )
            ]
        )

        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, a: jnp.ndarray):
        x = jnp.flatnonzero(x)
        x = jnp.concatenate([x, jnp.flatnonzero(a)], -1)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x
        
        
class Actor(nn.Module):
    action_dim: int
    action_scale: jnp.ndarray
    action_bias: jnp.ndarray

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        x = nn.tanh(x)
        x = x * self.action_scale + self.action_bias
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def train(args=None):
    if args is None:
        args = parse_args()
    else:
        args = parse_args(args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, qf1_key = jax.random.split(key, 3)

    # env setup
    xserver = start_xserver(4)
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(args.env_id, args.seed, i, args.capture_video, run_name)
            for i in range(args.num_envs)
        ],
    )

    assert isinstance(
        envs.single_action_space,
        gym.spaces.Box,
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device="cpu",
        handle_timeout_termination=False,
    )

    obs = envs.reset()[0]
    # print()
    # print(obs)
    # print()
    # print(type(obs))
    # print()

    actor = Actor(
        action_dim=np.prod(envs.single_action_space.shape),
        action_scale=jnp.array((envs.action_space.high - envs.action_space.low) / 2.0),
        action_bias=jnp.array((envs.action_space.high + envs.action_space.low) / 2.0),
    )
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        target_params=actor.init(actor_key, obs),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    qf = QNetwork()

    qf1_state = TrainState.create(
        apply_fn=qf.apply,
        params=qf.init(qf1_key, obs, envs.action_space.sample()),
        target_params=qf.init(qf1_key, obs, envs.action_space.sample()),
        tx=optax.adam(learning_rate=args.learning_rate),
    )
    actor.apply = jax.jit(actor.apply)
    qf.apply = jax.jit(qf.apply)

    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf1_state: TrainState,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminations: np.ndarray,
    ):
        next_state_actions = (actor.apply(actor_state.target_params, next_observations)).clip(-1, 1)  # TODO: proper clip
        qf1_next_target = qf.apply(qf1_state.target_params, next_observations, next_state_actions).reshape(-1)
        next_q_value = (rewards + (1 - terminations) * args.gamma * (qf1_next_target)).reshape(-1)

        def mse_loss(params):
            qf_a_values = qf.apply(params, observations, actions).squeeze()
            return ((qf_a_values - next_q_value) ** 2).mean(), qf_a_values.mean()

        (qf1_loss_value, qf1_a_values), grads1 = jax.value_and_grad(mse_loss, has_aux=True)(qf1_state.params)
        qf1_state = qf1_state.apply_gradients(grads=grads1)

        return qf1_state, qf1_loss_value, qf1_a_values


    @jax.jit
    def update_actor(
        actor_state: TrainState,
        qf1_state: TrainState,
        observations: np.ndarray,
    ):
        def actor_loss(params):
            return -qf.apply(qf1_state.params, observations, actor.apply(params, observations)).mean()

        actor_loss_value, grads = jax.value_and_grad(actor_loss)(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        actor_state = actor_state.replace(
            target_params=optax.incremental_update(actor_state.params, actor_state.target_params, args.tau)
        )

        qf1_state = qf1_state.replace(
            target_params=optax.incremental_update(qf1_state.params, qf1_state.target_params, args.tau)
        )
        return actor_state, qf1_state, actor_loss_value


    initialise_tracking()
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    # obs = envs.reset()[0] # this line is not included in CleanRL for ddpg. Why?
    for global_step in range(args.total_timesteps):
        print(global_step)
        if global_step%500 == 0:
            print(f'SPS: {int(global_step / (time.time() - start_time))}, current: {global_step}/{args.total_timesteps}')
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions = actor.apply(actor_state.params, obs)
            actions = np.array(
                [
                    (jax.device_get(actions)[0] + np.random.normal(0, actor.action_scale * args.exploration_noise)[0]).clip(
                        envs.single_action_space.low, envs.single_action_space.high
                    )
                ]
            )


        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        # print(infos)
        # print(type(infos))
        # for info in infos:
        #     if "episode" in info.keys():
        #         print(
        #             f"global_step={global_step},"
        #             f"episodic_return={info['episode']['r']}",
        #         )
        #         writer.add_scalar(
        #             "charts/episodic_return",
        #             info["episode"]["r"],
        #             global_step,
        #         )
        #         writer.add_scalar(
        #             "charts/episodic_length",
        #             info["episode"]["l"],
        #             global_step,
        #         )
        #         writer.add_scalar("charts/epsilon", epsilon, global_step)
        #         break

        # TRY NOT TO MODIFY: save data to replay buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
	
        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            qf1_state, qf1_loss_value, qf1_a_values = update_critic(
                actor_state,
                qf1_state,
                data.observations.numpy(),
                data.actions.numpy(),
                data.next_observations.numpy(),
                data.rewards.flatten().numpy(),
                data.dones.flatten().numpy(),
            )
            if global_step % args.policy_frequency == 0:
                actor_state, qf1_state, actor_loss_value = update_actor(
                    actor_state,
                    qf1_state,
                    data.observations.numpy(),
                )

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss_value.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss_value.item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)


    # Close training envs
    envs.close()
    # kill any remaining minetest processes
    for proc in psutil.process_iter():
        if proc.name() in ["minetest"]:
            proc.kill()

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.model"
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(q_state.params))
        print(f"model saved to {model_path}")

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=(Actor, QNetwork),
            exploration_noise=args.exploration_noise,
            seed=args.seed,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from minetest_baselines.utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(
                args,
                episodic_returns,
                repo_id,
                "DDPG",
                f"runs/{run_name}",
                f"videos/{run_name}-eval",
            )

    xserver.terminate()
    writer.close()


if __name__ == "__main__":
    train()