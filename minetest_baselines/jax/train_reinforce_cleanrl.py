# adapted from cleanRL: https://github.com/vwxyzjn/cleanrl
import argparse
import gc
import os
import random
import time
from distutils.util import strtobool
from functools import partial
from typing import Callable, Sequence

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import psutil
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax_smi import initialise_tracking
from tensorboardX import SummaryWriter

import minetest_baselines.tasks  # noqa

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.3"

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
        "--torch-deterministic",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`",
    )
    parser.add_argument(
        "--cuda",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="if toggled, cuda will be enabled by default",
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
        "--video-frequency",
        type=int,
        default=100,
        help="number of episodes between video recordings",
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
        default=5000000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2.5e-4,
        help="the learning rate of the optimizer",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="the number of parallel game environments",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=128,
        help="the number of steps to run in each environment per policy rollout",
    )
    parser.add_argument(
        "--anneal-lr",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggle learning rate annealing for policy and value networks",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="the discount factor gamma",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="the lambda for the general advantage estimation",
    )
    parser.add_argument(
        "--update-epochs",
        type=int,
        default=4,
        help="the K epochs to update the policy",
    )
    parser.add_argument(
        "--norm-adv",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="Toggles advantages normalization",
    )
    parser.add_argument(
        "--clip-coef",
        type=float,
        default=0.1,
        help="the surrogate clipping coefficient",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="coefficient of the entropy",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="coefficient of the value function",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="the target KL divergence threshold",
    )
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.num_updates = args.total_timesteps // args.batch_size
    return args

def make_env(env_id, seed, idx, capture_video, video_frequency, run_name):
    def thunk():
        env = gym.make(
            env_id,
            base_seed=seed + idx,
            headless=True,
            start_xvfb=False,
            env_port=5555 + idx,
            server_port=30000 + idx,
        )
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(
                    env,
                    f"videos/{run_name}",
                    lambda x: x % video_frequency == 0,
                    name_prefix=f"env-{idx}",
                    disable_logger=True,
                )
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
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
    video_frequency: int = 2,
    seed: int = 1,
):
    envs = gym.vector.SyncVectorEnv(
        [make_env(env_id, seed, 0, capture_video, video_frequency, run_name)],
    )
    Actor = Model
    # next_obs, _ = envs.reset()
    actor = Actor(action_dim=envs.single_action_space.n)
    key = jax.random.PRNGKey(seed)
    key, actor_key = jax.random.split(key, 2)

    actor_params = actor.init(
        actor_key,
        np.array([envs.single_observation_space.sample()]),
    )

    with open(model_path, "rb") as f:
        (
            args,
            actor_params,
        ) = flax.serialization.from_bytes(
            (None, actor_params),
            f.read(),
        )

    @jax.jit
    def get_action(
        actor_params: flax.core.FrozenDict,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        logits = actor.apply(actor_params, next_obs)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/
        #     sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        return action, key

    # a simple non-vectorized version
    episodic_returns = []
    for episode in range(eval_episodes):
        episodic_return = 0
        episodic_length = 0
        next_obs, _ = envs.reset()
        terminated = np.array([False])
        truncated = np.array([False])

        while not (terminated[0] or truncated[0]):
            actions, key = get_action(
                actor_params,
                next_obs,
                key,
            )
            next_obs, reward, terminated, truncated, infos = envs.step(
                np.array(actions),
            )
            episodic_return += reward[0]
            episodic_length += 1

            if terminated[0] or truncated[0]:
                print(
                    f"eval_episode={len(episodic_returns)}, "
                    f"episodic_return={episodic_return}, "
                    f"episodic_length={episodic_length}",
                )
                episodic_returns.append(episodic_return)

    return episodic_returns

class Actor(nn.Module):
    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = x / (255.0)
        x = nn.Conv(
            32,
            kernel_size=(8, 8),
            strides=(4, 4),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            64,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding="VALID",
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(512, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            x,
        )
        x = nn.relu(x)
        x = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(x)
        return x

# @flax.struct.dataclass
# class AgentParams:
#     network_params: flax.core.FrozenDict
#     actor_params: flax.core.FrozenDict
#     critic_params: flax.core.FrozenDict

@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array

@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array

def train(args=None):
    if args is None:
        args = parse_args()
    else:
        args = parse_args(args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key = jax.random.split(key, 2)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                args.seed,
                i,
                args.capture_video,
                args.video_frequency,
                run_name,
            )
            for i in range(args.num_envs)
        ],
    )
    assert isinstance(
        envs.single_action_space,
        gym.spaces.Discrete,
    ), "only discrete action space is supported"

    def step_env_wrapped(action, step):
        (next_obs, reward, next_done, next_truncated, _) = envs.step(action)
        reward = reward.astype(jnp.float32)
        return next_obs, reward, next_done, next_truncated

    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
    )

    # def linear_schedule(count):
    #     # anneal learning rate linearly after one training iteration which contains
    #     # (args.num_minibatches * args.update_epochs) gradient updates
    #     frac = (
    #         1.0
    #         - (count // args.num_updates)
    #     )
    #     return args.learning_rate * frac

    actor = Actor(action_dim=envs.single_action_space.n)

    actor_params = actor.init(
        actor_key,
        np.array([envs.single_observation_space.sample()]),
    )

    agent_state = TrainState.create(
        apply_fn=None,
        params=actor_params,
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    actor.apply = jax.jit(actor.apply)




    @jax.jit
    def get_action(
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
        step: int,
        key: jax.random.PRNGKey,
    ):
        logits = actor.apply(agent_state.params.actor_params, next_obs)
        # sample action: Gumbel-softmax trick
        # see https://stats.stackexchange.com/questions/359442/
        #     sampling-from-a-categorical-distribution
        key, subkey = jax.random.split(key)
        u = jax.random.uniform(subkey, shape=logits.shape)
        action = jnp.argmax(logits - jnp.log(-jnp.log(u)), axis=1)
        logprob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]
        storage = storage.replace(
            obs=storage.obs.at[step].set(next_obs),
            dones=storage.dones.at[step].set(next_done),
            actions=storage.actions.at[step].set(action),
            logprobs=storage.logprobs.at[step].set(logprob),
        )
        return storage, action, key

    def rollout(
        agent_state,
        episode_stats,
        next_obs,
        next_done,
        storage,
        key,
        global_step,
    ):
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            storage, action, key = get_action(
                agent_state,
                next_obs,
                next_done,
                storage,
                step,
                key,
            )

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, next_truncated = step_env_wrapped(action, step)
            new_episode_return = episode_stats.episode_returns + reward
            new_episode_length = episode_stats.episode_lengths + 1

            episode_stats = episode_stats.replace(
                episode_returns=(
                    new_episode_return * (1 - next_done) * (1 - next_truncated)
                ).astype(jnp.float32),
                episode_lengths=(
                    new_episode_length * (1 - next_done) * (1 - next_truncated)
                ).astype(jnp.int32),
                # only update the `returned_episode_returns` if the episode is done
                returned_episode_returns=jnp.where(
                    next_done + next_truncated,
                    new_episode_return,
                    episode_stats.returned_episode_returns,
                ),
                returned_episode_lengths=jnp.where(
                    next_done + next_truncated,
                    new_episode_length,
                    episode_stats.returned_episode_lengths,
                ),
            )
            storage = storage.replace(rewards=storage.rewards.at[step].set(reward))
        return (
            agent_state,
            episode_stats,
            next_obs,
            next_done,
            storage,
            key,
            global_step,
        )

    storage = Storage(
        obs=jnp.zeros(
            (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        ),
        actions=jnp.zeros(
            (args.num_steps, args.num_envs) + envs.single_action_space.shape,
            dtype=jnp.int32,
        ),
        logprobs=jnp.zeros((args.num_steps, args.num_envs)),
        dones=jnp.zeros((args.num_steps, args.num_envs)),
        advantages=jnp.zeros((args.num_steps, args.num_envs)),
        returns=jnp.zeros((args.num_steps, args.num_envs)),
        rewards=jnp.zeros((args.num_steps, args.num_envs)),
    )

    initialise_tracking()
    
    global_step = 0
    next_obs, _ = envs.reset()
    start_time = time.time()
    next_done = jnp.zeros(args.num_envs, dtype=jnp.bool_)

    for update in range(1, args.num_updates + 1):
        update_time_start = time.time()
        (
            agent_state,
            episode_stats,
            next_obs,
            next_done,
            storage,
            key,
            global_step,
        ) = rollout(
            agent_state,
            episode_stats,
            next_obs,
            next_done,
            storage,
            key,
            global_step,
        )

