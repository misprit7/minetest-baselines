# adapted from cleanRL: https://github.com/vwxyzjn/cleanrl
import argparse
import os
import random
import time
from distutils.util import strtobool
from functools import partial
from typing import Callable, Sequence

import flax
import flax.linen as nn
import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import psutil
from flax import optim
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax_smi import initialise_tracking
from minetester.utils import start_xserver
from tensorboardX import SummaryWriter

import minetest_baselines.tasks  # noqa

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"
# see https://github.com/google/jax/discussions/6332#discussioncomment-1279991


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
        "--num-minibatches",
        type=int,
        default=4,
        help="the number of mini-batches",
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
    parser.add_argument(
        "--batch-size",
        type=float,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=70,
        help="Percentile",
    )
    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    #args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_updates = args.total_timesteps // args.batch_size
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(
            env_id,
            base_seed=seed + idx,
            headless=True,
            start_xvfb=False,
            env_port=5555 + idx,
            server_port=30000 + idx,
            x_display=4,
        )
        if capture_video:
            if idx == 0 or idx < 0:
                env = gym.wrappers.RecordVideo(
                    env,
                    f"videos/{run_name}",
                    lambda x: x % 50 == 0,
                )
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env

    return thunk

class Network(nn.Module):
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
        return x

def array2dict(array):
        keys = ["terminated", "TimeLimit.truncated"]
        dict_ret = {}
        for key in keys:
            key_ar = []
            for di in array:
                if key in di:
                    key_ar.append(di[key])
                else:
                    key_ar.append(False)
            key_ar = np.array(key_ar)
            dict_ret[key] = key_ar
        return dict_ret

def train():
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
    key, network_key= jax.random.split(key, 2)

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
        gym.spaces.Discrete,
    ), "only discrete action space is supported"

    obs = envs.reset()

    network = Network()
    network_params = network.init(
        network_key,
        np.array([envs.single_observation_space.sample()]),
    )

    optimizer_def = optim.Adam(learning_rate = args.learning_rate)
    optimizer = optimizer_def.create(network_params)

    state = TrainState.create(apply_fn=network.apply, 
                                params=optimizer.target, tx=optimizer)
    
    for b in range(args.batch_size):
        obs = envs.reset()
        actions = []
        rewards = []

        for step in range(args.total_timesteps):
            action_probabilities = jax.nn.softmax(state.apply_fn(state.params, obs))
            action = jax.random.categorical(jax.random.PRNGKey(step), action_probabilities)[0]
            actions.append(action)

            (obs, reward, done, info) = envs.step(action)
            rewards.append(reward)

            if done[0]:
                print(
                    f"eval_episode={len(rewards)},"
                    f"episodic_return={rewards}",
                )

        rewards = jnp.array(rewards)
        reward_threshold = jnp.percentile(rewards, args.percentile) 
        filtered_actions = []
        filtered_rewards = []
        filtered_obs = []
        for idx, r in enumerate(rewards):
            if r >= reward_threshold:
                filtered_actions.append(actions[idx])
                filtered_rewards.append(r)
                filtered_obs.append(obs[idx])
        if len(filtered_actions) == 0:
            filtered_actions = actions
            filtered_rewards = rewards
            filtered_obs = obs
        
        loss = lambda net, obs, actions: jnp.mean(optax.softmax_cross_entropy(net.apply(obs), jax.nn.one_hot(actions, num_classes=envs.single_action_space.shape[-1])))

        grad_fn = jax.value_and_grad(loss)
        loss, grad = grad_fn(network, filtered_obs, filtered_actions)

        optimizer = optimizer.apply_gradient(grad)

        writer.add_scalar('Training Loss', loss, b)

    envs.close()

    for proc in psutil.process_iter():
        if proc.name() in ["minetest"]:
            proc.kill()

    xserver.terminate()
    writer.close()

if __name__ == "__main__":
    train()
