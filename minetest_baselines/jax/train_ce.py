import argparse
import os
import time
import random
import flax
from collections import namedtuple
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import psutil

import minetest_baselines.tasks

HIDDEN_SIZE = 128
BATCH_SIZE = 16
PERCENTILE = 75
NUM_ITERS = 5

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-name",
        type=str,
        default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment",
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=1, 
        help="seed of the experiment"
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
        default="minetest-baselines-ce",
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
        default=1,
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
        "--num-envs",
        type=int,
        default=1,
        help="the number of parallel game environments",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="the number of episodes per iteration",
    )
    parser.add_argument(
        "--percentile",
        type=int,
        default=70,
        help="the percentile of episodes that should be taken",
    )

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)
    return args

Episode = namedtuple('Episode', field_names=['reward', 'steps'])
EpisodeStep = namedtuple('EpisodeStep', field_names=['observation', 'action'])

def make_env(env_id, seed, idx, capture_video, video_frequency, run_name):
    def thunk():
        env = gym.make(
            env_id,
            base_seed=seed + idx,
            headless=True,
            start_xvfb=False,
            env_port=5555 + idx,
            server_port=30000 + idx,
            render_mode="rgb_array",
        )
        if capture_video:
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

def iterate_batches(env, net, batch_size, key, params = None):
    batch = []
    episode_reward = 0.0
    episode_steps = []
    obs, _ = env.reset()
    # params = net.init(key, jnp.array([obs[0]]))
    while True:
        obs_v = jnp.array([obs[0]])
        netted = net.apply(params, obs_v)
        act_probs_v = nn.softmax(netted)
        act_probs = jnp.array(act_probs_v)[0]
        key, subkey = jax.random.split(key)
        action = int(jax.random.choice(subkey, len(act_probs), p=act_probs))
        next_obs, reward, is_done, truncated, infos = env.step(np.array([action]))
        episode_reward += reward
        episode_steps.append(EpisodeStep(observation=obs, action=action))

        if is_done[0] or truncated[0]:
            batch.append(Episode(reward=episode_reward, steps=episode_steps))
            print(f"EPISODE COMPLETE. Reward was {episode_reward}")
            episode_reward = 0.0
            episode_steps = []
            next_obs, _ = env.reset()
            print(len(batch))
            if len(batch) == batch_size:
                yield batch, key
                batch = []
        obs = next_obs

def filter_batch(batch, percentile):
    rewards = list(map(lambda s: s.reward, batch))
    reward_bound = np.percentile(rewards, percentile)
    reward_mean = float(np.mean(rewards))

    train_obs = []
    train_act = []
    for example in batch:
        if example.reward < reward_bound:
            continue
        train_obs.extend(map(lambda step: step.observation, example.steps))
        train_act.extend(map(lambda step: step.action, example.steps))

    train_obs_v = jnp.array(train_obs)
    train_act_v = jnp.array(train_act)
    return train_obs_v, train_act_v, reward_bound, reward_mean

class CENetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        # shape = (batch, stack, img_x, img_y)
        batch_dim = x.shape[0]
        x = x / 255.0
        x = jnp.transpose(x, (0, 2, 3, 1))  # shape = (batch, img x, img y, stack)
        x = nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), padding="VALID")(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(3, 3), strides=(1, 1), padding="VALID")(x)
        x = nn.relu(x)
        x = x.reshape((batch_dim, -1))  # shape = (batch, output features)
        x = nn.Dense(512)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x

def train(args = None):
    def loss_fn(params, model, inputs, targets):
        logits = model.apply(params, inputs)
        log_prob = jax.nn.log_softmax(logits)
        one_hot_actions = jax.nn.one_hot(targets, num_classes=logits.shape[-1])
        neg_log_prob = jnp.sum(-one_hot_actions * log_prob, axis=-1)
        return jnp.mean(neg_log_prob)

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

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

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

    obs, _ = envs.reset()

    ce_network = CENetwork(action_dim=envs.single_action_space.n)
    print(envs.single_action_space.n)
    optimizer = optax.adam(0.001)
    # Initialize TrainState
    params = ce_network.init(key, jnp.array([obs[0]]))
    state = flax.training.train_state.TrainState.create(apply_fn=ce_network.apply, params=params, tx=optimizer)

    for iter_no, (batch, key) in enumerate(iterate_batches(envs, ce_network, BATCH_SIZE, key, params=state.params)):
        obs_v, acts_v, reward_b, reward_m = filter_batch(batch, PERCENTILE)
        obs_v = jnp.array([obs_v[0][0]])
        
        loss, grad = jax.value_and_grad(loss_fn)(state.params, ce_network, obs_v, acts_v)
        state = state.apply_gradients(grads=grad)
        
        print("%d: loss=%.3f, reward_mean=%.1f, reward_bound=%.1f" % (
            iter_no, loss, reward_m, reward_b))

        if args.track:
            wandb.log({"loss": loss, "mean reward": reward_m})

        if iter_no == NUM_ITERS:
            break
    
    if args.track:
        wandb.finish()

    # Close training envs
    envs.close()
    # kill any remaining minetest processes
    for proc in psutil.process_iter():
        if proc.name() in ["minetest"]:
            proc.kill()

if __name__ == "__main__":
    train()