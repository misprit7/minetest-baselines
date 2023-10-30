from minetest_baselines.jax.train_dqn_cleanrl import train as train_dqn
from minetest_baselines.jax.train_ppo_cleanrl import train as train_ppo
from minetest_baselines.jax.train_reinforce_cleanrl import train as train_reinforce

ALGOS = {"dqn": train_dqn, "ppo": train_ppo, "reinforce": train_reinforce}
