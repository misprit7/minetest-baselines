from minetest_baselines.jax.train_dqn_cleanrl import train as train_dqn
from minetest_baselines.jax.train_ppo_cleanrl import train as train_ppo
from minetest_baselines.jax.train_ddpg_cleanrl import train as train_ddpg

ALGOS = {"dqn": train_dqn, "ppo": train_ppo, "ddpg": train_ddpg}
