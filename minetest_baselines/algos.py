from minetest_baselines.jax.train_dqn_cleanrl import train as train_dqn
from minetest_baselines.jax.train_ppo_cleanrl import train as train_ppo
from minetest_baselines.jax.train_muax import train as train_muax

ALGOS = {"dqn": train_dqn, "ppo": train_ppo, "muax": train_muax}
