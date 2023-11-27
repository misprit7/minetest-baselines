import jax

import muax
from muax import nn


def train(args=None):
    support_size = 10 
    embedding_size = 8
    num_actions = 2
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
    model_path = muax.fit(model, 'CartPole-v1', 
                        max_episodes=1000,
                        max_training_steps=10000,
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

