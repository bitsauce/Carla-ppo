import os
import random
import re
import shutil
import time
from collections import deque

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from skimage import transform

from ppo import PPO
from vae.models import ConvVAE, MlpVAE
from CarlaEnv.wrappers import angle_diff, vector
from utils import VideoRecorder, compute_gae
from vae_common import create_encode_state_fn, load_vae
from reward_functions import reward_functions

USE_ROUTE_ENVIRONMENT = False

if USE_ROUTE_ENVIRONMENT:
    from CarlaEnv.carla_route_env import CarlaRouteEnv as CarlaEnv
else:
    from CarlaEnv.carla_lap_env import CarlaLapEnv as CarlaEnv


def run_eval(env, model, video_filename=None):
    # Init test env
    state, terminal, total_reward = env.reset(is_training=False), False, 0
    rendered_frame = env.render(mode="rgb_array")

    # Init video recording
    if video_filename is not None:
        print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape, int(env.average_fps)))
        video_recorder = VideoRecorder(video_filename,
                                       frame_size=rendered_frame.shape,
                                       fps=env.average_fps)
        video_recorder.add_frame(rendered_frame)
    else:
        video_recorder = None

    episode_idx = model.get_episode_idx()

    # While non-terminal state
    while not terminal:
        env.extra_info.append("Episode {}".format(episode_idx))
        env.extra_info.append("Running eval...".format(episode_idx))
        env.extra_info.append("")

        # Take deterministic actions at test time (std=0)
        action, _ = model.predict(state, greedy=True)
        state, reward, terminal, info = env.step(action)

        if info["closed"] == True:
            break

        # Add frame
        rendered_frame = env.render(mode="rgb_array")
        if video_recorder is not None:
            video_recorder.add_frame(rendered_frame)
        total_reward += reward

    # Release video
    if video_recorder is not None:
        video_recorder.release()

    if info["closed"] == True:
        exit(0)

    return total_reward

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Runs the model in evaluation mode")
    
    # Model params
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to train. Output written to models/model_name")
    parser.add_argument("--reward_fn", type=str,
                        default="reward_speed_centering_angle_multiply",
                        help="Reward function to use. See reward_functions.py for more info.")
    parser.add_argument("--vae_model", type=str,
                        default="vae/models/seg_bce_cnn_zdim64_beta1_kl_tolerance0.0_data/",
                        help="Trained VAE model to load")
    parser.add_argument("--vae_model_type", type=str, default=None, help="VAE model type (\"cnn\" or \"mlp\")")
    parser.add_argument("--vae_z_dim", type=int, default=None, help="Size of VAE bottleneck")

    # Environment settings
    parser.add_argument("--synchronous", type=int, default=True, help="Set this to True when running in a synchronous environment")
    parser.add_argument("--fps", type=int, default=30, help="Set this to the FPS of the environment")
    parser.add_argument("--action_smoothing", type=float, default=0.0, help="Action smoothing factor")
    parser.add_argument("-start_carla", action="store_true", help="Automatically start CALRA with the given environment settings")

    # Recording    
    parser.add_argument("--record_to_file", type=str, default=None, help="File to record evaluation video to (outputs in .avi format)")

    args = parser.parse_args()

    # Load VAE
    vae = load_vae(args.vae_model, args.vae_z_dim, args.vae_model_type)

    # Create state encoding fn
    measurements_to_include = set(["steer", "throttle", "speed"])
    encode_state_fn = create_encode_state_fn(vae, measurements_to_include)

    # Create env
    print("Creating environment...")
    env = CarlaEnv(obs_res=(160, 80),
                   action_smoothing=args.action_smoothing,
                   encode_state_fn=encode_state_fn,
                   reward_fn=reward_functions[args.reward_fn],
                   synchronous=args.synchronous,
                   fps=args.fps,
                   start_carla=args.start_carla)


    # Set seeds
    seed = 0
    if isinstance(seed, int):
        tf.random.set_random_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        env.seed(seed)

    # Create model
    print("Creating model...")
    input_shape = np.array([vae.z_dim + len(measurements_to_include)])
    model = PPO(input_shape, env.action_space,
                model_dir=os.path.join("models", args.model_name))
    model.init_session(init_logging=False)
    model.load_latest_checkpoint()

    # Run eval
    print("Running eval...")
    run_eval(env, model, video_filename=args.record_to_file)

    # Close env
    print("Done!")
    env.close()