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
from CarlaEnv.carla_env import CarlaLapEnv as CarlaEnv
from CarlaEnv.wrappers import angle_diff, vector
from utils import VideoRecorder, compute_gae
from common import create_encode_state_fn, reward_fn, load_vae

def run_eval(test_env, model, video_filename=None):
    # Init test env
    state, terminal, total_reward = test_env.reset(is_training=False), False, 0
    rendered_frame = test_env.render(mode="rgb_array")

    # Init video recording
    if video_filename is not None:
        print("Recording video to {} ({}x{}x{}@{}fps)".format(video_filename, *rendered_frame.shape, int(test_env.average_fps)))
        video_recorder = VideoRecorder(video_filename,
                                       frame_size=rendered_frame.shape,
                                       fps=test_env.average_fps)
        video_recorder.add_frame(rendered_frame)
    else:
        video_recorder = None

    episode_idx = model.get_episode_idx()

    # While non-terminal state
    while not terminal:
        test_env.extra_info.append("Episode {}".format(episode_idx))
        test_env.extra_info.append("Running eval...".format(episode_idx))
        test_env.extra_info.append("")

        # Take deterministic actions at test time (noise_scale=0)
        action, _ = model.predict([state], greedy=True)
        state, reward, terminal, info = test_env.step(action)

        if info["closed"] == True:
            break

        # Add frame
        rendered_frame = test_env.render(mode="rgb_array")
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
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--vae_model", type=str, default="bce_cnn_zdim64_beta1_kl_tolerance0.0_data")
    parser.add_argument("--vae_model_type", type=str, default=None)
    parser.add_argument("--vae_z_dim", type=int, default=None)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--action_smoothing", type=float, default=0.0)
    parser.add_argument("--record_to_file", type=str, default=None)
    args = parser.parse_args()

    # Load VAE
    vae = load_vae(args.vae_model, args.vae_z_dim, args.vae_model_type)

    # Create state encoding fn
    measurements_to_include = set(["steer", "throttle", "speed"])
    encode_state_fn = create_encode_state_fn(vae, measurements_to_include)

    # Create env
    print("Creating environment...")
    env = CarlaEnv(obs_res=(160, 80),
                   encode_state_fn=encode_state_fn, reward_fn=reward_fn,
                   action_smoothing=args.action_smoothing, fps=args.fps)
    env.seed(0)

    # Create model
    print("Creating model...")
    input_shape = np.array([vae.z_dim + len(measurements_to_include)])
    model = PPO(input_shape, env.action_space, model_dir=os.path.join("models", args.model_name))
    model.init_session(init_logging=False)
    model.load_latest_checkpoint()

    # Run eval
    print("Running eval...")
    run_eval(env, model, video_filename=args.record_to_file)

    # Close env
    print("Done!")
    env.close()