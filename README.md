# CARLA PPO agent

## About the Project
This project concerns how we may design environments in order to facilitate the training of
deep reinforcement learning based autonomous driving agents. The goal of the project is to
provide a working deep reinforcement learning framework that can learn to drive in visually
complex environments, with a focus on providing a solution that:

1. Works out-of-the-box.
2. Learns in the shortest time posible, to make it easier to quickly iterate on and test our hypoteses.
3. Provides the nessecary metrics to compare agents between runs.

We have used the urban driving simulator [TODO](CARLA) (version 0.9.5) as our environment.

Find a [TODO](detailed project write-up here (thesis).)

TODO video link

## Contributions

- We provide two gym-like environments for CARLA, one that focuses on following a predetermined lap (see [TODO](CarlaEnv/carla_lap_env.py),) and another that is focused on training agents that can navigate from point A to point B (see [TODO](CarlaEnv/carla_route_env.py). While there are existing examples of gym-like environments for CARLA, there is no implementation that is officially endorsed by CARLA. Furthermore, most of the third-party environments do not provide an example of an agent that works out-of-the-box, or they may use outdated reinforcement learning algorithms, such as Deep Q-learning.
- We have provided analysis of optimal PPO parameters, environment designs, reward functions, etc. with the aim of finding the optimal setup to train reinforcement learning based autonomous driving agents (see Results chapter of [TODO](the project write-up) for further details.)
- We have provided an example that shows how VAEs can be used with CARLA for reinforcement learning purposes.
- We have shown that how we train and use a VAE can be consequential to the performance of a deep reinforcement learning agent, and have found that major improvements can be made by training the VAE to reconstruct semantic segmentation maps instead of reconstructing the RGB input itself. Training the VAE this way ensures that the VAE has a greater focus on encoding the semantics of the environment, which further aids in the learning of state representation learning-based agents.
- We have used our findings to devise a model that can reliably solve the CARLA lap environment in approximately 8 hours on average on a Nvidia GTX 970.
- We have provided an example of how sub-policies can be used to navigate with PPO, and we found it to have moderately success in the route environment (TODO add code for this).

## Related Work

1. [https://arxiv.org/abs/1807.00412](Learning to Drive in a Day) by Kendall _et. al._
This paper by researchers at Wayve describes a method that showed how state representation learning through a variational autoencoder can be used to train a car to follow a straight country road in approximately 15 minutes.
2. [https://towardsdatascience.com/learning-to-drive-smoothly-in-minutes-450a7cdb35f4](Learning to Drive Smoothly in Minutes) by Raffin _et. al._ This medium articles lays out the details of a method that was able to train an agent in a Donkey Car simulator in only 5 minutes, using a similar approach as (1). They further provide some solutions to the unstable steering we may observe when we train with the straight forward speed-as-reward reward formulation of Kendall.
3. [https://arxiv.org/abs/1710.02410](End-to-end Driving via Conditional Imitation Learning) by Codevilla _et. al._ This paper outlines an imitation learning model that is able to learn to navigate arbitrary routes by using multiple actor networks, conditioned on what the current manouver the vehicle should take is. We have used a similar approach in our route environment agent.

# How to Run

## Prerequisites

CARLA 0.9.5
Tensorflow 1.13
GPU with at least 4 GB VRAM

## 

Training

Evaluation

Collecting data

Training VAE

Inspecting VAE

Inspecting agent

TensorBoard

# File Overview

# Method

## Quick Overview

This is a high-level overview of our default approach.

1. Collect 10k 160x80x3 images by driving around manually.
2. Train a VAE to reconstruct the images.
3. Use the output of the trained VAE + (steering, throttle, speed) as input to a PPO-based agent.

## Analysis

### Reward functions

### Fail faster

### VAE trained on semantic maps

### Exploration Noise

### Environment Synchroncity


