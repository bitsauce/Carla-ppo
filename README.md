# About the Project
This repository documents my experiments in trying to train a car to follow
a long stretch of road in the urban driving simulator, CARLA.

Inspired by Learning to Drive in a Day - a paper that describes a method that
was able to teach a car to follow a country road in approximately 15 minutes -
I was interested to find out if a similar approach can work in more complicated environments.

## Contributions

- This project features a OpenAI gym-like environment that is easy to customize (see carla_env.py)
- The importance of failing faster to reduce training time
- Analysis of different reward formulations; how to we make the car follow a predefined speed limit?
- Learn to complete a 1245m lap with hills and complicated background scenery in ~6 hours on a Nvidia GTX 970
- Self-contained, less general - but easier to understand - code that can be used for easier experimenting.

## Method

1) Collect 10k 160x80x3 images by driving around manually
2) Train a VAE to reconstruct the images
3) Use output of trained VAE + (steering, throttle, speed) as input to a PPO-based agent

## Analysis

### Reward function

### Effect of background clutter

#### VAE trained on semantic maps

### Fail-faster
