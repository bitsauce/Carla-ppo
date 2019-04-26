
import numpy as np
import re
import time
from CarlaEnv.wrappers import angle_diff, vector
from vae.models import ConvVAE, MlpVAE

def load_vae(model_dir, z_dim=None, model_type=None):
    """
        Loads and returns a pretrained VAE
    """
    
    # Parse z_dim and model_type from name if None
    if z_dim is None: z_dim = int(re.findall("zdim(\d+)", model_dir)[0])
    if model_type is None: model_type = "mlp" if "mlp" in model_dir else "cnn"
    VAEClass = MlpVAE if model_type == "mlp" else ConvVAE

    # Load pre-trained variational autoencoder
    vae_source_shape = np.array([80, 160, 3])
    vae = VAEClass(source_shape=vae_source_shape,
                   target_shape=np.array([80,160,1]),
                   z_dim=z_dim, models_dir="vae",
                   model_dir=model_dir,
                   training=False)
    vae.init_session(init_logging=False)
    if not vae.load_latest_checkpoint():
        raise Exception("Failed to load VAE")
    return vae

def preprocess_frame(frame):
    frame = frame.astype(np.float32) / 255.0
    return frame

def create_encode_state_fn(vae, measurements_to_include):
    """
        Returns a function that encodes the current state of
        the environment into some feature vector.
    """

    # Turn into bool array for performance
    measure_flags = ["steer" in measurements_to_include,
                     "throttle" in measurements_to_include,
                     "speed" in measurements_to_include,
                     "orientation" in measurements_to_include]

    def encode_state(env):
        # Encode image with VAE
        frame = preprocess_frame(env.observation)
        encoded_state = vae.encode([frame])[0]
        
        # Append measurements
        measurements = []
        if measure_flags[0]: measurements.append(env.vehicle.control.steer)
        if measure_flags[1]: measurements.append(env.vehicle.control.throttle)
        if measure_flags[2]: measurements.append(env.vehicle.get_speed())

        # Orientation could be usedful for predicting movements that occur due to gravity
        if measure_flags[3]: measurements.extend(vector(env.vehicle.get_forward_vector()))

        encoded_state = np.append(encoded_state, measurements)
        
        return encoded_state
    return encode_state

def reward_fn(env):
    reward = 0
    max_distance = 3.0 # Max distance from center before terminating
    target_speed = 20.0 # kmh
    terminal_reason = "Running..."

    # If speed is less than 1.0 km/h after 5s, stop
    speed = env.vehicle.get_speed()
    if time.time() - env.start_t > 5.0 and speed < 1.0 / 3.6:
        env.terminal_state = True
        terminal_reason = "Vehicle stopped"
        reward -= 10

    # If distance from center > max distance, stop
    if env.distance_from_center > max_distance:
        env.terminal_state = True
        terminal_reason = "Off-track"
        reward -= 10

    # Kendall Reward:
    # - Reward = Speed
    # - Std = 0.1
    # Results:
    """
    if not env.terminal_state:
        reward += speed
    """

    # Reward 1:
    # - Reward = Normalized speed reward linearly increasing until 1 at target speed,
    #            then decreasing linearly starting at 1 with greater speeds
    #          + normalized centering reward [0, 1]
    # - Target speed = 20
    # - Std = 0.1
    # Results:
    # - Steering fluctuating
    # - Best run did 2 laps
    """
    if not env.terminal_state:
        norm_speed = 3.6 * speed / target_speed
        reward += np.minimum(norm_speed, 2.0 - norm_speed)
        reward += 1.0 - env.distance_from_center / max_distance
    """

    # Reward 1 run 2:
    # - Reward = Positive speed reward for being close to target speed,
    #            however, quick decline in reward beyond target speed
    #          + normalized centering reward [0, 1]
    # - Target speed = 20
    # - Std = 0.1
    # - Hypothesis: Giving the agent some leeway may improve learning
    # Results:
    """
    min_speed = 15.0 # km/h
    max_speed = 25.0 # km/h
    if not env.terminal_state:
        speed_kmh = 3.6 * speed
        if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
            speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
        elif speed_kmh > target_speed:                # When speed is in [target_speed, inf]
                                                      # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
            speed_reward = 1.0 - (speed_kmh-target_speed) / (max_speed-target_speed)
        else:                                         # Otherwise
            speed_reward = 1.0                        # Return 1 for speeds in range [min_speed, target_speed]

        # Interpolated from 1 when centered to 0 when 3 m from center
        centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

        # Final reward
        reward += speed_reward + centering_factor
    """

    # Reward 1 run 3:
    # - Reward = Positive speed reward for being close to target speed,
    #            however, quick decline in reward beyond target speed
    #          + normalized centering reward [0, 1]
    # - Target speed = 20
    # - Std = 0.1
    # - Hypothesis: Give angle reward to aid learning
    # Results:
    """
    # Get angle difference between closest waypoint and vehicle forward vector
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)

    min_speed = 15.0 # km/h
    max_speed = 25.0 # km/h
    if not env.terminal_state:
        speed_kmh = 3.6 * speed
        if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
            speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
        elif speed_kmh > target_speed:                # When speed is in [target_speed, inf]
                                                      # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
            speed_reward = 1.0 - (speed_kmh-target_speed) / (max_speed-target_speed)
        else:                                         # Otherwise
            speed_reward = 1.0                        # Return 1 for speeds in range [min_speed, target_speed]

        # Interpolated from 1 when centered to 0 when 3 m from center
        centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

        # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
        angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)

        # Final reward
        reward += speed_reward + centering_factor + angle_factor
    """


    # Reward 1 run 4:
    # - Reward = Normalized speed reward linearly increasing until 1 at target speed,
    #            then decreasing linearly starting at 1 with greater speeds
    #          * normalized centering reward [0, 1]
    # - Target speed = 20
    # - Std = 0.1 and 1.0
    # - Hypothesis: Multiplication acts more like AND
    # Results:
    # 1) Std = 0.1 is unreliable and suseptible to getting stuck
    #    However, increasing std makes movement more zig-zagy;
    #    Perhaps give more priority to staying centered? 
    """
    strength = 4
    if not env.terminal_state:
        norm_speed = 3.6 * speed / target_speed
        speed_reward = np.minimum(norm_speed, 2.0 - norm_speed)
        centering_factor = 1.0 - env.distance_from_center / max_distance
        reward += speed_reward * np.power(centering_factor, strength)
    """

    # Reward 2:
    # - Reward = Normalized speed reward [0, 1] until target speed, then 0
    #          + normalized centering reward [0, 1]
    # - Target speed = 20
    # - Std = 0.1
    # Results:
    """
    if not env.terminal_state:
        norm_speed = 3.6 * speed / target_speed
        if norm_speed <= 1.0:
            reward += norm_speed
        reward += 1.0 - env.distance_from_center / max_distance
    """

    # Reward 3:
    #
    # | speed | throttle | reward |
    # +-------+----------+--------+
    # | 0     | 0        |  0     |
    # | 1     | 0        |  0     |
    # | 1     | 1        |  0     |
    # | 0     | 1        |  1     |
    # | 2     | 1        | -1     |
    # | 0     | 0.5      |  0.5   |
    # | 0.5   | 0.5      |  0.25  |
    #
    # throttle - speed * throttle = throttle * (1 - speed)
    # =>
    # - Reward = More reward when we apply throttle on low speeds than throttle on speeds close to target speed
    #          + normalized centering reward [0, 1]
    # - Target Speed = 20
    # - Std = 0.1
    # - Hypothesis: Throttle may be more important when we're moving slowly
    # Results:
    """
    if not env.terminal_state:
        norm_speed = 3.6 * speed / target_speed
        reward += env.vehicle.control.throttle * (1.0 - norm_speed)
        reward += 1.0 - env.distance_from_center / max_distance
    """

    # Reward 4:
    # - Reward = Centering is more incentivized when speed is higher
    # - Target Speed = 20
    # - Std = 0.1
    # - Hypothesis: Learn to speed up first, then center
    # Results:
    """
    if not env.terminal_state:
        norm_speed = 3.6 * speed / target_speed
        norm_speed2 = np.minimum(norm_speed, 2.0 - norm_speed)
        reward += norm_speed2
        reward += (1.0 - env.distance_from_center / max_distance) * norm_speed2
    """

    # Reward 5:
    # - Reward = Positive speed reward for being close to target speed,
    #            however, quick decline in reward beyond target speed
    #          + centering factor (1 when centered, 0 when not)
    # - Target Speed = 20
    # - Std = 0.1
    # - Hypothesis: Give the agent some leeway when it's close to target speed,
    #               but penalize quickly beyond that
    # Results:
    """
    min_speed = 15.0 # km/h
    max_speed = 25.0 # km/h
    if not env.terminal_state:
        speed_kmh = 3.6 * speed
        if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
            speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
        elif speed_kmh > target_speed:                # When speed is in [target_speed, inf]
                                                      # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
            speed_reward = 1.0 - (speed_kmh-target_speed) / (max_speed-target_speed)
        else:                                         # Otherwise
            speed_reward = 1.0                        # Return 1 for speeds in range [min_speed, target_speed]


        # Interpolated from 1 when 
    # Get angle difference between closest waypoint and vehicle forward vector
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)
aligned with the road to 0 when +/- 20 degress of road
        #angle_factor = max(1.0 - a
    # Get angle difference between closest waypoint and vehicle forward vector
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)
bs(angle / np.deg2rad(20)), 0.0)

    # Get angle difference between closest waypoint and vehicle forward vector
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)

        # Interpolated from 1 when 
    # Get angle difference between closest waypoint and vehicle forward vector
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)
centered to 0 when 3 m from center
        centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

        # Final reward
        reward += speed_reward + centering_factor
    """

    # Reward 5 run 2:
    # - Reward = Positive speed reward for being close to target speed,
    #            however, quick decline in reward beyond target speed
    #          + centering factor (1 when centered, 0 when not)
    #          + angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    # - Target Speed = 20
    # - Std = 0.1
    # - Hypothesis: Angle reward will encourage the agent to stay aligned with the road,
    #               resulting in less fluctuations in steering and quicker learning
    # Results:
    """
    # Get angle difference between closest waypoint and vehicle forward vector
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)

    min_speed = 15.0 # km/h
    max_speed = 25.0 # km/h
    if not env.terminal_state:
        speed_kmh = 3.6 * speed
        if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
            speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
        elif speed_kmh > target_speed:                # When speed is in [target_speed, inf]
                                                      # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
            speed_reward = 1.0 - (speed_kmh-target_speed) / (max_speed-target_speed)
        else:                                         # Otherwise
            speed_reward = 1.0                        # Return 1 for speeds in range [min_speed, target_speed]


        # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
        angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)

        # Interpolated from 1 when centered to 0 when 3 m from center
        centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

        # Final reward
        reward += speed_reward + centering_factor + angle_factor
    """

    # Reward 6:
    # - Reward = Positive speed reward for being close to target speed,
    #            however, quick decline in reward beyond target speed
    #          * centering factor (1 when centered, 0 when not)
    #          * angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    # - Target Speed = 20
    # - Std = 0.1
    # - Hypothesis: Multiplication acts more like an AND statement
    #               (we want the agent to drive forward AND be centered AND aligned with the road)
    # Results:
    
    # Get angle difference between closest waypoint and vehicle forward vector
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)

    min_speed = 15.0 # km/h
    max_speed = 25.0 # km/h
    if not env.terminal_state:
        speed_kmh = 3.6 * speed
        if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
            speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
        elif speed_kmh > target_speed:                # When speed is in [target_speed, inf]
                                                      # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
            speed_reward = 1.0 - (speed_kmh-target_speed) / (max_speed-target_speed)
        else:                                         # Otherwise
            speed_reward = 1.0                        # Return 1 for speeds in range [min_speed, target_speed]


        # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
        #angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)
        angle_factor = max(max(1.0 - abs(angle / np.deg2rad(180)), 0.0)**4, 0.1) # 6.2

        # Interpolated from 1 when centered to 0 when 3 m from center
        centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.1)

        # Final reward
        reward += speed_reward * centering_factor * angle_factor
    

    # Reward 6 run 2:
    # - Reward = Positive speed reward for being close to target speed,
    #            however, quick decline in reward beyond target speed
    #          * centering factor (1 when centered, 0 when not)
    #          * angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    # - Target Speed = 20
    # - Std = 0.1
    # - Max distance = 0.5
    # - Hypothesis: Multiplication acts more like an AND statement
    #               (we want the agent to drive forward AND be centered AND aligned with the road)
    # Results:
    """
    # Get angle difference between closest waypoint and vehicle forward vector
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)

    min_speed = 15.0 # km/h
    max_speed = 25.0 # km/h
    if not env.terminal_state:
        speed_kmh = 3.6 * speed
        if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
            speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
        elif speed_kmh > target_speed:                # When speed is in [target_speed, inf]
                                                      # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
            speed_reward = 1.0 - (speed_kmh-target_speed) / (max_speed-target_speed)
        else:                                         # Otherwise
            speed_reward = 1.0                        # Return 1 for speeds in range [min_speed, target_speed]


        # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
        angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)

        # Interpolated from 1 when centered to 0 when 3 m from center
        centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

        # Final reward
        reward += speed_reward * centering_factor * angle_factor
    """
    
    # Reward 6 run 3:
    # - Reward = Positive speed reward for being close to target speed,
    #            however, quick decline in reward beyond target speed
    #          * centering factor (1 when centered, 0 when not)
    #          * angle factor (1 when aligned with the road, 0 when more than 20 degress off)
    # - Target Speed = 20
    # - Std = 1.0
    # - Hypothesis: Multiplication acts more like an AND statement
    #               (we want the agent to drive forward AND be centered AND aligned with the road)
    # Results:
    """
    # Get angle difference between closest waypoint and vehicle forward vector
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    angle  = angle_diff(fwd, wp_fwd)

    min_speed = 15.0 # km/h
    max_speed = 25.0 # km/h
    if not env.terminal_state:
        speed_kmh = 3.6 * speed
        if speed_kmh < min_speed:                     # When speed is in [0, min_speed] range
            speed_reward = speed_kmh / min_speed      # Linearly interpolate [0, 1] over [0, min_speed]
        elif speed_kmh > target_speed:                # When speed is in [target_speed, inf]
                                                      # Interpolate from [1, 0, -inf] over [target_speed, max_speed, inf]
            speed_reward = 1.0 - (speed_kmh-target_speed) / (max_speed-target_speed)
        else:                                         # Otherwise
            speed_reward = 1.0                        # Return 1 for speeds in range [min_speed, target_speed]

        # Interpolated from 1 when aligned with the road to 0 when +/- 20 degress of road
        angle_factor = max(1.0 - abs(angle / np.deg2rad(20)), 0.0)

        # Interpolated from 1 when centered to 0 when 3 m from center
        centering_factor = max(1.0 - env.distance_from_center / max_distance, 0.0)

        # Final reward
        reward += speed_reward * centering_factor * angle_factor
    """

    if env.terminal_state:
        env.extra_info.extend([
            terminal_reason,
            ""
        ])
    return reward