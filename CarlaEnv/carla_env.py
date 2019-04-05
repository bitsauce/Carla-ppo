import pygame
import carla
import gym
import time
import random
from gym.utils import seeding
from wrappers import *
from keyboard_control import KeyboardControl
from hud import HUD

class CarlaEnv(gym.Env):
    """
        To get this environment to run, start CARLA beforehand with:

        $> ./CarlaUE4.sh Town07 -benchmark -fps=30
        
        Or replace "Town07" with your map of choice.

        The benchmark flag is used to set a fixed time-step update loop,
        making the delta time between observations more regular,
        and the tick rate is set to 30 updates per second.
    """

    metadata = {
        "render.modes": ["human", "rgb_array", "rgb_array_no_hud", "state_pixels"]
    }

    def __init__(self, host="127.0.0.1", port=2000, viewer_res=(1280, 720), obs_res=(1280, 720),
                 reward_fn=None, encode_state_fn=None, fps=30, spawn_point=1):
        """
            Initializes a gym-like environment that can be used to interact with CARLA.

            Connects to a running CARLA enviromment (tested on version 0.9.4) and
            spwans a lincoln mkz2017 passenger car with automatic transmission.
            
            This vehicle can be controlled using the step() function,
            taking an action that consists of [steering_angle, throttle].

            host (string):
                IP address of the CARLA host
            port (short):
                Port used to connect to CARLA
            viewer_res (int, int):
                Resolution of the spectator camera (placed behind the vehicle by default)
                as a (width, height) tuple
            obs_res (int, int):
                Resolution of the observation camera (placed on the dashboard by default)
                as a (width, height) tuple
            reward_fn (function):
                Custom reward function that is called every step.
                If None, no reward function is used.
            encode_state_fn (function):
                Function that takes the image (of obs_res resolution) from the
                observation camera and encodes it to some state vector to returned
                by step(). If None, step() returns the full image.
            fps (int):
                FPS of the sensors
            spawn_point (int):
                Index of the spawn point to use (spawn point 1 of Town07 is a good starting point)
        """

        # Initialize pygame for visualization
        pygame.init()
        pygame.font.init()
        width, height = viewer_res
        if obs_res is None:
            out_width, out_height = width, height
        else:
            out_width, out_height = obs_res
        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

        # Setup gym environment
        self.seed()
        self.action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32) # steer, throttle
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(*obs_res, 3), dtype=np.float32)
        self.metadata["video.frames_per_second"] = fps
        self.spawn_point = spawn_point
        self.encode_state_fn = (lambda x: x) if not callable(encode_state_fn) else encode_state_fn
        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn

        self.world = None
        try:
            # Connect to carla
            self.client = carla.Client(host, port)
            self.client.set_timeout(2.0)

            # Create world wrapper
            self.world = World(self.client)

            # Create vehicle and attach camera to it
            self.vehicle = Vehicle(self.world, self.world.map.get_spawn_points()[self.spawn_point],
                                   on_collision_fn=lambda e: self._on_collision(e),
                                   on_invasion_fn=lambda e: self._on_invasion(e))

            # Create hud
            self.hud = HUD(width, height)
            self.hud.set_vehicle(self.vehicle)
            self.world.on_tick(self.hud.on_world_tick)

            # Create cameras
            self.dashcam = Camera(self.world, out_width, out_height,
                                  transform=camera_transforms["dashboard"],
                                  attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_image(e),
                                  sensor_tick=1.0/self.metadata["video.frames_per_second"])
            self.camera  = Camera(self.world, width, height,
                                  transform=camera_transforms["spectator"],
                                  attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e),
                                  sensor_tick=1.0/self.metadata["video.frames_per_second"])

            # Attach keyboard controls
            self.controller = KeyboardControl(self.world, self.vehicle, self.hud)
            self.controller.set_enabled(False)

            # Reset env to set initial state
            self.reset()
        except Exception as e:
            self.close()
            raise e

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        # Do a soft reset (teleport vehicle)
        self.vehicle.control.steer = float(0.0)
        self.vehicle.control.throttle = float(0.0)
        self.vehicle.tick()
        if False:#randomize_spawn
            self.vehicle.set_transform(random.choice(self.world.map.get_spawn_points()))
        else:
            self.vehicle.set_transform(self.world.map.get_spawn_points()[self.spawn_point])
        self.vehicle.set_simulate_physics(False) # Resets the car physics
        self.vehicle.set_simulate_physics(True)

        # Give 2 seconds to reset
        time.sleep(2.0)

        self.terminal_state = False # Set to True when we want to end episode
        self.closed = False         # Set to True when ESC is pressed
        self.extra_info = []        # List of extra info shown on the HUD
        self.current_obs = None     # Most recent observation
        self.viewer_image = None    # Most recent rendered image
        self.start_t = time.time()
        self.step_count = 0
        
        # Metrics
        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0

        # Return initial observation
        return self.step(None)[0]

    def close(self):
        pygame.quit()
        if self.world is not None:
            self.world.destroy()
        self.closed = True

    def render(self, mode="human"):
        # Add metrics to HUD
        self.extra_info.extend([
            "Distance traveled: % 8.2fm" % self.distance_traveled,
            "Avg center dev:    % 8.2fm" % (self.center_lane_deviation / self.step_count)
        ])

        # Render
        image = self.viewer_image.copy()
        surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        self.display.blit(surface, (0, 0))
        self.hud.render(self.display, extra_info=self.extra_info)
        pygame.display.flip()
        self.extra_info = [] # Reset extra info list

        if mode == "rgb_array_no_hud":
            return image
        elif mode == "rgb_array":
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2]) # Turn surface into rgb_array
        elif mode == "state_pixels":
            return self.current_obs

    def step(self, action):
        if self.closed:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")

        # Take action
        if action is not None:
            self.vehicle.control.steer = float(action[0])
            self.vehicle.control.throttle = float(action[1])
            #self.vehicle.control.brake = float(action[2])

        # Get most recent observation
        self.observation = self._get_observation()
        encoded_state = self.encode_state_fn(self)

        # Tick game
        self.clock.tick()
        self.world.tick()
        self.hud.tick(self.world, self.clock)

        # Calculate deviation from center of the lane
        transform = self.vehicle.get_transform()
        self.closest_waypoint = self.vehicle.get_closest_waypoint()       # Store closest waypoint for reuse
        loc, wp_loc = carla_as_array(self.closest_waypoint.transform.location), carla_as_array(transform.location)
        # world.debug.draw_point(wp_loc, life_time=1.0)                  # Draw point on waypoint to visualize
        self.distance_from_center = np.linalg.norm(loc[:2] - wp_loc[:2]) # XY-distance from center
        self.center_lane_deviation += self.distance_from_center

        # Calculate distance traveled
        self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        self.speed_accum += self.vehicle.get_speed()
        
        # Call external reward fn
        reward = self.reward_fn(self)
        self.total_reward += reward
        self.step_count += 1

        # Check for ESC press
        if self.controller.parse_events(self.client, self.clock):
            self.close()
            self.terminal_state = True
        
        return encoded_state, reward, self.terminal_state, { "closed": self.closed }

    def _get_observation(self):
        while self.current_obs is None:
            pass
        obs = self.current_obs.copy()
        self.current_obs = None
        return obs

    def _get_viewer_image(self):
        while self.viewer_image is None:
            pass
        image = self.viewer_image.copy()
        self.viewer_image = None
        return image

    def _on_collision(self, event):
        self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))

    def _on_invasion(self, event):
        text = ["%r" % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
        self.hud.notification("Crossed line %s" % " and ".join(text))

    def _set_observation_image(self, image):
        self.current_obs = image

    def _set_viewer_image(self, image):
        self.viewer_image = image

if __name__ == "__main__":
    # Example of using CarlaEnv with keyboard controls
    from pygame.locals import *
    env = CarlaEnv(obs_res=(160, 80), spawn_point=10)
    action = np.zeros(env.action_space.shape[0])
    while True:
        env.reset()
        while True:
            # Process key inputs
            keys = pygame.key.get_pressed()
            steer_increment = 5e-4 * env.clock.get_time()
            if keys[K_LEFT] or keys[K_a]:
                action[0] -= steer_increment
            elif keys[K_RIGHT] or keys[K_d]:
                action[0] += steer_increment
            else:
                action[0] = 0.0
            action[0] = np.clip(action[0], -1, 1)
            action[1] = 1.0 if keys[K_UP] or keys[K_w] else 0.0

            obs, _, done, info = env.step(action) # Take action
            if info["closed"]: # Check if closed
                exit(0)
            env.render() # Render
            if done: break
    env.close()
