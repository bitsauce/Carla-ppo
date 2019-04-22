import pygame
from pygame.locals import *
import carla
import gym
import time
import random
from collections import deque
from gym.utils import seeding
from wrappers import *
from hud import HUD
from planner import plan_route, compute_route_waypoints

# TODO:
# Some solution to avoid using the same env instance for training and eval

class CarlaRouteEnv(gym.Env):
    """
        This is a simple CARLA environment, where the goal is to drive in a lap
        around the outskirts of Town07. This environment can be used it to compare
        different models/reward functions in a realtively predictable environment.

        This version of CarlaEnv also uses the idea of failing faster, that is,
        the agent is reset to an earlier point on the current stretch road on failure
        until the agent surpasses it.

        To get this environment to run, start CARLA beforehand with:

        $> ./CarlaUE4.sh Town07

        Note that this environment has only been tested asynchronously.
        While CARLA does support synchronous simulation, I opted for testing
        asynchronously as this is more in-line with a real car.
        (see https://carla.readthedocs.io/en/latest/configuring_the_simulation/#synchronous-mode)
    """

    metadata = {
        "render.modes": ["human", "rgb_array", "rgb_array_no_hud", "state_pixels"]
    }

    def __init__(self, host="127.0.0.1", port=2000, viewer_res=(1280, 720), obs_res=(1280, 720),
                 reward_fn=None, encode_state_fn=None, action_smoothing=0.9, fps=30,
                 **kwargs):
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
            action_smoothing:
                Scalar used to smooth the incomming action signal.
                1.0 = max smoothing, 0.0 = no smoothing
            fps (int):
                FPS of the client. If fps <= 0 then use unbounded FPS.
                Note: Sensors will have a tick rate of fps when fps > 0, 
                otherwise they will tick as fast as possible.
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
        self.metadata["video.frames_per_second"] = self.fps = self.average_fps = fps
        self.spawn_point = 1
        self.action_smoothing = action_smoothing
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
                                  sensor_tick=1.0/self.fps)#,
                                  #camera_type="sensor.camera.semantic_segmentation", color_converter=carla.ColorConverter.CityScapesPalette)
            self.camera  = Camera(self.world, width, height,
                                  transform=camera_transforms["spectator"],
                                  attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e),
                                  sensor_tick=1.0/self.fps)#,
                                  #camera_type="sensor.camera.semantic_segmentation", color_converter=carla.ColorConverter.CityScapesPalette)
        except Exception as e:
            self.close()
            raise e

        # Reset env to set initial state
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, is_training=True):
        # Do a soft reset (teleport vehicle)
        self.vehicle.control.steer = float(0.0)
        self.vehicle.control.throttle = float(0.0)
        #self.vehicle.control.brake = float(0.0)
        self.vehicle.tick()
        
        # Generate waypoints along the lap
        self.start_wp, self.end_wp = [self.world.map.get_waypoint(p) for p in np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)]
        route = compute_route_waypoints(*plan_route(self.start_wp, self.end_wp), hop_resolution=1.0)
        self.route_waypoints = [p[0] for p in route]
        self.current_waypoint_index = 0
        self.vehicle.set_transform(self.start_wp.transform)
        self.vehicle.set_simulate_physics(False) # Reset the car's physics
        self.vehicle.set_simulate_physics(True)

        # Give 2 seconds to reset
        time.sleep(2.0)

        self.terminal_state = False # Set to True when we want to end episode
        self.closed = False         # Set to True when ESC is pressed
        self.extra_info = []        # List of extra info shown on the HUD
        self.observation = self.observation_buffer = None   # Last received observation
        self.viewer_image = self.viewer_image_buffer = None # Last received image to show in the viewer
        self.start_t = time.time()
        self.step_count = 0
        self.is_training = is_training
        
        # Metrics
        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.route_completion = 0.0

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
            "Reward: % 19.2f" % self.last_reward,
            "",
            "Route completion:  % 7.2f %%" % (self.route_completion * 100.0),
            "Distance traveled: % 7d m"    % self.distance_traveled,
            "Center deviance:   % 7.2f m"  % self.distance_from_center,
            "Avg center dev:    % 7.2f m"  % (self.center_lane_deviation / self.step_count),
            "Avg speed:      % 7.2f km/h"  % (3.6 * self.speed_accum / self.step_count)
        ])

        # Blit image from spectator camera
        self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

        # Superimpose current observation into top-right corner
        obs_h, obs_w = self.observation.shape[:2]
        view_h, view_w = self.viewer_image.shape[:2]
        pos = (view_w - obs_w - 10, 10)
        self.display.blit(pygame.surfarray.make_surface(self.observation.swapaxes(0, 1)), pos)

        # Render HUD
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = [] # Reset extra info list

        # Render to screen
        pygame.display.flip()

        if mode == "rgb_array_no_hud":
            return self.viewer_image
        elif mode == "rgb_array":
            # Turn display surface into rgb_array
            return np.array(pygame.surfarray.array3d(self.display), dtype=np.uint8).transpose([1, 0, 2])
        elif mode == "state_pixels":
            return self.observation

    def step(self, action):
        if self.closed:
            raise Exception("CarlaEnv.step() called after the environment was closed." +
                            "Check for info[\"closed\"] == True in the learning loop.")

        # Update fps metrics
        if self.fps <= 0:
            self.clock.tick()
        else:
            self.clock.tick_busy_loop(self.fps)
        if action is not None: # Dont update average fps on the first step() call
            self.average_fps = self.average_fps * 0.5 + self.clock.get_fps() * 0.5

        # Take action
        if action is not None:
            steer, throttle = [float(a) for a in action]
            #steer, throttle, brake = [float(a) for a in action]
            self.vehicle.control.steer    = self.vehicle.control.steer * self.action_smoothing + steer * (1.0-self.action_smoothing)
            self.vehicle.control.throttle = self.vehicle.control.throttle * self.action_smoothing + throttle * (1.0-self.action_smoothing)
            #self.vehicle.control.brake = self.vehicle.control.brake * self.action_smoothing + brake * (1.0-self.action_smoothing)
        
        # Get most recent observation and viewer image
        self.observation = self._get_observation()
        self.viewer_image = self._get_viewer_image()
        encoded_state = self.encode_state_fn(self)

        # Tick game
        self.world.tick()
        self.hud.tick(self.world, self.clock)

        # Get vehicle transform
        transform = self.vehicle.get_transform()

        # Keep track of closest waypoint on the route
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            # Check if we passed the next waypoint along the route
            next_waypoint_index = waypoint_index + 1
            wp = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                         vector(transform.location - wp.transform.location)[:2])
            if dot > 0.0: # Did we pass the waypoint?
                waypoint_index += 1 # Go to next waypoint
            else:
                break
        self.current_waypoint_index = waypoint_index

        # Calculate deviation from center of the lane
        self.current_waypoint = self.route_waypoints[ self.current_waypoint_index    % len(self.route_waypoints)]
        self.next_waypoint    = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
        self.distance_from_center = distance_to_line(vector(self.current_waypoint.transform.location),
                                                     vector(self.next_waypoint.transform.location),
                                                     vector(transform.location))
        self.center_lane_deviation += self.distance_from_center

        #self.world.debug.draw_point(wp0.transform.location, color=carla.Color(0, 255, 0), life_time=1.0)
        #self.world.debug.draw_point(wp1.transform.location, color=carla.Color(255, 0, 0), life_time=1.0)

        # Calculate distance traveled
        self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        # Accumulate speed
        self.speed_accum += self.vehicle.get_speed()
        
        # Get lap count
        self.route_completion = self.current_waypoint_index / len(self.route_waypoints)
        if self.route_completion >= 1:
            # End after 1 laps
            self.terminal_state = True
        
        # Call external reward fn
        self.last_reward = self.reward_fn(self)
        self.total_reward += self.last_reward
        self.step_count += 1

        # Check for ESC press
        pygame.event.pump()
        if pygame.key.get_pressed()[K_ESCAPE]:
            self.close()
            self.terminal_state = True
        
        return encoded_state, self.last_reward, self.terminal_state, { "closed": self.closed }

    def _get_observation(self):
        while self.observation_buffer is None:
            pass
        obs = self.observation_buffer.copy()
        self.observation_buffer = None
        return obs

    def _get_viewer_image(self):
        while self.viewer_image_buffer is None:
            pass
        image = self.viewer_image_buffer.copy()
        self.viewer_image_buffer = None
        return image

    def _on_collision(self, event):
        self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))

    def _on_invasion(self, event):
        text = ["%r" % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
        self.hud.notification("Crossed line %s" % " and ".join(text))

    def _set_observation_image(self, image):
        self.observation_buffer = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

def reward_fn(env):
    fail_faster = False
    if fail_faster:
        # If speed is less than 1.0 km/h after 5s, stop
        if time.time() - env.start_t > 5.0 and speed < 1.0 / 3.6:
            env.terminal_state = True

        # If distance from center > 3, stop
        if env.distance_from_center > 3.0:
            env.terminal_state = True
    return 0

if __name__ == "__main__":
    # Example of using CarlaEnv with keyboard controls
    env = CarlaRouteEnv(obs_res=(160, 80), reward_fn=reward_fn)
    action = np.zeros(env.action_space.shape[0])
    while True:
        env.reset(is_training=True)
        while True:
            # Process key inputs
            pygame.event.pump()
            keys = pygame.key.get_pressed()
            if keys[K_LEFT] or keys[K_a]:
                action[0] = -0.5
            elif keys[K_RIGHT] or keys[K_d]:
                action[0] = 0.5
            else:
                action[0] = 0.0
            action[0] = np.clip(action[0], -1, 1)
            action[1] = 1.0 if keys[K_UP] or keys[K_w] else 0.0

            # Take action
            obs, _, done, info = env.step(action)
            if info["closed"]: # Check if closed
                exit(0)
            env.render() # Render
            if done: break
    env.close()
