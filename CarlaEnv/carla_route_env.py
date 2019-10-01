import os
import subprocess
import time

import carla
import gym
import pygame
from gym.utils import seeding
from pygame.locals import *

from hud import HUD
from planner import RoadOption, compute_route_waypoints
from wrappers import *

# TODO:
# - Some solution to avoid using the same env instance for training and eval
# - Just found out gym provides ObservationWrapper and RewardWrapper classes.
#   Should replace encode_state_fn and reward_fn with these.

class CarlaRouteEnv(gym.Env):
    """
        This is a simple CARLA environment where the goal is to drive in a lap
        around the outskirts of Town07. This environment can be used to compare
        different models/reward functions in a realtively predictable environment.

        To run an agent in this environment, either start start CARLA beforehand with:

        Synchronous:  $> ./CarlaUE4.sh Town07 -benchmark -fps=30
        Asynchronous: $> ./CarlaUE4.sh Town07

        Or, pass argument -start_carla in the command-line.
        Note that ${CARLA_ROOT} needs to be set to CARLA's top-level directory
        in order for this option to work.

        And also remember to set the -fps and -synchronous arguments to match the
        command-line arguments of the simulator (not needed with -start_carla.) 
        
        Note that you may also need to add the following line to
        Unreal/CarlaUE4/Config/DefaultGame.ini to have the map included in the package:
        
        +MapsToCook=(FilePath="/Game/Carla/Maps/Town07")
    """

    metadata = {
        "render.modes": ["human", "rgb_array", "rgb_array_no_hud", "state_pixels"]
    }

    def __init__(self, host="127.0.0.1", port=2000,
                 viewer_res=(1280, 720), obs_res=(1280, 720),
                 reward_fn=None, encode_state_fn=None,
                 synchronous=True, fps=30, action_smoothing=0.9,
                 start_carla=True):
        """
            Initializes a gym-like environment that can be used to interact with CARLA.

            Connects to a running CARLA enviromment (tested on version 0.9.5) and
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
            synchronous (bool):
                If True, run in synchronous mode (read the comment above for more info)
            start_carla (bool):
                Automatically start CALRA when True. Note that you need to
                set the environment variable ${CARLA_ROOT} to point to
                the CARLA root directory for this option to work.
        """

        # Start CARLA from CARLA_ROOT
        self.carla_process = None
        if start_carla:
            if "CARLA_ROOT" not in os.environ:
                raise Exception("${CARLA_ROOT} has not been set!")
            dist_dir = os.path.join(os.environ["CARLA_ROOT"], "Dist")
            if not os.path.isdir(dist_dir):
                raise Exception("Expected to find directory \"Dist\" under ${CARLA_ROOT}!")
            sub_dirs = [os.path.join(dist_dir, sub_dir) for sub_dir in os.listdir(dist_dir) if os.path.isdir(os.path.join(dist_dir, sub_dir))]
            if len(sub_dirs) == 0:
                raise Exception("Could not find a packaged distribution of CALRA! " +
                                "(try building CARLA with the \"make package\" " +
                                "command in ${CARLA_ROOT})")
            sub_dir = sub_dirs[0]
            carla_path = os.path.join(sub_dir, "LinuxNoEditor", "CarlaUE4.sh")
            launch_command = [carla_path]
            launch_command += ["Town07"]
            if synchronous: launch_command += ["-benchmark"]
            launch_command += ["-fps=%i" % fps]
            print("Running command:")
            print(" ".join(launch_command))
            self.carla_process = subprocess.Popen(launch_command, stdout=subprocess.DEVNULL)
            print("Waiting for CARLA to initialize")
            for line in self.carla_process.stdout:
                if "LogCarla: Number Of Vehicles" in line:
                    break
            time.sleep(2)

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
        self.synchronous = synchronous

        # Setup gym environment
        self.seed()
        self.action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32) # steer, throttle
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(*obs_res, 3), dtype=np.float32)
        self.metadata["video.frames_per_second"] = self.fps = self.average_fps = fps
        self.spawn_point = 1
        self.action_smoothing = action_smoothing
        self.encode_state_fn = (lambda x: x) if not callable(encode_state_fn) else encode_state_fn
        self.reward_fn = (lambda x: 0) if not callable(reward_fn) else reward_fn
        self.max_distance = 3000 # m

        self.world = None
        try:
            # Connect to carla
            self.client = carla.Client(host, port)
            self.client.set_timeout(60.0)

            # Create world wrapper
            self.world = World(self.client)

            if self.synchronous:
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                self.world.apply_settings(settings)

            # Create vehicle and attach camera to it
            self.vehicle = Vehicle(self.world, self.world.map.get_spawn_points()[0],
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
                                  sensor_tick=0.0 if self.synchronous else 1.0/self.fps)
            self.camera  = Camera(self.world, width, height,
                                  transform=camera_transforms["spectator"],
                                  attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e),
                                  sensor_tick=0.0 if self.synchronous else 1.0/self.fps)
        except Exception as e:
            self.close()
            raise e

        # Reset env to set initial state
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, is_training=False):
        # Create new route
        self.num_routes_completed = -1
        self.new_route()

        # Set env vars
        self.terminal_state = False # Set to True when we want to end episode
        self.closed = False         # Set to True when ESC is pressed
        self.extra_info = []        # List of extra info shown on the HUD
        self.observation = self.observation_buffer = None   # Last received observation
        self.viewer_image = self.viewer_image_buffer = None # Last received image to show in the viewer
        self.step_count = 0
        
        # Init metrics
        self.total_reward = 0.0
        self.previous_location = self.vehicle.get_transform().location
        self.distance_traveled = 0.0
        self.center_lane_deviation = 0.0
        self.speed_accum = 0.0
        self.routes_completed = 0.0

        # Return initial observation
        return self.step(None)[0]

    def new_route(self):
        # Do a soft reset (teleport vehicle)
        self.vehicle.control.steer = float(0.0)
        self.vehicle.control.throttle = float(0.0)
        #self.vehicle.control.brake = float(0.0)
        self.vehicle.tick()
        
        # Generate waypoints along the lap
        self.start_wp, self.end_wp = [self.world.map.get_waypoint(spawn.location) for spawn in np.random.choice(self.world.map.get_spawn_points(), 2, replace=False)]
        self.route_waypoints = compute_route_waypoints(self.world.map, self.start_wp, self.end_wp, resolution=1.0)
        self.current_waypoint_index = 0
        self.num_routes_completed += 1
        self.vehicle.set_transform(self.start_wp.transform)
        self.vehicle.set_simulate_physics(False) # Reset the car's physics
        self.vehicle.set_simulate_physics(True)

        # Give 2 seconds to reset
        if self.synchronous:
            ticks = 0
            while ticks < self.fps * 2:
                self.world.tick()
                try:
                    self.world.wait_for_tick(seconds=1.0/self.fps + 0.1)
                    ticks += 1
                except:
                    pass
        else:
            time.sleep(2.0)
        
    def close(self):
        if self.carla_process:
            self.carla_process.terminate()
        pygame.quit()
        if self.world is not None:
            self.world.destroy()
        self.closed = True

    def render(self, mode="human"):
        # Get maneuver name
        if self.current_road_maneuver == RoadOption.LANEFOLLOW: maneuver = "Follow Lane"
        elif self.current_road_maneuver == RoadOption.LEFT:     maneuver = "Left"
        elif self.current_road_maneuver == RoadOption.RIGHT:    maneuver = "Right"
        elif self.current_road_maneuver == RoadOption.STRAIGHT: maneuver = "Straight"
        elif self.current_road_maneuver == RoadOption.VOID:     maneuver = "VOID"
        else:                                                   maneuver = "INVALID(%i)" % self.current_road_maneuver

        # Add metrics to HUD
        self.extra_info.extend([
            "Reward: % 19.2f" % self.last_reward,
            "",
            "Maneuver:        % 11s"       % maneuver,
            "Routes completed:    % 7.2f"  % self.routes_completed,
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

        # Create new route on route completion
        if self.current_waypoint_index >= len(self.route_waypoints)-1:
            self.new_route()

        # Asynchronous update logic
        if not self.synchronous:
            if self.fps <= 0:
                # Go as fast as possible
                self.clock.tick()
            else:
                # Sleep to keep a steady fps
                self.clock.tick_busy_loop(self.fps)
            
            # Update average fps (for saving recordings)
            if action is not None:
                self.average_fps = self.average_fps * 0.5 + self.clock.get_fps() * 0.5

        # Take action
        if action is not None:
            steer, throttle = [float(a) for a in action]
            #steer, throttle, brake = [float(a) for a in action]
            self.vehicle.control.steer    = self.vehicle.control.steer * self.action_smoothing + steer * (1.0-self.action_smoothing)
            self.vehicle.control.throttle = self.vehicle.control.throttle * self.action_smoothing + throttle * (1.0-self.action_smoothing)
            #self.vehicle.control.brake = self.vehicle.control.brake * self.action_smoothing + brake * (1.0-self.action_smoothing)
        
        # Tick game
        self.hud.tick(self.world, self.clock)
        self.world.tick()

        # Synchronous update logic
        if self.synchronous:
            self.clock.tick()
            while True:
                try:
                    self.world.wait_for_tick(seconds=1.0/self.fps + 0.1)
                    break
                except:
                    # Timeouts happen occationally for some reason, however, they seem to be fine to ignore
                    self.world.tick()

        # Get most recent observation and viewer image
        self.observation = self._get_observation()
        self.viewer_image = self._get_viewer_image()
        encoded_state = self.encode_state_fn(self)

        # Get vehicle transform
        transform = self.vehicle.get_transform()

        # Keep track of closest waypoint on the route
        waypoint_index = self.current_waypoint_index
        for _ in range(len(self.route_waypoints)):
            # Check if we passed the next waypoint along the route
            next_waypoint_index = waypoint_index + 1
            wp, _ = self.route_waypoints[next_waypoint_index % len(self.route_waypoints)]
            dot = np.dot(vector(wp.transform.get_forward_vector())[:2],
                         vector(transform.location - wp.transform.location)[:2])
            if dot > 0.0: # Did we pass the waypoint?
                waypoint_index += 1 # Go to next waypoint
            else:
                break
        self.current_waypoint_index = waypoint_index

        # Check for route completion
        if self.current_waypoint_index < len(self.route_waypoints)-1:
            self.next_waypoint, self.next_road_maneuver = self.route_waypoints[(self.current_waypoint_index+1) % len(self.route_waypoints)]
        self.current_waypoint, self.current_road_maneuver = self.route_waypoints[self.current_waypoint_index % len(self.route_waypoints)]
        self.routes_completed = self.num_routes_completed + (self.current_waypoint_index + 1) / len(self.route_waypoints)

        # Calculate deviation from center of the lane
        self.distance_from_center = distance_to_line(vector(self.current_waypoint.transform.location),
                                                     vector(self.next_waypoint.transform.location),
                                                     vector(transform.location))
        self.center_lane_deviation += self.distance_from_center

        # Calculate distance traveled
        self.distance_traveled += self.previous_location.distance(transform.location)
        self.previous_location = transform.location

        # Accumulate speed
        self.speed_accum += self.vehicle.get_speed()

        # Terminal on max distance
        if self.distance_traveled >= self.max_distance:
            self.terminal_state = True
        
        # Call external reward fn
        self.last_reward = self.reward_fn(self)
        self.total_reward += self.last_reward
        self.step_count += 1

        # DEBUG: Draw path
        #self._draw_path(life_time=5.0, skip=10)

        # DEBUG: Draw current waypoint
        #self.world.debug.draw_point(self.current_waypoint.transform.location, color=carla.Color(0, 255, 0), life_time=5.0)

        # Check for ESC press
        pygame.event.pump()
        if pygame.key.get_pressed()[K_ESCAPE]:
            self.close()
            self.terminal_state = True
        
        return encoded_state, self.last_reward, self.terminal_state, { "closed": self.closed }

    def _draw_path(self, life_time=60.0, skip=0):
        """
            Draw a connected path from start of route to end.
            Green node = start
            Red node   = point along path
            Blue node  = destination
        """
        for i in range(0, len(self.route_waypoints)-1, skip+1):
            w0 = self.route_waypoints[i][0]
            w1 = self.route_waypoints[i+1][0]
            self.world.debug.draw_line(
                w0.transform.location + carla.Location(z=0.25),
                w1.transform.location + carla.Location(z=0.25),
                thickness=0.1, color=carla.Color(255, 0, 0),
                life_time=life_time, persistent_lines=False)
            self.world.debug.draw_point(
                w0.transform.location + carla.Location(z=0.25), 0.1,
                carla.Color(0, 255, 0) if i == 0 else carla.Color(255, 0, 0),
                life_time, False)
        self.world.debug.draw_point(
            self.route_waypoints[-1][0].transform.location + carla.Location(z=0.25), 0.1,
            carla.Color(0, 0, 255),
            life_time, False)

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
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        self.hud.notification("Crossed line %s" % " and ".join(text))

    def _set_observation_image(self, image):
        self.observation_buffer = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

def reward_fn(env):
    early_termination = False
    if early_termination:
        # If speed is less than 1.0 km/h after 5s, stop
        if time.time() - env.start_t > 5.0 and env.vehicle.get_speed() < 1.0 / 3.6:
            env.terminal_state = True

        # If distance from center > 3, stop
        if env.distance_from_center > 3.0:
            env.terminal_state = True
        
    fwd    = vector(env.vehicle.get_velocity())
    wp_fwd = vector(env.current_waypoint.transform.rotation.get_forward_vector())
    if np.dot(fwd[:2], wp_fwd[:2]) > 0:
        return env.vehicle.get_speed()
    return 0

if __name__ == "__main__":
    # Example of using CarlaEnv with keyboard controls
    env = CarlaRouteEnv(obs_res=(160, 80), reward_fn=reward_fn)
    action = np.zeros(env.action_space.shape[0])
    while True:
        env.reset()
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
