import pygame
import carla
import gym
import time
import random
from gym.utils import seeding
from wrappers import *
from keyboard_control import KeyboardControl
from hud_v2 import HUD

class CarlaEnv(gym.Env):
    metadata = {
        "render.modes": ["human", "rgb_array", "state_pixels"],
        "video.frames_per_second" : 30
    }

    def __init__(self, host="127.0.0.1", port=2000, viewer_res=(1280, 720), obs_res=(1280, 720),
                 reward_fn=None, encode_state_fn=None, spawn_point=1):
        # Initialize pygame
        pygame.init()
        pygame.font.init()
        width, height = viewer_res
        if obs_res is None:
            out_width, out_height = width, height
        else:
            out_width, out_height = obs_res
        self.display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.viewer_image = None

        self.seed()
        self.action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32)  # steer, gas
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(*obs_res, 3), dtype=np.float32)
        self.spawn_point = spawn_point

        self.terminal_state = False
        self.extra_info = []
        self.closed = False
        self.current_obs = None
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
                                  transform=carla.Transform(carla.Location(x=1.6, z=1.7)),
                                  attach_to=self.vehicle,
                                  sensor_tick=1/30.0, on_recv_image=lambda e: self._set_observation_image(e))
            self.camera = Camera(self.world, width, height,
                                 transform=carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                                 attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e))
            self.controller = KeyboardControl(self.world, self.vehicle, self.hud)
            self.controller.set_enabled(False)

            self.clock = pygame.time.Clock()
            
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

        self.terminal_state = False
        self.extra_info = []
        self.closed = False
        self.current_obs = None
        self.start_t = time.time()

        return self.step(None)[0]

    def close(self):
        pygame.quit()
        if self.world is not None:
            self.world.destroy()
        self.closed = True

    def render(self, mode="human"):
        # Render
        image = self.viewer_image.copy()
        surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
        self.display.blit(surface, (0, 0))
        self.hud.render(self.display, extra_info=self.extra_info)
        pygame.display.flip()
        self.extra_info = []

        if mode == "rgb_array":
            return image
        elif mode == "state_pixels":
            return self.current_obs

    def step(self, action):
        if self.closed:
            raise Exception("CarlaEnv.step() called after the environment was closed."+
                            "Check for info[\"closed\"] == True in the learning loop.")

        # Tick clock
        self.clock.tick()

        if action is not None:
            # Set control signal
            self.vehicle.control.steer = float(action[0])
            self.vehicle.control.throttle = float(action[1])

        # Get most recent observation
        self.observation = self._get_observation()
        encoded_state = self.encode_state_fn(self)

        #self.terminal_state = self.is_terminal_fn(self)

        # Tick
        self.world.tick()
        self.hud.tick(self.world, self.clock)


        """velocity = self.vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2)
        if time.time() - self.start_t > 5.0 and speed < 1.0:
            self.terminal_state = True"""


        # TODO:
        reward = self.reward_fn(self)

        # Check for ESC
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
        while self.camera.image is None:
            pass
        obs = self.camera.image.copy()
        self.camera.image = None
        return obs

    def _on_collision(self, event):
        # Display notification
        self.hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))
        #self.terminal_state = True

    def _on_invasion(self, event):
        # Display notification
        text = ["%r" % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
        self.hud.notification("Crossed line %s" % " and ".join(text))
        #self.terminal_state = True

    def _set_observation_image(self, image):
        self.current_obs = image

    def _set_viewer_image(self, image):
        self.viewer_image = image

if __name__ == "__main__":
    def reward_fn(env):
        terminal_state = env.terminal_state

        # If speed is less than 1 after 5s, stop
        speed = env.vehicle.get_speed()
        if time.time() - env.start_t > 5.0 and speed < 1.0:
            terminal_state = True

        # If heading is oposite, stop
        transform = env.vehicle.get_transform()
        waypoint = env.world.map.get_waypoint(transform.location, project_to_road=True) # Get closest waypoint
        #world.debug.draw_point(wp_loc, life_time=1.0)
        loc, wp_loc = carla_as_array(waypoint.transform.location), carla_as_array(transform.location)
        distance_from_center = np.linalg.norm(loc[:2] - wp_loc[:2])

        fwd = transform.rotation.get_forward_vector()
        wp_fwd = waypoint.transform.rotation.get_forward_vector()
        angle = angle_diff(carla_as_array(fwd), carla_as_array(wp_fwd))

        if angle > np.pi/2 or angle < -np.pi/2 or distance_from_center > 3.0:
            terminal_state = True

        reward = 0
        if terminal_state == True:
            env.terminal_state = True
            reward -= 10
        else:
            #if 3.6 * speed < 20.0: # No reward over 20 kmh
            #    reward += env.vehicle.control.throttle
            norm_speed = 3.6 * speed / 20.0
            if norm_speed > 1.0:
                reward += (1.0 - norm_speed) * 3
            else:
                reward += norm_speed * 3
            reward -= distance_from_center

        env.extra_info = [
            "Distance from center: %.2f" % distance_from_center,
            "Angle difference: %.2f" % np.rad2deg(angle),
            "Wrong way" if (np.rad2deg(angle) > 90 or np.rad2deg(angle) < -90) else "Right way",
            "Reward: %.4f" % reward
        ]
        return reward

    from pygame.locals import *
    env = CarlaEnv(obs_res=(160, 80), reward_fn=reward_fn, spawn_point=10)
    action = np.zeros(env.action_space.shape[0])
    while True:
        env.reset()
        restart = False
        while True:
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

            obs, _, done, info = env.step(action)

            if info["closed"]:
                exit(0)
            env.render()
            if done or restart: break
    env.close()
