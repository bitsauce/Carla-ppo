import os
import shutil
import subprocess

import carla
import gym
import pygame
from PIL import Image
from pygame.locals import *

from hud import HUD
from wrappers import *


class CarlaDataCollector:
    """
        To be able to drive in this environment, either start start CARLA beforehand with:

        Synchronous:  $> ./CarlaUE4.sh Town07 -benchmark -fps=30
        Asynchronous: $> ./CarlaUE4.sh Town07

        Or pass argument -start_carla in the command-line.
        Note that ${CARLA_ROOT} needs to be set to CARLA's top-level directory
        in order for this option to work.
    """

    def __init__(self, host="127.0.0.1", port=2000, 
                 viewer_res=(1280, 720), obs_res=(1280, 720),
                 num_images_to_save=10000, output_dir="images",
                 synchronous=True, fps=30, action_smoothing=0.9,
                 start_carla=True):
        """
            Initializes an environment that can be used to save camera/sensor data
            from driving around manually in CARLA.

            Connects to a running CARLA enviromment (tested on version 0.9.5) and
            spwans a lincoln mkz2017 passenger car with automatic transmission.

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
            num_images_to_save (int):
                Number of images to collect
            output_dir (str):
                Output directory to save the images to
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
            self.carla_process = subprocess.Popen(launch_command, stdout=subprocess.PIPE, universal_newlines=True)
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

        # Setup gym environment
        self.action_space = gym.spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32) # steer, throttle
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(*obs_res, 3), dtype=np.float32)
        self.fps = fps
        self.spawn_point = 1
        self.action_smoothing = action_smoothing

        self.done = False
        self.recording = False
        self.extra_info = []
        self.num_saved_observations = 0
        self.num_images_to_save = num_images_to_save
        self.observation = {key: None for key in ["rgb", "segmentation"]}        # Last received observations
        self.observation_buffer = {key: None for key in ["rgb", "segmentation"]}
        self.viewer_image = self.viewer_image_buffer = None                   # Last received image to show in the viewer

        self.output_dir = output_dir
        os.makedirs(os.path.join(self.output_dir, "rgb"))
        os.makedirs(os.path.join(self.output_dir, "segmentation"))

        self.world = None
        try:
            # Connect to carla
            self.client = carla.Client(host, port)
            self.client.set_timeout(2.0)

            # Create world wrapper
            self.world = World(self.client)

            # Example: Synchronizing a camera with synchronous mode.
            settings = self.world.get_settings()
            settings.synchronous_mode = True
            self.world.apply_settings(settings)

            # Get spawn location
            lap_start_wp = self.world.map.get_waypoint(carla.Location(x=-180.0, y=110))
            spawn_transform = lap_start_wp.transform
            spawn_transform.location += carla.Location(z=1.0)

            # Create vehicle and attach camera to it
            self.vehicle = Vehicle(self.world, spawn_transform,
                                   on_collision_fn=lambda e: self._on_collision(e),
                                   on_invasion_fn=lambda e: self._on_invasion(e))

            # Create hud
            self.hud = HUD(width, height)
            self.hud.set_vehicle(self.vehicle)
            self.world.on_tick(self.hud.on_world_tick)

            # Create cameras
            self.dashcam_rgb = Camera(self.world, out_width, out_height,
                                      transform=camera_transforms["dashboard"],
                                      attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_image("rgb", e))
            self.dashcam_seg = Camera(self.world, out_width, out_height,
                                      transform=camera_transforms["dashboard"],
                                      attach_to=self.vehicle, on_recv_image=lambda e: self._set_observation_image("segmentation", e),
                                      camera_type="sensor.camera.semantic_segmentation")#, color_converter=carla.ColorConverter.CityScapesPalette)
            self.camera  = Camera(self.world, width, height,
                                  transform=camera_transforms["spectator"],
                                  attach_to=self.vehicle, on_recv_image=lambda e: self._set_viewer_image(e))
        except Exception as e:
            self.close()
            raise e

        self.hud.notification("Press \"Enter\" to start collecting data.")

    def close(self):
        if self.carla_process:
            self.carla_process.terminate()
        pygame.quit()
        if self.world is not None:
            self.world.destroy()
        self.closed = True

    def save_observation(self):
        # Blit image from spectator camera
        self.display.blit(pygame.surfarray.make_surface(self.viewer_image.swapaxes(0, 1)), (0, 0))

        # Superimpose current observation into top-right corner
        for i, (_, obs) in enumerate(self.observation.items()):
            obs_h, obs_w = obs.shape[:2]
            view_h, view_w = self.viewer_image.shape[:2]
            pos = (view_w - obs_w - 10, obs_h * i + 10 * (i+1))
            self.display.blit(pygame.surfarray.make_surface(obs.swapaxes(0, 1)), pos)

        # Save current observations
        if self.recording:
            for obs_type, obs in self.observation.items():
                img = Image.fromarray(obs)
                img.save(os.path.join(self.output_dir, obs_type, "{}.png".format(self.num_saved_observations)))
            self.num_saved_observations += 1
            if self.num_saved_observations >= self.num_images_to_save:
                self.done = True

        # Render HUD
        self.extra_info.extend([
            "Images: %i/%i" % (self.num_saved_observations, self.num_images_to_save),
            "Progress: %.2f%%" % (self.num_saved_observations / self.num_images_to_save * 100.0)
        ])
        self.hud.render(self.display, extra_info=self.extra_info)
        self.extra_info = [] # Reset extra info list

        # Render to screen
        pygame.display.flip()

    def step(self, action):
        if self.is_done():
            raise Exception("Step called after CarlaDataCollector was done.")

        # Take action
        if action is not None:
            steer, throttle = [float(a) for a in action]
            #steer, throttle, brake = [float(a) for a in action]
            self.vehicle.control.steer    = self.vehicle.control.steer * self.action_smoothing + steer * (1.0-self.action_smoothing)
            self.vehicle.control.throttle = self.vehicle.control.throttle * self.action_smoothing + throttle * (1.0-self.action_smoothing)
            #self.vehicle.control.brake = self.vehicle.control.brake * self.action_smoothing + brake * (1.0-self.action_smoothing)
        
        # Tick game
        self.clock.tick()
        self.hud.tick(self.world, self.clock)
        self.world.tick()
        try:
            self.world.wait_for_tick(seconds=0.5)
        except RuntimeError as e:
            pass # Timeouts happen for some reason, however, they are fine to ignore

        # Get most recent observation and viewer image
        self.observation["rgb"] = self._get_observation("rgb")
        self.observation["segmentation"] = self._get_observation("segmentation")
        self.viewer_image = self._get_viewer_image()

        pygame.event.pump()
        keys = pygame.key.get_pressed()
        if keys[K_ESCAPE]:
            self.done = True
        if keys[K_SPACE]:
            self.recording = True

    def is_done(self):
        return self.done

    def _get_observation(self, name):
        while self.observation_buffer[name] is None:
            pass
        obs = self.observation_buffer[name].copy()
        self.observation_buffer[name] = None
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

    def _set_observation_image(self, name, image):
        self.observation_buffer[name] = image

    def _set_viewer_image(self, image):
        self.viewer_image_buffer = image

if __name__ == "__main__":
    import argparse
    argparser = argparse.ArgumentParser(description="Run this script to drive around with WASD/arrow keys. " +
                                                    "Press SPACE to start recording RGB and semanting segmentation images from the front facing camera to the disk")
    argparser.add_argument("--host", default="127.0.0.1", type=str, help="IP of the host server (default: 127.0.0.1)")
    argparser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")
    argparser.add_argument("--viewer_res", default="1280x720", type=str, help="Window resolution (default: 1280x720)")
    argparser.add_argument("--obs_res", default="160x80", type=str, help="Output resolution (default: same as --res)")
    argparser.add_argument("--output_dir", default="images", type=str, help="Directory to save images to")
    argparser.add_argument("--num_images", default=10000, type=int, help="Number of images to collect")
    argparser.add_argument("--fps", default=30, type=int, help="FPS. Delta time between samples is 1/FPS")
    argparser.add_argument("--synchronous", type=int, default=True, help="Set this to True when running in a synchronous environment")
    argparser.add_argument("-start_carla", action="store_true", help="Automatically start CALRA with the given environment settings")
    args = argparser.parse_args()

    # Remove existing output directory
    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    # Parse viewer_res and obs_res
    viewer_res = [int(x) for x in args.viewer_res.split("x")]
    if args.obs_res is None:
        obs_res = viewer_res
    else:
        obs_res = [int(x) for x in args.obs_res.split("x")]

    # Create vehicle and actors for data collecting
    data_collector = CarlaDataCollector(host=args.host, port=args.port,
                                        viewer_res=viewer_res, obs_res=obs_res, fps=args.fps,
                                        num_images_to_save=args.num_images, output_dir=args.output_dir,
                                        synchronous=args.synchronous, start_carla=args.start_carla)
    action = np.zeros(data_collector.action_space.shape[0])

    # While there are more images to collect
    while not data_collector.is_done():
        # Process keyboard input
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
        data_collector.step(action)
        data_collector.save_observation()
        
    # Destroy carla actors
    data_collector.close()
