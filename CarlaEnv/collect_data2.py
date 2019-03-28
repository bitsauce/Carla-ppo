import pygame
import carla
import os, shutil
from wrappers import *
from keyboard_control import KeyboardControl
from hud_v2 import HUD

"""
import os
from threading import Thread, Lock
mutex = Lock()

class Camera(CarlaActorBase):
    def __init__(self, world, width, height, transform=carla.Transform(), sensor_tick=0.0, on_recv_image=None,
                 attach_to=None, output_images_path="images", output_control=False):
        self.surface = pygame.surfarray.make_surface(np.zeros((width, height, 3)))
        self.recording = False
        self.num_images = 0
        self.image = None
        self.output_images_path = output_images_path
        self.output_control = output_control
        if self.output_control:
            self.controls_outfile = open("%s/controls.csv" % self.output_images_path, "w")
        self.on_recv_image = on_recv_image

        # Setup camera blueprint
        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(width))
        camera_bp.set_attribute("image_size_y", str(height))
        camera_bp.set_attribute("sensor_tick", str(sensor_tick))

        # Create and setup camera actor
        weak_self = weakref.ref(self)
        actor = world.spawn_actor(camera_bp, transform, attach_to=attach_to.get_carla_actor())
        actor.listen(lambda image: Camera.process_camera_input(weak_self, image))
        print("Spawned actor \"{}\"".format(actor.type_id))

        super().__init__(world, actor)
    
    @staticmethod
    def process_camera_input(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.image = array
        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        if callable(self.on_recv_image):
            self.on_recv_image(array)

        if self.recording:
            mutex.acquire()
            try:
                # TODO: It seems to be writing the same image multiple times. FIX!
                path = "%s/%08d" % (self.output_images_path, self.num_images)
                #print("Saving",path,"...")
                if os.path.isfile(path):
                    raise Exception("File already created")
                image.save_to_disk(path)
                #print("Saved",path)
                c = self.world.vehicle.get_control()
                if self.output_control:
                    self.controls_outfile.write("%f, %f, %f\n" % (c.steer, c.throttle, c.brake))
                self.num_images += 1
            finally:
                mutex.release()

    def destroy(self):
        if self.output_control:
            self.controls_outfile.close()
        super().destroy()
"""

def main():
    import argparse
    argparser = argparse.ArgumentParser(description="Script for driving around to collecting data")
    argparser.add_argument("--host", default="127.0.0.1", type=str, help="IP of the host server (default: 127.0.0.1)")
    argparser.add_argument("--port", default=2000, type=int, help="TCP port to listen to (default: 2000)")
    argparser.add_argument("--autopilot", action="store_true", help="Enable autopilot")
    argparser.add_argument("--res", default="1280x720", type=str, help="Window resolution (default: 1280x720)")
    argparser.add_argument("--output_res", default=None, type=str, help="Output resolution (default: same as --res)")
    argparser.add_argument("--output_dir", default="images", type=str, help="Output directory for saved images")
    argparser.add_argument("--n_steps", default=10000, type=int, help="Number of images to collect")
    args = argparser.parse_args()
    
    if os.path.isdir(args.output_dir):
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir)

    # Initialize pygame
    pygame.init()
    pygame.font.init()
    width, height = [int(x) for x in args.res.split("x")]
    if args.output_res is None:
        out_width, out_height = width, height
    else:
        out_width, out_height = [int(x) for x in args.output_res.split("x")]
    display = pygame.display.set_mode((width, height), pygame.HWSURFACE | pygame.DOUBLEBUF)

    world = None
    try:
        # Connect to carla
        client = carla.Client("localhost", 2000)
        client.set_timeout(2.0)

        # Create hud
        hud = HUD(width, height)

        # Create world wrapper
        world = World(client, width, height)

        # Set up hud
        world.on_tick(hud.on_world_tick)

        soft_reset = True
        init = True
        clock = pygame.time.Clock()

        while True:
            # Flag to reset the world
            reset = False

            if init:
                def do_reset():
                    nonlocal reset
                    reset = True

                def on_collision(event):
                    # Display notification
                    hud.notification("Collision with {}".format(get_actor_display_name(event.other_actor)))
                    do_reset()

                def on_invasion(event):
                    # Display notification
                    text = ["%r" % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
                    hud.notification("Crossed line %s" % " and ".join(text))
                    do_reset()
                    
                # Create vehicle and attach camera to it
                vehicle = Vehicle(world, world.map.get_spawn_points()[0],
                                on_collision_fn=on_collision, on_invasion_fn=on_invasion)
                hud.set_vehicle(vehicle)

                # Create cameras
                dashcam = Camera(world, out_width, out_height,
                                transform=carla.Transform(carla.Location(x=1.6, z=1.7)),
                                attach_to=vehicle)
                camera = Camera(world, width, height,
                                transform=carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                                attach_to=vehicle)
                controller = KeyboardControl(world, vehicle, hud)
                
                init = False

            while not reset:
                clock.tick_busy_loop(60)

                if controller.parse_events(client, clock):
                    return

                # Tick
                world.tick()
                hud.tick(world, clock)

                # Render
                display.blit(camera.surface, (0, 0))
                hud.render(display)

                pygame.display.flip()

            if soft_reset:
                vehicle.set_transform(world.map.get_spawn_points()[0])
                vehicle.set_simulate_physics(False)
                vehicle.set_simulate_physics(True)
            else:    
                # Destroy all actors
                world.destroy()
                init = True
    finally:
        pygame.quit()
        if world is not None:
            world.destroy()

if __name__ == "__main__":
    main()