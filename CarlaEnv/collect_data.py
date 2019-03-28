import pygame
import carla
import os, shutil
from wrappers import *
from keyboard_control import KeyboardControl
from hud_v2 import HUD

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
    surface = pygame.surfarray.make_surface(np.zeros((width, height, 3)))

    world = None
    try:
        # Connect to carla
        client = carla.Client("localhost", 2000)
        client.set_timeout(2.0)

        # Create hud
        hud = HUD(width, height)

        # Create world wrapper
        world = World(client)

        # Set up hud
        world.on_tick(hud.on_world_tick)
        
        clock = pygame.time.Clock()

        while True:
            # Flag to reset the world
            reset = False
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
                
            def update_surface(image):
                nonlocal surface
                surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))

            # Create vehicle and attach camera to it
            idx = np.random.randint(len(world.map.get_spawn_points()))
            print("Spawn",idx)
            vehicle = Vehicle(world, world.map.get_spawn_points()[idx],
                            on_collision_fn=on_collision, on_invasion_fn=on_invasion)
            hud.set_vehicle(vehicle)

            # Create cameras
            dashcam = Camera(world, out_width, out_height,
                            transform=carla.Transform(carla.Location(x=1.6, z=1.7)),
                            attach_to=vehicle)
            camera = Camera(world, width, height,
                            transform=carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                            attach_to=vehicle, on_recv_image=update_surface)
            controller = KeyboardControl(world, vehicle, hud)

            while True:#recording_camera.num_images < args.n_steps:
                clock.tick_busy_loop(60)

                if controller.parse_events(client, clock):
                    return

                transform = vehicle.get_transform()
                waypoint = world.map.get_waypoint(transform.location, project_to_road=True) # Get closest waypoint
                #world.debug.draw_point(wp_loc, life_time=1.0)
                loc, wp_loc, = carla_as_array(waypoint.transform.location), carla_as_array(transform.location)
                distance_from_center = np.linalg.norm(loc[:2] - wp_loc[:2])

                fwd = transform.rotation.get_forward_vector()
                wp_fwd = waypoint.transform.rotation.get_forward_vector()
                angle = angle_diff(carla_as_array(fwd), carla_as_array(wp_fwd))

                extra_info = ["Distance from center: %.2f" % distance_from_center,
                              "Angle difference: %.2f" % np.rad2deg(angle),
                              "Wrong way" if (np.rad2deg(angle) > 90 or np.rad2deg(angle) < -90) else "Right way"]
                # Tick
                world.tick()
                hud.tick(world, clock)

                # Render
                display.blit(surface, (0, 0))
                hud.render(display, extra_info=extra_info)

                pygame.display.flip()
                
            # Destroy all actors
            world.destroy()
            break
    finally:
        pygame.quit()
        if world is not None:
            world.destroy()

if __name__ == "__main__":
    main()