"""
Welcome to CARLA manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down

    TAB          : change sensor position
    `            : next sensor
    [1-9]        : change to sensor [1-9]
    C            : change weather (Shift+C reverse)
    Backspace    : change vehicle

    R            : toggle recording images to disk

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

import pygame
import datetime
import math
from utils import get_actor_display_name

#===============================================================================
# HUD
#===============================================================================

class HUD(object):
    """
        HUD class for displaying on-screen information
    """

    def __init__(self, width, height):
        self.dim = (width, height)

        # Select a monospace font for the info panel
        fonts = [x for x in pygame.font.get_fonts() if "mono" in x]
        default_font = "ubuntumono"
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self.font_mono = pygame.font.Font(mono, 14)

        # Use default font for everything else
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        
        self.notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self.show_info = True
        self.info_text = []
        self.server_clock = pygame.time.Clock()

    def tick(self, world, clock):
        if self.show_info:
            # Get transform, velocity and heading
            t = world.vehicle.get_transform()
            v = world.vehicle.get_velocity()
            c = world.vehicle.get_control()
            heading = "N" if abs(t.rotation.yaw) < 89.5 else ""
            heading += "S" if abs(t.rotation.yaw) > 90.5 else ""
            heading += "E" if 179.5 > t.rotation.yaw > 0.5 else ""
            heading += "W" if -0.5 > t.rotation.yaw > -179.5 else ""

            # Get collision history for the last 200 frames
            colhist = world.collision_sensor.get_collision_history()
            collision = [colhist[x + self.frame_number - 200] for x in range(0, 200)]
            max_col = max(1.0, max(collision)) # 1 if max(collision) < 1 else max(collision)
            collision = [x / max_col for x in collision]

            # Get all world vehicles
            vehicles = world.get_actors().filter("vehicle.*")

            # Create info text
            self.info_text = [
                "Images:  % 16i    " % world.recording_camera.num_images,
                "Server:  % 16d FPS" % self.server_fps,
                "Client:  % 16d FPS" % clock.get_fps(),
                "",
                "Vehicle: % 20s" % get_actor_display_name(world.vehicle, truncate=20),
                "Map:     % 20s" % world.map.name,
                "Simulation time: % 12s" % datetime.timedelta(seconds=int(self.simulation_time)),
                "",
                "Speed:   % 15.0f km/h" % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
                u"Heading:% 16.0f\N{DEGREE SIGN} % 2s" % (t.rotation.yaw, heading),
                "Location:% 20s" % ("(% 5.1f, % 5.1f)" % (t.location.x, t.location.y)),
                "Height:  % 18.0f m" % t.location.z,
                "",
                ("Throttle:", c.throttle, 0.0, 1.0),
                ("Steer:", c.steer, -1.0, 1.0),
                ("Brake:", c.brake, 0.0, 1.0),
                ("Reverse:", c.reverse),
                ("Hand brake:", c.hand_brake),
                ("Manual:", c.manual_gear_shift),
                "Gear:        %s" % {-1: "R", 0: "N"}.get(c.gear, c.gear),
                "",
                "Collision:",
                collision,
                "",
                "Number of vehicles: % 8d" % len(vehicles)
            ]

            # Append info about nearby vehicles
            if len(vehicles) > 1:
                self.info_text += ["Nearby vehicles:"]
                distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
                vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.vehicle.id]
                for d, vehicle in sorted(vehicles):
                    if d > 200.0:
                        break
                    vehicle_type = get_actor_display_name(vehicle, truncate=22)
                    self.info_text.append("% 4dm %s" % (d, vehicle_type))

        # Tick notifications
        self.notifications.tick(world, clock)

    def render(self, display):
        if self.show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self.info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item: # At this point has to be a str.
                    surface = self.font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self.notifications.render(display)
        self.help.render(display)

    def on_world_tick(self, timestamp):
        # Store info when server ticks
        self.server_clock.tick()
        self.server_fps = self.server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def toggle_info(self):
        self.show_info = not self.show_info

    def notification(self, text, seconds=2.0):
        self.notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self.notifications.set_text("Error: %s" % text, (255, 0, 0))


#===============================================================================
# FadingText
#===============================================================================

class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


#===============================================================================
# HelpText
#===============================================================================

class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split("\n")
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)
