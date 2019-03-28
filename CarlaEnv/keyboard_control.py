
import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

index = 0

class KeyboardControl(object):
    def __init__(self, world, vehicle, hud):
        self._steer_cache = 0.0
        self.world = world
        self.vehicle = vehicle
        self.hud = hud

        self.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

        self.enabled = True

    def set_enabled(self, enabled):
        self.enabled = enabled

    def parse_events(self, client, clock):
        world = self.world
        hud = self.hud
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    hud.help.toggle()
                elif event.key == K_TAB:
                    #world.camera_manager.toggle_camera()

                    transforms = [carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
                                  carla.Transform(carla.Location(x=1.6, z=1.7))]
                    
                    global index
                    index = (index + 1) % 2
                    world.camera.set_transform(transforms[index])

                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    #world.camera_manager.toggle_recording()
                    world.recording_camera.recording = not world.recording_camera.recording
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    currentIndex = world.camera_manager.index
                    world.destroy_sensors()
                    hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(currentIndex)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    hud.notification("Recording start time is %d" % (world.recording_start))
                if isinstance(self.vehicle.control, carla.VehicleControl):
                    if event.key == K_q:
                        self.vehicle.control.gear = 1 if self.vehicle.control.reverse else -1
                    elif event.key == K_m:
                        self.vehicle.control.manual_gear_shift = not self.vehicle.control.manual_gear_shift
                        self.vehicle.control.gear = self.vehicle.get_control().gear
                        hud.notification('%s Transmission' %
                                               ('Manual' if self.vehicle.control.manual_gear_shift else 'Automatic'))
                    elif self.vehicle.control.manual_gear_shift and event.key == K_COMMA:
                        self.vehicle.control.gear = max(-1, self.vehicle.control.gear - 1)
                    elif self.vehicle.control.manual_gear_shift and event.key == K_PERIOD:
                        self.vehicle.control.gear = self.vehicle.control.gear + 1
        
        if self.enabled:
            if isinstance(self.vehicle.control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self.vehicle.control.reverse = self.vehicle.control.gear < 0

    def _parse_vehicle_keys(self, keys, milliseconds):
        world = self.world
        self.vehicle.control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self.vehicle.control.steer = round(self._steer_cache, 1)
        self.vehicle.control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self.vehicle.control.hand_brake = keys[K_SPACE]

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)
