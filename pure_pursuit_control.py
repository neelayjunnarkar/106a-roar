import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import numpy as np
import math

"""
Citation: https://github.com/AtsushiSakai/PythonRobotics/blob/master
/PathTracking/pure_pursuit/pure_pursuit.py
"""

WHEEL_BASE_LENGTH = 3

class PurePursuitController:
    def __init__(
            self,
            vehicle: carla.Vehicle,
            look_ahead_gain: float = 0.1,
            look_ahead_distance: float = 2,
            target_speed=16,
    ):
        """

        Args:
            vehicle: Vehicle information
            look_ahead_gain: Look ahead factor
            look_ahead_distance: look ahead distance
            target_speed: desired longitudinal speed to maintain
        """
        self.vehicle = vehicle
        self.target_speed = target_speed
        self.look_ahead_gain = look_ahead_gain
        self.look_ahead_distance = look_ahead_distance
        self.latitunal_controller = LatitunalPurePursuitController(
            vehicle=self.vehicle,
            look_ahead_gain=look_ahead_gain,
            look_ahead_distance=look_ahead_distance,
        )
        self.longitunal_controller = LongitunalPurePursuitController(
            vehicle=self.vehicle, target_speed=target_speed
        )

    def run_step(
            self, vehicle: carla.Vehicle, next_waypoint: carla.Waypoint, **kwargs
    ) -> carla.VehicleControl:
        """
        run one step of Pure Pursuit Control

        Args:
            vehicle: current vehicle state
            next_waypoint: Next waypoint, Waypoint
            **kwargs:

        Returns:
            Vehicle Control

        """
        control = carla.VehicleControl(
            throttle=self.longitunal_controller.run_step(vehicle=vehicle),
            steer=self.latitunal_controller.run_step(
                vehicle=vehicle, next_waypoint=next_waypoint
            ),
        )
        return control


class LongitunalPurePursuitController:
    # target_speed in m/s
    def __init__(self, vehicle: carla.Vehicle, target_speed=16, kp=0.1):
        self.vehicle = vehicle
        self.target_speed = target_speed
        self.kp = kp

    def run_step(self, vehicle: carla.Vehicle) -> float:
        self.sync(vehicle)
        velocity = vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return np.clip(self.kp * (self.target_speed - speed), 0, 1)

    def sync(self, vehicle: carla.Vehicle):
        self.vehicle = vehicle


class LatitunalPurePursuitController:
    def __init__(
            self, vehicle: carla.Vehicle, look_ahead_gain: float,
            look_ahead_distance: float
    ):
        self.vehicle = vehicle
        self.look_ahead_gain = look_ahead_gain
        self.look_ahead_distance = look_ahead_distance

    def run_step(self, vehicle: carla.Vehicle, next_waypoint: carla.Waypoint) -> float:
        self.sync(vehicle=vehicle)
        target_y = next_waypoint.transform.location.y
        target_x = next_waypoint.transform.location.x
        angle_difference = math.atan2(
            target_y - self.vehicle.get_location().y,
            target_x - self.vehicle.get_location().x,
        ) - np.radians(self.vehicle.get_transform().rotation.yaw)
        velocity = vehicle.get_velocity()
        speed = np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        curr_look_forward = (
                self.look_ahead_gain * speed
                + self.look_ahead_distance
        )
        lateral_difference = math.atan2(
            2.0
            * WHEEL_BASE_LENGTH
            * math.sin(angle_difference)
            / curr_look_forward,
            1.0,
        )
        return np.clip(lateral_difference, -1, 1)

    def sync(self, vehicle: carla.Vehicle):
        self.vehicle = vehicle
