""" Test program to follow a sequence of closely-spaced waypoints """

# Author: Neelay Junnarkar
# Date: 2020-11

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

import argparse
import logging
import numpy as np

WAYPOINT_ERROR_THRESHOLD = 3

class Waypoint:
    def __init__(self, position, rotation):
        """
        position: carla.Location
        rotation: float
        """
        self.position = position
        self.rotation = rotation
    
    def error(self, position, rotation):
        """ L2 Error between (position, rotation) and this waypoint"""
        return np.sqrt(sum([
            (self.position.x - position.x)**2,
            (self.position.y - position.y)**2,
            (self.position.z - position.z)**2,
            (self.rotation - rotation)**2
        ]))

def setup_waypoints(position_init, rotation_init):
    waypoints = []
    waypoints.append(Waypoint(
        position_init,
        rotation_init.yaw
    ))
    waypoints.append(Waypoint(
        position_init + 20 * rotation_init.get_forward_vector()
        + -5 * rotation_init.get_right_vector(),
        rotation_init.yaw - 15
    ))
    waypoints.append(Waypoint(
        position_init + 30 * rotation_init.get_forward_vector()
        + -10 * rotation_init.get_right_vector(),
        rotation_init.yaw - 45
    ))
    return waypoints

def distance(a, b):
    return np.sqrt((a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2)

def dot(a, b):
    return a.x * b.x + a.y * b.y + a.z * b.z

def generate_control(client, world, vehicle, waypoint, prev_waypoint):

    forward_dir = vehicle.get_transform().rotation.get_forward_vector()
    p_vehicle = vehicle.get_transform().location
    pd_vehicle = vehicle.get_velocity()
    pdd_vehicle = vehicle.get_acceleration()
    R_vehicle = vehicle.get_transform().rotation.yaw
    Rd_vehicle = distance(vehicle.get_angular_velocity(), carla.Vector3D(0, 0, 0))
    clockwise = False if vehicle.get_angular_velocity().z > 0 else True
    if clockwise:
        Rd_vehicle = -Rd_vehicle

    if generate_control.prev_yaw is None:
        generate_control.prev_yaw = R_vehicle

    # This yaw stuff is meant to correct the angles being given in a range [-180, 180] which makes it really discontinuous.
    # However, this correction code does not work.
    if R_vehicle < 0 and generate_control.prev_yaw > 0 and clockwise:
        generate_control.yaw_shift += 360
    elif R_vehicle > 0 and generate_control.prev_yaw < 0 and not clockwise:
        generate_control.yaw_shift -= 360
    R_vehicle += generate_control.yaw_shift

    p_target = waypoint.position
    pd_target = carla.Vector3D(0, 0, 0)

    cur_dist_to_tgt = distance(p_target, p_vehicle)
    orig_dist_to_tgt = distance(p_target, prev_waypoint.position)
    frac_dist_to_tgt = 1 - np.clip(cur_dist_to_tgt / (orig_dist_to_tgt), 0, 1)
    R_target = (1 - frac_dist_to_tgt) * prev_waypoint.rotation + frac_dist_to_tgt * waypoint.rotation
    Rd_target = 0

    Kp_p = 0.2
    Kd_p = 0.25

    Kp_R = 0.1
    Kd_R = 0

    control = carla.VehicleControl()

    travel_vector = p_target - p_vehicle
    # Ignoring perpendicular case for now
    control.reverse = False if dot(travel_vector, forward_dir) > 0 else True

    control.throttle = Kp_p * distance(p_target, p_vehicle)
    traveld_vector = pd_target - pd_vehicle
    if dot(pdd_vehicle, traveld_vector) > 0:
        control.throttle += Kd_p * distance(pd_target, pd_vehicle)
    else:
        control.throttle -= Kd_p * distance(pd_target, pd_vehicle)

    if control.throttle < 0:
        control.throttle = -control.throttle
        control.reverse = not control.reverse
    control.throttle = np.clip(control.throttle, 0, 1)
    
    # Need to verify that the derivative control on angle works
    control.steer = np.clip(Kp_R * (R_target - R_vehicle) + Kd_R * (Rd_target - Rd_vehicle) , -1, 1)

    moving_forward = True if dot(vehicle.get_velocity(), forward_dir) > 0 else False
    if not moving_forward:
        control.steer = -control.steer

    return control

generate_control.prev_yaw = None
generate_control.yaw_shift = 0

def follow_waypoints(client, world, vehicle, waypoints):
    assert(len(waypoints) >= 1)
    current_waypoint_i = 0
    current_waypoint = waypoints[current_waypoint_i]
    prev_waypoint = current_waypoint

    while True:
        position = vehicle.get_transform().location
        rotation = vehicle.get_transform().rotation.yaw

        print("")
        print(current_waypoint.position, current_waypoint.rotation)
        print(position, rotation)
        print(current_waypoint.error(position, rotation))
        if current_waypoint.error(position, rotation) < WAYPOINT_ERROR_THRESHOLD \
            and current_waypoint_i + 1 < len(waypoints):
            current_waypoint_i += 1
            prev_waypoint = current_waypoint
            current_waypoint = waypoints[current_waypoint_i]

        control = generate_control(client, world, vehicle, current_waypoint, prev_waypoint)
        vehicle.apply_control(control)
        
        world.tick()


def main(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    rng = np.random.default_rng(args.seed)

    vehicle = None
    try:
        world = client.get_world()
        spectator = world.get_spectator()

        # Select and spawn vehicle.
        vehicle_blueprint = rng.choice(world.get_blueprint_library().filter(args.filter))
        
        spawn_point = world.get_map().get_spawn_points()[12]
        vehicle = world.try_spawn_actor(vehicle_blueprint, spawn_point)

        # Move spectator to newly created vehicle.
        world.tick()
        world_snapshot = world.wait_for_tick()
        actor_snapshot = world_snapshot.find(vehicle.id)
        spectator.set_transform(actor_snapshot.get_transform())

        vehicle_position_init = vehicle.get_transform().location
        vehicle_rotation_init = vehicle.get_transform().rotation  

        waypoints = setup_waypoints(vehicle_position_init, vehicle_rotation_init)
        follow_waypoints(client, world, vehicle, waypoints)
    finally:
        # Destroy vehicle.
        client.apply_batch([carla.command.DestroyActor(vehicle)])

if __name__ == '__main__':
    try:
        argparser = argparse.ArgumentParser(description = __doc__)
        argparser.add_argument(
            '--host',
            metavar = 'H',
            default = '127.0.0.1',
            help = 'IP of the host server (default: 127.0.0.1)')
        argparser.add_argument(
            '-p', '--port',
            metavar = 'P',
            default = 2000,
            type = int,
            help = 'TCP port to listen to (default: 2000)')
        argparser.add_argument(
            '--filter',
            metavar = 'PATTERN',
            default = 'vehicle.carlamotors.carlacola',
            help = 'vehicles filter (default: vehicle.carlamotors.carlacola)')
        argparser.add_argument(
            '--tm-port',
            metavar = 'P',
            default = 8000,
            type = int,
            help = 'port to communicate with TM (default: 8000)')
        argparser.add_argument(
            '--sync',
            action = 'store_true',
            help = 'Synchronous mode execution')
        argparser.add_argument(
            '--hybrid',
            action = 'store_true',
            help = 'Enanble')
        argparser.add_argument(
            '-s', '--seed',
            metavar = 'S',
            type = int,
            help = 'Random device seed')
        args = argparser.parse_args()
        main(args)
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')