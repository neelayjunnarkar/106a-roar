""" Test program to follow a sequence of closely-spaced waypoints """

from pure_pursuit_control import PurePursuitController
import glob
import os
import sys
import time

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

import argparse
import numpy as np
from queue import Queue

WAYPOINT_ERROR_THRESHOLD = 3  # Error to consider waypoint reached
PASSENGER_ARRIVAL_RATE = 30  # Expected time between passengers in seconds
MAP_SAMPLING_RESOLUTION = 10 # Distance between possible waypoints for pickup / dropoff

class PassengerGenerator:
    def __init__(self, passenger_arrival_rate, rng):
        self.passenger_arrival_rate = passenger_arrival_rate
        self.rng = rng
        self.next_passenger_time = None

    def new_passenger_arrived(self, time):
        if self.next_passenger_time is None or time >= self.next_passenger_time:
            self.next_passenger_time = time + \
                self.rng.exponential(self.passenger_arrival_rate)
            return True
        else:
            return False

    def generate_new_passenger(self, possible_waypoints):
        return self.rng.choice(possible_waypoints, size=(2), replace=False)


class EndpointSelector:
    def __init__(self):
        self.passenger_points = Queue()
        self.current_passenger_point = None
        self.point = 0

    def add_passenger_point(self, passenger_point):
        self.passenger_points.put(passenger_point)

    def next_endpoint(self):
        if self.current_passenger_point is None or self.point == 1:
            if self.passenger_points.empty():
                print('Tried to select new passenger but there are no more passengers')
                return None
            else:
                print('Selected new passenger')
                self.current_passenger_point = self.passenger_points.get()
                self.point = 0
        else:
            print('Moving to drop off passenger')
            self.point = self.point + 1
            assert(self.point == 1)
        return self.current_passenger_point[self.point]


class Planner:
    def __init__(self, world_map, sampling_resolution, waypoint_error_threshold):
        self.global_route_planner_dao = GlobalRoutePlannerDAO(
            world_map, 1)
        self.global_route_planner = GlobalRoutePlanner(
            self.global_route_planner_dao)
        self.global_route_planner.setup()
        self.endpoint = None
        self.waypoints = None
        self.target_waypoint_i = 0
        self.waypoint_error_threshold = waypoint_error_threshold

    def set_endpoint(self, vehicle, endpoint):
        self.endpoint = endpoint
        vehicle_location = vehicle.get_transform().location
        endpoint_location = endpoint.transform.location
        self.waypoints = self.global_route_planner.trace_route(
            vehicle_location, endpoint_location)
        self.waypoints = [waypoint[0] for waypoint in self.waypoints]
        self.target_waypoint_i = 0
        print("Set new endpoint")
        print("Path length: ", len(self.waypoints))
        # print("Path: ")
        # for w in self.waypoints:
        #     print(w.transform.location)
        print("First waypoint: ", self.waypoints[0].transform.location)

    def get_target_waypoint(self, vehicle):
        vehicle_location = vehicle.get_transform().location
        target_waypoint = self.waypoints[self.target_waypoint_i]
        if target_waypoint.transform.location.distance(vehicle_location) < self.waypoint_error_threshold:
            if self.target_waypoint_i + 1 >= len(self.waypoints):
                print('No more waypoints')
                return None
            self.target_waypoint_i += 1
            target_waypoint = self.waypoints[self.target_waypoint_i]
            print('New waypoint (', self.target_waypoint_i, '/',
                  len(self.waypoints), '): ', target_waypoint.transform.location)
        return target_waypoint

    def reached_endpoint(self, vehicle):
        vehicle_location = vehicle.get_transform().location
        end_waypoint = self.waypoints[-1]
        if end_waypoint.transform.location.distance(vehicle_location) < self.waypoint_error_threshold:
            print('Reached endpoint')
            return True
        return False


def main(args):
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    rng = np.random.default_rng(args.seed)

    vehicle = None
    try:
        world = client.get_world()
        spectator = world.get_spectator()

        # Select and spawn vehicle.
        vehicle_blueprint = rng.choice(
            world.get_blueprint_library().filter(args.filter))

        world_map = world.get_map()
        spawn_point = world_map.get_spawn_points()[12]
        vehicle = world.try_spawn_actor(vehicle_blueprint, spawn_point)

        # Move spectator to newly created vehicle.
        world.tick()
        world_snapshot = world.wait_for_tick()
        actor_snapshot = world_snapshot.find(vehicle.id)
        spectator.set_transform(actor_snapshot.get_transform())
        print("Initial vehicle position: ", vehicle.get_transform().location)

        all_waypoints = world_map.generate_waypoints(10) # meters between pickup/dropoff positions 

        passenger_gen = PassengerGenerator(PASSENGER_ARRIVAL_RATE, rng)

        endpoint_selector = EndpointSelector()
        endpoint_selector.add_passenger_point(
            passenger_gen.generate_new_passenger(all_waypoints))
        current_endpoint = endpoint_selector.next_endpoint()
        assert(current_endpoint is not None)
        print(current_endpoint.transform.location)

        planner = Planner(world_map, MAP_SAMPLING_RESOLUTION,
                          WAYPOINT_ERROR_THRESHOLD)
        planner.set_endpoint(vehicle, current_endpoint)

        controller = PurePursuitController(vehicle, target_speed=25)

        while True:
            world_snapshot = world.wait_for_tick()
            time = world_snapshot.timestamp.elapsed_seconds

            if passenger_gen.new_passenger_arrived(time):
                print("New passenger at time ", time)
                endpoint_selector.add_passenger_point(
                    passenger_gen.generate_new_passenger(all_waypoints))

            target_waypoint = planner.get_target_waypoint(vehicle)
            while target_waypoint is None:
                current_endpoint = endpoint_selector.next_endpoint()
                if current_endpoint is None:
                    target_waypoint = world_map.get_waypoint(
                        vehicle.get_transform().location)
                    break
                planner.set_endpoint(vehicle, current_endpoint)
                target_waypoint = planner.get_target_waypoint(vehicle)

            control = controller.run_step(vehicle, target_waypoint)
            vehicle.apply_control(control)
    finally:
        # Destroy vehicle.
        print("Destroy Vehicle")
        client.apply_batch([carla.command.DestroyActor(vehicle)])


if __name__ == '__main__':
    try:
        argparser = argparse.ArgumentParser(description=__doc__)
        argparser.add_argument(
            '--host',
            metavar='H',
            default='127.0.0.1',
            help='IP of the host server (default: 127.0.0.1)')
        argparser.add_argument(
            '-p', '--port',
            metavar='P',
            default=2000,
            type=int,
            help='TCP port to listen to (default: 2000)')
        argparser.add_argument(
            '--filter',
            metavar='PATTERN',
            default='vehicle.carlamotors.carlacola',
            help='vehicles filter (default: vehicle.carlamotors.carlacola)')
        argparser.add_argument(
            '--tm-port',
            metavar='P',
            default=8000,
            type=int,
            help='port to communicate with TM (default: 8000)')
        argparser.add_argument(
            '--sync',
            action='store_true',
            help='Synchronous mode execution')
        argparser.add_argument(
            '--hybrid',
            action='store_true',
            help='Enanble')
        argparser.add_argument(
            '-s', '--seed',
            metavar='S',
            type=int,
            help='Random device seed')
        args = argparser.parse_args()
        main(args)
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
