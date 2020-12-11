import math

from pid_controller.pid import PID
from carla.client import VehicleControl



class Controller(object):

    def __init__(self, params):
        self.params = params
        # PID speed controller
        self.pid = PID(p=params['pid_p'], i=params['pid_i'], d=params['pid_d'])

    def get_control(self, wp_angle, wp_angle_speed, speed_factor, current_speed):
        
        control = VehicleControl()
        current_speed = max(current_speed, 0)

        steer = self.params['steer_gain'] * wp_angle
        if steer > 0:
            control.steer = min(steer, 1)
        else:
            control.steer = max(steer, -1)

        if math.fabs(wp_angle_speed) < 0.1:
            target_speed_adjusted = self.params['target_speed'] * speed_factor
        # Depending on the angle of the curve the speed is either 20 (beginning) 15 (most of it)
        elif math.fabs(wp_angle_speed) < 0.5:
            target_speed_adjusted = 20 * speed_factor
        else:
            target_speed_adjusted = 15 * speed_factor

        self.pid.target = target_speed_adjusted
        pid_gain = self.pid(feedback=current_speed)
       
        throttle = min(max(self.params['default_throttle'] - 1.3 * pid_gain, 0),
                       self.params['throttle_max'])

        if pid_gain > 0.5:
            brake = min(0.35 * pid_gain * self.params['brake_strength'], 1)
        else:
            brake = 0

        control.throttle = max(throttle, 0)
        control.brake = brake


        return control
