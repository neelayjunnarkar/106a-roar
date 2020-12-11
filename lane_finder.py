import glob
import os
import sys
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

import random
import time
import numpy as np
import cv2
import queue

class LaneFinder:
    im_width = 640
    im_height = 480
    new_factor = 0.5

    def __init__(self, world, vehicle):
        self.world = world
        self.vehicle = vehicle

        self.tMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]], np.float32)

        self.kf = cv2.KalmanFilter(4, 4)
        self.kf.transitionMatrix = self.tMatrix
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        self.kf.processNoiseCov = cv2.setIdentity(self.kf.processNoiseCov, 1e-2)
        self.kf.measurementNoiseCov = cv2.setIdentity(self.kf.measurementNoiseCov, 1e-3)
        self.kf.errorCovPost = cv2.setIdentity(self.kf.errorCovPost, 1e-1)

        print(self.kf.predict())

        self.left_x = 0
        self.right_x = 0

        # Setup sensor
        blueprint_library = world.get_blueprint_library()
        # https://carla.readthedocs.io/en/latest/cameras_and_sensors
        # get the blueprint for this sensor
        self.blueprint = blueprint_library.find('sensor.camera.rgb')
        print(self.blueprint)
        # change the dimensions of the image
        self.blueprint.set_attribute('image_size_x', f'{self.im_width}')
        self.blueprint.set_attribute('image_size_y', f'{self.im_height}')
        self.blueprint.set_attribute('fov', '110')

        # Adjust sensor relative to vehicle
        self.spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

        # spawn the sensor and attach to vehicle.
        self.sensor = world.spawn_actor(self.blueprint, self.spawn_point, attach_to=self.vehicle)
        # print('sensor: ', self.sensor)
        self.image_queue = queue.Queue()
        self.sensor.listen(self.image_queue.put)

    def process_img(self, image):
        i = np.array(image.raw_data)
        i2 = i.reshape((self.im_height, self.im_width, 4))
        img = np.copy(i2[:, :, :3])
        # cv2.imshow("", img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        cv2.imshow("b", blur)
        canny = cv2.Canny(blur, 30, 150)
        masked = cv2.bitwise_and(canny, self.region_of_interest(canny))
        lanes = self.line_image(masked)
        final = cv2.addWeighted(masked, 0.8, lanes, 1, 1)
        cv2.imshow("f", final)
        cv2.waitKey(1)

    def region_of_interest(self, image):
        margin = 200
        poly = np.array([
            [(margin, self.im_height//2), (0, 9*self.im_height//16), (0, self.im_height), (self.im_width,
                                                                                           self.im_height), (self.im_width, 9*self.im_height//16), (self.im_width-margin, self.im_height//2)]
        ])
        # poly = np.array([
        # 	[(0, im_height//2), (0, im_height), (im_width, im_height), (im_width-margin, im_height//2)]
        # 	])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, poly, 255)
        return mask

    def line_image(self, image):
        lanes = np.zeros_like(image)
        lines = cv2.HoughLinesP(image, 2, np.pi/180, 70,
                                minLineLength=40, maxLineGap=5)

        left = []
        right = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x1 == x2 or abs(np.arctan(abs(y2-y1)/abs(x2-x1))) >= np.pi/10:
                    m, b = np.polyfit((x1, x2), (y1, y2), 1)
                    if m < 0:
                        left.append((m, b))
                    else:
                        right.append((m, b))
            left_avg = np.average(left, axis=0)
            right_avg = np.average(right, axis=0)

            def make_coords(image, params):
                (m, b) = params
                y1 = image.shape[0]
                y2 = int(y1*3/5)
                x1 = int((y1-b)/(m+.00001))
                x2 = int((y2-b)/(m+.00001))
                return np.array([x1, y1, x2, y2])

            def left_coord(params):
                (m, b) = params
                return int((y1-b)/(m+.00001))

            def right_coord(params):
                (m, b) = params
                return int((y2-b)/(m+.00001))

            if not np.any(np.isnan(left_avg)) and not np.any(np.isnan(right_avg)):
                measurements = np.array(
                    list(left_avg) + list(right_avg), dtype=np.float32)
                left_measure = make_coords(image, left_avg)
                right_measure = make_coords(image, right_avg)
                # print(abs(right_measure[2]-left_measure[2]))
                if 200 <= right_measure[2]-left_measure[2] <= 350:
                    self.kf.correct(measurements)

            prediction = self.kf.predict()
            left_line = make_coords(image, prediction[:2])
            right_line = make_coords(image, prediction[2:])

            for x1, y1, x2, y2 in [left_line, right_line]:
                cv2.line(lanes, (x1, y1), (x2, y2), (255, 0, 0), 10)

            self.left_x = (1-self.new_factor) * self.left_x + \
                self.new_factor * left_coord(prediction[:2])
            self.right_x = (1-self.new_factor) * self.right_x + \
                self.new_factor * right_coord(prediction[2:])
        return lanes

    def get_left_lane_x(self):
        return self.im_width / 2.0 - self.left_x
    def get_right_lane_x(self):
        return self.right_x  - self.im_width / 2.0
    
    def average_slope(self, image, lines):
        left = []
        right = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            m, b = np.polyfit((x1, x2), (y1, y2), 1)
            if m < 0:
                left.append((m, b))
            else:
                right.append((m, b))
        left_avg = np.average(left, axis=0)
        right_avg = np.average(right, axis=0)


    def process_img2(image):
        print(
            f"Obstacle: {image.other_actor} is {image.distance} units away from {image.actor}.")

    def run_step(self):
        if not self.image_queue.empty():
            self.process_img(self.image_queue.get())