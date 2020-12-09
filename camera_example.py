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

im_width = 640
im_height = 480

tMatrix = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], np.float32)

kf = cv2.KalmanFilter(4, 4)
kf.transitionMatrix = tMatrix
kf.measurementMatrix = np.array([
	[1, 0, 0, 0],
	[0, 1, 0, 0],
	[0, 0, 1, 0],
	[0, 0, 0, 1]
	], np.float32)
kf.processNoiseCov = cv2.setIdentity(kf.processNoiseCov, 1e-2)
kf.measurementNoiseCov = cv2.setIdentity(kf.measurementNoiseCov, 1e-1)
kf.errorCovPost = cv2.setIdentity(kf.errorCovPost, 1e-1)
print(kf.predict())

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((im_height, im_width, 4))
    img = np.copy(i2[:, :, :3])
    # cv2.imshow("", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imshow("b", blur)
    canny = cv2.Canny(blur, 30, 150)
    masked = cv2.bitwise_and(canny, region_of_interest(canny))
    lanes = line_image(masked)
    final = cv2.addWeighted(masked, 0.8, lanes, 1, 1)
    cv2.imshow("f", final)
    cv2.waitKey(1)

def region_of_interest(image):
	margin = 200
	poly = np.array([
		[(margin, im_height//2), (0, 9*im_height//16), (0, im_height), (im_width, im_height), (im_width, 9*im_height//16), (im_width-margin, im_height//2)]
		])
	# poly = np.array([
	# 	[(0, im_height//2), (0, im_height), (im_width, im_height), (im_width-margin, im_height//2)]
	# 	])
	mask = np.zeros_like(image)
	cv2.fillPoly(mask, poly, 255)
	return mask

def line_image(image):
	lanes = np.zeros_like(image)
	lines = cv2.HoughLinesP(image, 2, np.pi/180, 70, minLineLength=40, maxLineGap=5)

	left = []
	right = []
	if lines is not None:
	    for line in lines:
	    	x1, y1, x2, y2 = line[0]
	    	if x1 == x2 or abs(np.arctan(abs(y2-y1)/abs(x2-x1))) >= np.pi/36:
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

	    if not np.any(np.isnan(left_avg)) and not np.any(np.isnan(right_avg)):
	    	measurements = np.array(list(left_avg) + list(right_avg), dtype=np.float32)
	    	left_measure = make_coords(image, left_avg)
	    	right_measure = make_coords(image, right_avg)
	    	print(abs(right_measure[2]-left_measure[2]))
	    	if 200 <= abs(right_measure[2]-left_measure[2]) <= 400:
	    		kf.correct(measurements)
	    
	    prediction = kf.predict()
	    left_line = make_coords(image, prediction[:2])
	    right_line = make_coords(image, prediction[2:])
	    for x1, y1, x2, y2 in [left_line, right_line]:
		    cv2.line(lanes, (x1, y1), (x2, y2), (255, 0, 0), 10)
	return lanes

def average_slope(image, lines):
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
	print(f"Obstacle: {image.other_actor} is {image.distance} units away from {image.actor}.")


actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # world = client.load_world('Town03')
    world = client.get_world()
    settings = world.get_settings()
    world.set_weather(carla.WeatherParameters())
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(True)
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    actor_list.append(vehicle)

    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    blueprint = blueprint_library.find('sensor.camera.rgb')
    blueprint2 = blueprint_library.find('sensor.other.obstacle')
    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', f'{im_width}')
    blueprint.set_attribute('image_size_y', f'{im_height}')
    blueprint.set_attribute('fov', '110')

    blueprint2.set_attribute('debug_linetrace', f'{True}')

    # Adjust sensor relative to vehicle
    spawn_point = carla.Transform(carla.Location(x=2.5, z=0.7))

    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)
    # sensor2 = world.spawn_actor(blueprint2, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(sensor)
    # actor_list.append(sensor2)

    image_queue = queue.Queue()
    sensor.listen(image_queue.put)
    # do something with this sensor

    # start = time.clock()
    while True:
    	world.tick()
    	process_img(image_queue.get())
    	# print(start - time.clock())
    	# start = time.clock()

    # sensor2.listen(lambda data: process_img2(data))

    # sleep for 5 seconds, then finish:
    # time.sleep(15)

finally:
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')