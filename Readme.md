106a ROAR Variant
=========

106a final project of Sareum, Amit, Neelay

In this project, we implement parts of a 'taxicab service' where a car drives around on a road network, 'picking up and dropping off' passengers.

We simulate the arrival of passengers as random pairs of 'pick-up' and 'drop-off' points, use the Carla API to generate a list of turning directions at intersections, and use lane detection and lane following to stay on the road and in a lane between intersections. Lane following becomes particularly apparent on curved roads.

# Dependencies

* [Carla](http://carla.org/) simulation environment. Tested with version 0.9.10
* Python. Tested with version 3.7.

# Installation

Once Carla is installed, clone this repo into `Carla/PythonAPI`. It should be in the same directory as the `examples` folder.

# Running

Once the Carla server is started (e.g. by running `Carla.exe`), run `python 106a_interpolator.py` to start the final version of our code.
This will spawn a vehicle, select random points to drive it to, and then use vision to detect lanes and follow the lanes, turning at intersections.
Two more windows may pop up showing the camera view in gray-scale and the output of the lane detector.
