# -*- coding: utf-8 -*-
"""
This script defines a function that takes in sensor data & a set of previously estimated sensor positions, and output a new set of estimated positions for the sensors.

This initial version is a simple adaptation of the code Ted created in the notebook bayes_filter_sim_rssi.ipynb.
"""

import math
import time
import sys
import numpy as np
from scipy import misc
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import patches

#This is right from Ted's code. Simply initializing the number of sensors.
num_moving_sensors = 3
num_stationary_sensors = 4 # Currently must be 4 or fewer because we manually place them from a fixed list of 4 positions

#Initialize sensor_names to empty array.
sensor_names = []

#Self explanatory, but one thing to note is that arrays start with an index of
#zero in Python, rather than 1 in R. Hence the "+1" below.
for moving_sensor_index in range(num_moving_sensors):
    sensor_names.append('Moving sensor {}'.format(moving_sensor_index + 1))
    
for stationary_sensor_index in range(num_stationary_sensors):
    sensor_names.append('Stationary sensor {}'.format(stationary_sensor_index + 1))

num_sensors = len(sensor_names)