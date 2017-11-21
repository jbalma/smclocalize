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

#Parameters

num_moving_sensors = 3
num_stationary_sensors = 4 # Currently must be 4 or fewer because we manually place them from a fixed list of 4 positions





#Variables

#Initialize sensor_names to empty array.
sensor_names = []

for moving_sensor_index in range(num_moving_sensors):
    sensor_names.append('Moving sensor {}'.format(moving_sensor_index + 1))
    
for stationary_sensor_index in range(num_stationary_sensors):
    sensor_names.append('Stationary sensor {}'.format(stationary_sensor_index + 1))

num_sensors = len(sensor_names)

dimension_names = [
    '$h$',
    '$v$'
]

num_dimensions = len(dimension_names)

x_names = []

for sensor_index in range(num_sensors):
    for dimension_index in range(num_dimensions):
        x_names.append('{} {} position'.format(sensor_names[sensor_index],
                                                    dimension_names[dimension_index]))


y_discrete_names = []
y_continuous_names = []

for sending_sensor_index in range(num_sensors):
    receiving_sensor_range = range(num_sensors)
    del receiving_sensor_range[sending_sensor_index]
    for receiving_sensor_index in receiving_sensor_range:
        y_discrete_names.append('Status of ping from {} to {}'.format(sensor_names[sending_sensor_index],
                                                                           sensor_names[receiving_sensor_index]))        
        y_continuous_names.append('RSSI of ping from {} to {}'.format(sensor_names[sending_sensor_index],
                                                                           sensor_names[receiving_sensor_index]))
        
ping_status_names = [
    'Received',
    'Not received'
]

num_x_vars = len(x_names)
num_y_discrete_vars = len(y_discrete_names)
num_y_continuous_vars = len(y_continuous_names)

num_ping_statuses = len(ping_status_names)

room_size = np.array([20.0, 10.0])

stationary_sensor_position_guesses = room_size*np.array([[0.5, 1.0],
                                                         [0.5, 0.0],
                                                         [0.0, 0.5],
                                                         [1.0, 0.5]])[0:num_stationary_sensors]

stationary_sensor_position_guess_error = 1.0




class SensorModel:
    def __init__(self, moving_sensors, stationary_sensors, room_width, room_height):
        














