#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:23:02 2017

@author: charles

A class wrapper for the HMM localization model with sampling using a particle
filter.
"""

import math
import time
import sys
import numpy as np
from scipy import misc
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import patches

def initialize_variables(num_moving_sensors, num_stationary_sensors):
    
    #Initialize variables
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
    
    names = (sensor_names, dimension_names, x_names, y_discrete_names, y_continuous_names, ping_status_names)
    numbers = (num_moving_sensors, num_stationary_sensors, num_sensors, num_dimensions, num_x_vars, num_y_discrete_vars,
               num_y_continuous_vars, num_ping_statuses) 
    return(names, numbers)

class SensorModel:
    
    def __init__(self, moving_sensors = 3, stationary_sensors = 4,
                     room_width = 20.0, room_height = 10.0):
        
        self.names, self.numbers = initialize_variables(moving_sensors,
                                                        stationary_sensors)
        
        self.sensor_names, self.dimension_names, self.x_names, self.y_discrete_names, self.y_continuous_names, self.ping_status_names = self.names
        
        self.num_moving_sensors, self.num_stationary_sensors, self.num_sensors, self.num_dimensions, self.num_x_vars, self.num_y_discrete_vars, self.num_y_continuous_vars, self.num_ping_statuses = self.numbers
        
        self.room_size = np.array([room_width, room_height])

    def sample_x_initial(self, num_samples = 1, stationary_position_guess_error = 1.0):
        self.stationary_sensor_position_guesses = self.room_size*np.array([[0.5, 1.0],
                                                                           [0.5, 0.0],
                                                                           [0.0, 0.5],
                                                                           [1.0, 0.5]])[0:self.num_stationary_sensors]
        
        if self.num_moving_sensors > 0:
            moving_sensors = np.random.uniform(high=np.tile(self.room_size, self.num_moving_sensors),
                                               size=(num_samples, self.num_moving_sensors*self.num_dimensions))
        else:
            moving_sensors = np.array([])
        if self.num_stationary_sensors > 0:
            stationary_sensors = np.random.normal(loc=self.stationary_sensor_position_guesses.flatten(),
                                                  scale=stationary_position_guess_error,
                                                  size=(num_samples, self.num_stationary_sensors*self.num_dimensions))
        else:
            stationary_sensors = np.array([])
        return np.squeeze(np.concatenate((moving_sensors, stationary_sensors), axis=1))
    
    def plot_x_initial_samples(self, num_samples = 1, stationary_position_guess_error = 1.0):
        x_initial_samples = self.sample_x_initial(num_samples, stationary_position_guess_error)
        for sensor_index in range(self.num_sensors):
            plt.plot(x_initial_samples[:,sensor_index*2],
                     x_initial_samples[:,sensor_index*2 + 1],
                     'b.',
                     alpha = 0.2,
                     label="Samples")
            plt.title(self.sensor_names[sensor_index])
            plt.xlabel('{} position'.format(self.dimension_names[0]))
            plt.ylabel('{} position'.format(self.dimension_names[1]))
            ax=plt.gca()
            ax.add_patch(patches.Rectangle((0,0),
                                           self.room_size[0],
                                           self.room_size[1],
                                           fill=False,
                                           color='green',
                                           label='Room boundary'))
            plt.xlim(0 - stationary_position_guess_error,
                     self.room_size[0] + stationary_position_guess_error)
            plt.ylim(0 - stationary_position_guess_error,
                     self.room_size[1] + stationary_position_guess_error)
            plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            plt.show()


test_model = SensorModel(3, 4, 20.0, 10.0)

x_initial_samples = test_model.sample_x_initial(1000, 1.0)

test_model.plot_x_initial_samples(1000, 1.0)