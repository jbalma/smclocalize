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
        
        self.stationary_sensor_position_guesses = self.room_size*np.array([[0.5, 1.0],
                                                                           [0.5, 0.0],
                                                                           [0.0, 0.5],
                                                                           [1.0, 0.5]])[0:self.num_stationary_sensors]
        
        self.stationary_position_guess_error = 1.0
        
        self.receive_probability_zero_distance = 1.0 # Value from real data is approximately 1.0
        self.receive_probability_reference_distance = 0.7 #Value from real data is approximately 0.7
        self.reference_distance = 20.0 # Value from real data is approximately 20.0
        self.scale_factor = self.reference_distance/np.log(self.receive_probability_reference_distance/self.receive_probability_zero_distance)

    def sample_x_initial(self, num_samples = 1):
        if self.num_moving_sensors > 0:
            moving_sensors = np.random.uniform(high=np.tile(self.room_size, self.num_moving_sensors),
                                               size=(num_samples, self.num_moving_sensors*self.num_dimensions))
        else:
            moving_sensors = np.array([])
        if self.num_stationary_sensors > 0:
            stationary_sensors = np.random.normal(loc=self.stationary_sensor_position_guesses.flatten(),
                                                  scale=self.stationary_position_guess_error,
                                                  size=(num_samples, self.num_stationary_sensors*self.num_dimensions))
        else:
            stationary_sensors = np.array([])
        return np.squeeze(np.concatenate((moving_sensors, stationary_sensors), axis=1))
    
    def plot_x_initial_samples(self, num_samples = 1):
        x_initial_samples = self.sample_x_initial(num_samples)
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
            plt.xlim(0 - self.stationary_position_guess_error,
                     self.room_size[0] + self.stationary_position_guess_error)
            plt.ylim(0 - self.stationary_position_guess_error,
                     self.room_size[1] + self.stationary_position_guess_error)
            plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
            plt.show()
    
    def sample_x_bar_x_prev(self, x_prev, moving_sensor_drift = 0.5, stationary_sensor_drift = 0.25):
        if self.num_moving_sensors > 0:
            x_prev_moving_sensors = x_prev[...,:self.num_moving_sensors*self.num_dimensions]
            moving_sensor_positions = np.random.normal(loc=x_prev_moving_sensors,
                                                       scale=moving_sensor_drift)
        else:
            moving_sensor_positions = np.array([])
        if self.num_stationary_sensors > 0:
            x_prev_stationary_sensors = x_prev[...,self.num_moving_sensors*self.num_dimensions:]
            position_guesses = np.random.normal(loc=np.tile(self.stationary_sensor_position_guesses.flatten(),
                                                            x_prev_stationary_sensors.shape[:-1] + (1,)),
                                                scale=self.stationary_sensor_position_guess_error)
            drifted_positions = np.random.normal(loc=x_prev_stationary_sensors,
                                                 scale=stationary_sensor_drift)
            stationary_sensor_positions = (position_guesses + drifted_positions)/2
        else:
            stationary_sensor_positions = np.array([])
        return np.concatenate((moving_sensor_positions, stationary_sensor_positions), axis=-1)

    def distances(self, x):
        if x.ndim == 1:
            x = np.expand_dims(x, 0)
        num_x_values = x.shape[0]
        return np.squeeze(np.delete(np.linalg.norm(np.subtract(np.tile(x.reshape((num_x_values, self.num_sensors, self.num_dimensions)),
                                                                       (1, self.num_sensors, 1)),
                np.repeat(x.reshape((num_x_values,
                                     self.num_sensors,
                                     self.num_dimensions)),
                self.num_sensors,
                axis = 1)),
                axis=2),
                np.arange(self.num_sensors)*self.num_sensors + np.arange(self.num_sensors),1))

    def ping_status_probabilities(self, distance):
        receive_probability=self.receive_probability_zero_distance*np.exp(distance/self.scale_factor)
        return np.stack((receive_probability, 1 - receive_probability), axis=0)
    
    def sample_y_discrete_bar_x(self, x):
        return np.apply_along_axis(lambda p_array: np.random.choice(len(p_array), p=p_array),
                                   axis=0,
                                   arr=self.ping_status_probabilities(self.distances(x)))




del test_model

test_model = SensorModel(3, 4, 20.0, 10.0)

x_initial_samples = test_model.sample_x_initial(1000)

test_model.plot_x_initial_samples(1000)

test_x_value = test_model.sample_x_initial()

test_model.sample_y_discrete_bar_x(np.tile(test_x_value, (1000, 1)))

