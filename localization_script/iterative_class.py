#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 05:12:02 2017

@author: charles

A class wrapper for the HMM localization model with sampling using a particle
filter.

This version 
"""

import math
import time
import sys
import numpy as np
from scipy import misc
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib import patches

def left_truncnorm_logpdf(x, untruncated_mean, untruncated_std_dev, left_cutoff):
    logf = np.array(np.subtract(stats.norm.logpdf(x, loc=untruncated_mean,
                                                  scale=untruncated_std_dev),
    np.log(1 - stats.norm.cdf(left_cutoff,
                              loc=untruncated_mean,
                              scale=untruncated_std_dev))))
    logf[x < left_cutoff] = -np.inf
    return logf

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
                     room_width = 20.0, room_height = 10.0, stationary_sensor_positions = []):
        
        self.names, self.numbers = initialize_variables(moving_sensors,
                                                        stationary_sensors)
        
        self.sensor_names, self.dimension_names, self.x_names, self.y_discrete_names, self.y_continuous_names, self.ping_status_names = self.names
        
        self.num_moving_sensors, self.num_stationary_sensors, self.num_sensors, self.num_dimensions, self.num_x_vars, self.num_y_discrete_vars, self.num_y_continuous_vars, self.num_ping_statuses = self.numbers
        
        self.room_size = np.array([room_width, room_height])
        
        #Note that in subsequent iterations, we may want to
        #actually estimate some of these parameters as part of the modeling process.
        
        if len(stationary_sensor_positions) == 0:
            assert stationary_sensors == 4 #We only have default positions for
            #four stationary sensors. If positions aren't specified and this isn't
            #the case, we throw an error.
            self.stationary_sensor_position_guesses = self.room_size*np.array([[0.5, 1.0],
                                                                               [0.5, 0.0],
                                                                               [0.0, 0.5],
                                                                               [1.0, 0.5]])
        else:
            #Stationary sensor positions must be a numpy array
            assert type(stationary_sensor_positions) is np.ndarray
            #There should be as many stationary positions as there are stationary sensors.
            assert stationary_sensors == len(stationary_sensor_positions)
            #All stationary sensor positions should be within the bounds of the room.
            for (x, y), value in np.ndenumerate(stationary_sensor_positions):
                assert (0 <= x <= room_width) and (0 <= y <= room_height)
            #Assign to object variable if all conditions are met.
            self.stationary_sensor_position_guesses = stationary_sensor_positions
       
        #These are constants at the moment. Potentially change to all caps to 
        #reflect this?
        
        self.stationary_position_guess_error = 1.0
        
        self.receive_probability_zero_distance = 1.0 # Value from real data is approximately 1.0
        self.receive_probability_reference_distance = 0.7 #Value from real data is approximately 0.7
        self.reference_distance = 20.0 # Value from real data is approximately 20.0
        self.scale_factor = self.reference_distance/np.log(self.receive_probability_reference_distance/self.receive_probability_zero_distance)
        
        self.rssi_untruncated_mean_intercept = -64.0 # Value for real data is approximately -64.0
        self.rssi_untruncated_mean_slope = -20.0 # Value for real data is approximately -20.0
        self.rssi_untruncated_std_dev = 9.0 # Value for real data is approximately 9.0
        self.lower_rssi_cutoff = -82.0 # Value for real data is approximately -82.0

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
                                                scale=self.stationary_position_guess_error)
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
        
    def rssi_untruncated_mean(self, distance):
        return self.rssi_untruncated_mean_intercept + self.rssi_untruncated_mean_slope*np.log10(distance)
    
    def rssi_truncated_mean(self, distance):
        return stats.truncnorm.stats(a=(self.lower_rssi_cutoff - self.rssi_untruncated_mean(distance))/self.rssi_untruncated_std_dev,
                                     b=np.inf,
                                     loc=self.rssi_untruncated_mean(distance),
                                     scale=self.rssi_untruncated_std_dev,
                                     moments='m')
        
    def sample_rssi(self, distance):
        return stats.truncnorm.rvs(a=(self.lower_rssi_cutoff - self.rssi_untruncated_mean(distance))/self.rssi_untruncated_std_dev,
                                   b=np.inf,
                                   loc=self.rssi_untruncated_mean(distance),
                                   scale=self.rssi_untruncated_std_dev)
        
    def sample_y_continuous_bar_x(self, x):
        return self.sample_rssi(self.distances(x))
    
    def log_f_y_bar_x(self, x, y_discrete, y_continuous):
        distances_x = self.distances(x)
        discrete_log_probabilities = np.log(np.choose(y_discrete,
                                                      self.ping_status_probabilities(distances_x)))
        continuous_log_probability_densities = left_truncnorm_logpdf(y_continuous,
                                                                     self.rssi_untruncated_mean(distances_x),
                                                                     self.rssi_untruncated_std_dev,
                                                                     self.lower_rssi_cutoff)
        continuous_log_probability_densities[y_discrete == 1] = 0.0
        return np.sum(discrete_log_probabilities, axis=-1) + np.sum(continuous_log_probability_densities, axis=-1)

    def simulate_data(self, num_timesteps = 100, timestep_size = 1.0):
        t = np.zeros(num_timesteps, dtype='float')
        x_t = np.zeros((num_timesteps, self.num_x_vars), dtype='float')
        y_discrete_t = np.zeros((num_timesteps, self.num_y_discrete_vars), dtype='int')
        y_continuous_t = np.zeros((num_timesteps, self.num_y_continuous_vars), dtype='float')
        
        x_t[0] = self.sample_x_initial()
        y_discrete_t[0] = self.sample_y_discrete_bar_x(x_t[0])
        y_continuous_t[0] = self.sample_y_continuous_bar_x(x_t[0])
        
        for t_index in range(1,num_timesteps):
            t[t_index] = t[t_index - 1] + timestep_size
            x_t[t_index] = self.sample_x_bar_x_prev(x_t[t_index - 1])
            y_discrete_t[t_index] = self.sample_y_discrete_bar_x(x_t[t_index])
            y_continuous_t[t_index] = self.sample_y_continuous_bar_x(x_t[t_index])
        
        self.simulation_results = (x_t, y_discrete_t, y_continuous_t, num_timesteps)
        return(self.simulation_results)

    def sample_posterior(self, num_particles = 10000, sensor_data = ()):
        
        if len(sensor_data) == 0:
            try:
                self.simulation_results
            except NameError:
                print "No simulation data exists."
            else:
                sensor_data = self.simulation_results
        
        x_t, y_discrete_t, y_continuous_t, num_timesteps = sensor_data
        particle_values = np.zeros((num_timesteps, num_particles, self.num_x_vars), dtype = 'float')
        log_particle_weights = np.zeros((num_timesteps, num_particles), dtype = 'float')
        sampled_particle_indices = np.zeros((num_timesteps, num_particles), dtype = 'int')
        
        particle_values[0] = self.sample_x_initial(num_particles)
        log_particle_weights[0] = self.log_f_y_bar_x(particle_values[0],
                            np.tile(y_discrete_t[0], (num_particles,1)),
                            np.tile(y_continuous_t[0], (num_particles,1)))
        log_particle_weights[0] = log_particle_weights[0] - misc.logsumexp(log_particle_weights[0])
        
        sys.stdout.write('t_index =')
        sys.stdout.flush()
        time_start = time.clock()
        for t_index in range(1, num_timesteps):
            sys.stdout.write(' {}'.format(t_index))
            sys.stdout.flush()
            sampled_particle_indices[t_index - 1] = np.random.choice(num_particles,
                                                                     size=num_particles,
                                                                     p=np.exp(log_particle_weights[t_index - 1]))
            particle_values[t_index] = self.sample_x_bar_x_prev(particle_values[t_index - 1,
                           sampled_particle_indices[t_index - 1]])
            log_particle_weights[t_index] = self.log_f_y_bar_x(particle_values[t_index],
                                np.tile(y_discrete_t[t_index], (num_particles,1)),
                                np.tile(y_continuous_t[t_index], (num_particles,1)))
            log_particle_weights[t_index] = log_particle_weights[t_index]- misc.logsumexp(log_particle_weights[t_index])
            
        x_mean_particle = np.average(particle_values, axis=1,
                                     weights=np.repeat(np.exp(log_particle_weights),
                                                       self.num_x_vars).reshape((num_timesteps, num_particles, self.num_x_vars)))
        
        x_squared_mean_particle = np.average(np.square(particle_values), 
                                             axis=1,
                                             weights=np.repeat(np.exp(log_particle_weights),
                                                               self.num_x_vars).reshape((num_timesteps,
                                                                         num_particles,
                                                                         self.num_x_vars)))
        
        x_sd_particle = np.sqrt(np.abs(x_squared_mean_particle - np.square(x_mean_particle)))
        self.x_estimate = (x_mean_particle, x_sd_particle)
        print'\nTime elapsed = {}'.format(time.clock()-time_start)
        return(self.x_estimate)

    def sample_initial_posterior(self, sensor_data, num_particles=10000):
        y_discrete, y_continuous = sensor_data
        particle_values = np.zeros((num_particles, self.num_x_vars), dtype = 'float')
        log_particle_weights = np.zeros((num_particles), dtype = 'float')
        
        particle_values = self.sample_x_initial(num_particles)
        log_particle_weights = self.log_f_y_bar_x(particle_values,
                            np.tile(y_discrete, (num_particles,1)),
                            np.tile(y_continuous, (num_particles,1)))
        log_particle_weights = log_particle_weights - misc.logsumexp(log_particle_weights)
        
        particles = (particle_values, log_particle_weights)
        
        x_mean_particle = np.average(particle_values, axis=1,
                                     weights=np.repeat(np.exp(log_particle_weights),
                                                       self.num_x_vars).reshape((num_particles, self.num_x_vars)))
        
        x_squared_mean_particle = np.average(np.square(particle_values), 
                                             axis=1,
                                             weights=np.repeat(np.exp(log_particle_weights),
                                                               self.num_x_vars).reshape((num_particles,
                                                                              self.num_x_vars)))
        
        x_sd_particle = np.sqrt(np.abs(x_squared_mean_particle - np.square(x_mean_particle)))
        x = (x_mean_particle, x_sd_particle)
        return(x, particles)
    
    def update_posterior_sample(self, sensor_data, particles_prev):
        y_discrete, y_continuous = sensor_data
        particle_values_prev, log_particle_weights_prev = particles_prev
        num_particles = len(particle_values_prev)
        assert len(particle_values_prev) == len(log_particle_weights_prev)
        sampled_particle_indices = np.zeros(num_particles, dtype = 'int')
        sampled_particle_indices = np.random.choice(num_particles, size=num_particles,
                                                    p=np.exp(log_particle_weights_prev))
        particle_values = self.sample_x_bar_x_prev(particle_values_prev[sampled_particle_indices])
        log_particle_weights = self.log_f_y_bar_x(particle_values,
                                                      np.tile(y_discrete, (num_particles,1)),
                                                      np.tile(y_continuous, (num_particles,1)))
        log_particle_weights = log_particle_weights - misc.logsumexp(log_particle_weights)
        
        particles = (particle_values, log_particle_weights)
            
        x_mean_particle = np.average(particle_values, axis=1,
                                     weights=np.repeat(np.exp(log_particle_weights),
                                                       self.num_x_vars).reshape((num_particles, self.num_x_vars)))
        
        x_squared_mean_particle = np.average(np.square(particle_values), 
                                             axis=1,
                                             weights=np.repeat(np.exp(log_particle_weights),
                                                               self.num_x_vars).reshape((num_particles,
                                                                              self.num_x_vars)))
        
        x_sd_particle = np.sqrt(np.abs(x_squared_mean_particle - np.square(x_mean_particle)))
        x = (x_mean_particle, x_sd_particle)
        
        return(x, particles)

#Testing the new functionality of variable stationary sensors with positions
#passed in as an argument to the constructor.
stationary_sensor_pos = np.array([[0.0, 5.0],
                                  [20.0, 5.0],
                                  [10.0, 0.0],
                                  [10.0, 10.0],
                                  [20.0, 10.0]])

test_model_2 = SensorModel(10, 5, 20.0, 10.0, stationary_sensor_pos)

test_model_2.simulate_data()
test_model_2.simulation_results

x_t, y_discrete_t, y_continuous_t, num_timesteps = test_model_2.simulation_results

x_t[0]
y_discrete_t[0]
y_continuous_t[0]

y_discrete = y_discrete_t[1]
y_continuous = y_continuous_t[1]

sensor_data = (y_discrete, y_continuous)

x, particles = test_model_2.sample_initial_posterior(sensor_data)

x_1, particles_1 = test_model_2.update_posterior_sample(sensor_data, particles)

test_model_2.sample_posterior()

