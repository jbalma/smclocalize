import numpy as np
from scipy import special
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import csv
import json
import math
import time
import sys
import os

from smclocalize import *

data_json_path = './data/json/'
json_input_files = [x for x in os.listdir(data_json_path) if x.endswith('.json')]
dataframes = []

for json_input_file in json_input_files:
    with open(os.path.join(data_json_path, json_input_file), 'r') as input_fullpath:
        dataframes.append(pd.read_json(input_fullpath))

all_data = pd.concat(dataframes, ignore_index = True)

sensor_diagnostics = all_data.groupby('remote_id')['observed_at'].agg(['nunique', 'min', 'max']).rename(
    columns = {'nunique': 'n_times', 'min': 'min_time', 'max': 'max_time'}
)

usable_sensors = sensor_diagnostics[(sensor_diagnostics['n_times'] > 200) &
                                    (sensor_diagnostics['min_time'] < pd.Timestamp('2017-11-28 14:30')) &
                                    (sensor_diagnostics['max_time'] > pd.Timestamp('2017-11-28 15:50'))].index.values.tolist()

usable_sensors.remove(8)

usable_data = all_data[(all_data['remote_id'].isin(usable_sensors)) &
                       (all_data['local_id'].isin(usable_sensors)) &
                       (all_data['observed_at'] <= pd.Timestamp('2017-11-28 16:00'))].reset_index(drop=True)

child_entity_ids = np.union1d(pd.unique(usable_data[usable_data.local_type == 'child'].local_id),
                              pd.unique(usable_data[usable_data.remote_type == 'child'].remote_id)).tolist()
material_entity_ids = np.union1d(pd.unique(usable_data[usable_data.local_type == 'material'].local_id),
                                 pd.unique(usable_data[usable_data.remote_type == 'material'].remote_id)).tolist()
teacher_entity_ids = np.union1d(pd.unique(usable_data[usable_data.local_type == 'teacher'].local_id),
                                pd.unique(usable_data[usable_data.remote_type == 'teacher'].remote_id)).tolist()
area_entity_ids = np.union1d(pd.unique(usable_data[usable_data.local_type == 'area'].local_id),
                             pd.unique(usable_data[usable_data.remote_type == 'area'].remote_id)).tolist()

variable_structure = SensorVariableStructure(child_entity_ids,
                                             material_entity_ids,
                                             teacher_entity_ids,
                                             area_entity_ids)

timestamps = np.sort(usable_data['observed_at'].unique())
num_timesteps = len(timestamps)
y_discrete_t = np.ones(
    (num_timesteps, variable_structure.num_y_discrete_vars),
    dtype='int')
y_continuous_t = np.zeros(
    (num_timesteps, variable_structure.num_y_continuous_vars),
    dtype='float')
for t_index in range(num_timesteps):
    (y_discrete_t[t_index], y_continuous_t[t_index]) = variable_structure.sensor_data_parse_one_timestep(
        usable_data[usable_data['observed_at'] == timestamps[t_index]])

timestamp_range = pd.date_range(usable_data['observed_at'].min(), usable_data['observed_at'].max(), freq='10S')

print np.setdiff1d(timestamp_range, timestamps)

#Room geometry
feet_to_meters = 12*2.54/100
room_size = np.array([(19.0 + 4.0/12.0 + 15.0/12.0 + 43.0 + 2.0/12.0 + 2.0)*feet_to_meters,
                      (11.0 + 9.0/12.0)*feet_to_meters])
room_corners = np.array([[0.0, 0.0], room_size])

fixed_sensor_positions = np.array ([[(19.0 + 4.0/12.0 + 15.0/12.0 + 2.0)*feet_to_meters,
                                     (11.0 + 9.0/12.0 - 1.0)*feet_to_meters],
                                   [(2.0)*feet_to_meters,
                                    (11.0 + 9.0/12.0 - 1.0)*feet_to_meters],
                                   [(19.0 + 4.0/12.0 + 15.0/12.0 + 43.0 + 2.0/12.0 + 1.0)*feet_to_meters,
                                    (3.0)*feet_to_meters],
                                   [(19.0 + 4.0/12.0 + 15.0/12.0 + 15.0)*feet_to_meters,
                                    (1.0)*feet_to_meters],
                                    [(19.0 + 4.0/12.0 + 15.0/12.0 +3.0)*feet_to_meters,
                                    (3.0)*feet_to_meters]])

print("fixed_sensor_positions = %s" % fixed_sensor_positions)

sensor_model = SensorModel(
    variable_structure,
    room_corners,
    fixed_sensor_positions)

num_timesteps_test = 50
t_test = timestamps[:num_timesteps_test]

num_particles = 10000

# Initialize the variables.
x_discrete_particles_trajectory = np.zeros(
    (num_timesteps_test, num_particles, variable_structure.num_x_discrete_vars),
    dtype = 'int')
x_continuous_particles_trajectory = np.zeros(
    (num_timesteps_test, num_particles, variable_structure.num_x_continuous_vars),
    dtype = 'float')
log_weights_trajectory = np.zeros(
    (num_timesteps_test, num_particles),
    dtype = 'float')
ancestors_trajectory = np.zeros(
    (num_timesteps_test, num_particles),
    dtype = 'int')

# Generate the particles for the initial state  Xsub0
x_discrete_particles_trajectory[0], x_continuous_particles_trajectory[0], log_weights_trajectory[0] = sensor_model.generate_initial_particles(
    y_discrete_t[0],
    y_continuous_t[0],
    num_particles)

# Generate the particles for all later times.
start_time = time.time()
for t_index in range(1, num_timesteps_test):
    print '{}: {}'.format(t_index, t_test[t_index])
    x_discrete_particles_trajectory[t_index], x_continuous_particles_trajectory[t_index], log_weights_trajectory[t_index], ancestors_trajectory[t_index] = sensor_model.generate_next_particles(
        x_discrete_particles_trajectory[t_index - 1],
        x_continuous_particles_trajectory[t_index - 1],
        log_weights_trajectory[t_index - 1],
        y_discrete_t[t_index],
        y_continuous_t[t_index],
        t_test[t_index] - t_test[t_index - 1])

elapsed_time = time.time() - start_time
print("Total elapsed time: %ds" % int(elapsed_time))
print("Time per timestep: %f" % (elapsed_time / num_timesteps_test))

max_weights = np.max(np.exp(log_weights_trajectory), axis=1)

num_ancestors = np.array([len(np.unique(ancestors_trajectory[t_index])) for t_index in range(1, num_timesteps_test)])

x_continuous_mean_particle = np.average(
    x_continuous_particles_trajectory,
    axis=1,
    weights=np.repeat(np.exp(log_weights_trajectory), variable_structure.num_x_continuous_vars).reshape(
        (num_timesteps_test,
         num_particles,
         variable_structure.num_x_continuous_vars)))

x_continuous_squared_mean_particle = np.average(
    np.square(x_continuous_particles_trajectory),
    axis=1,
    weights=np.repeat(np.exp(log_weights_trajectory), variable_structure.num_x_continuous_vars).reshape(
        (num_timesteps_test,
         num_particles,
         variable_structure.num_x_continuous_vars)))

x_continuous_sd_particle = np.sqrt(np.abs(x_continuous_squared_mean_particle - np.square(x_continuous_mean_particle)))

print x_continuous_mean_particle
