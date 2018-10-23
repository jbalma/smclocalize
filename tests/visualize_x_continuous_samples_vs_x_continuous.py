#!/usr/bin/env python
# coding: utf-8

# # Visualize $X$ continuous samples vs. $X$ continuous

# ## Specify the data run

# In[1]:


io_directory = 'C:/Users/anonymous/Desktop/Wildflower/Wfdata'
run_identifier = 'aster_data'


# ## Load the libraries we need

# Load the third-party libraries.

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import os
import pickle


# Load our `smclocalize` module.

# In[3]:


from smclocalize import *


# ## Load the required data

# In[4]:


entity_ids_data = np.load(
    os.path.join(
        io_directory,
        run_identifier,
        'entity_ids_data.npz'))
child_entity_ids = entity_ids_data['child_entity_ids']
material_entity_ids = entity_ids_data['material_entity_ids']
teacher_entity_ids = entity_ids_data['teacher_entity_ids']
area_entity_ids = entity_ids_data['area_entity_ids']


# In[5]:


timestamps_data = np.load(
    os.path.join(
        io_directory,
        run_identifier,
        'timestamps_data.npz'))
timestamps = timestamps_data['timestamps']
num_timesteps = len(timestamps)


# In[6]:


room_geometry_data = np.load(
    os.path.join(
        io_directory,
        run_identifier,
        'room_geometry_data.npz'))
fixed_sensor_positions = room_geometry_data['fixed_sensor_positions']
room_corners = room_geometry_data['room_corners']


# In[7]:


x_samples_summaries = np.load(
    os.path.join(
        io_directory,
        run_identifier,
        'x_samples_summaries.npz'))
num_nonzero_weights = x_samples_summaries['num_nonzero_weights']
num_nonnegligible_weights = x_samples_summaries['num_nonnegligible_weights']
num_after_resampling = x_samples_summaries['num_after_resampling']
x_discrete_mean_particle = x_samples_summaries['x_discrete_mean_particle']
x_continuous_mean_particle = x_samples_summaries['x_continuous_mean_particle']
x_continuous_sd_particle = x_samples_summaries['x_continuous_sd_particle']


# In[8]:


x_continuous_data = np.load(
    os.path.join(
        io_directory,
        run_identifier,
        'x_continuous_data.npz'))
x_continuous_t = x_continuous_data['x_continuous_t']


# ## Define the variable structure for the model

# In[9]:


variable_structure = SensorVariableStructure(child_entity_ids,
                                             material_entity_ids,
                                             teacher_entity_ids,
                                             area_entity_ids)


# ## Compare the model results to the ground truth data

# In[10]:


plt.rcParams["date.autoformatter.minute"] = "%H:%M:%S"
for x_var_index in range(variable_structure.num_x_continuous_vars):
    plt.plot(
        timestamps,
        x_continuous_mean_particle[:,x_var_index],
        'b-',
        alpha=0.5,
        label='Particle sample mean')
    plt.plot(
        timestamps,
        x_continuous_t[:,x_var_index],
        'g-',
        alpha=1.0,
        label='Actual position')
    plt.gcf().autofmt_xdate()
    plt.axhline(0, color='black', linestyle='dashed', label='Room boundary')
    plt.axhline(room_corners[1][x_var_index % 2], color='black', linestyle='dashed')
    plt.ylim(
        0 - 0.05*np.max(room_corners[1]),
        np.max(room_corners[1]) + 0.05*np.max(room_corners[1]))
    plt.xlabel('$t$')
    plt.ylabel('Position (meters)')
    plt.title(variable_structure.x_continuous_names[x_var_index])
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.show()
plt.rcParams["date.autoformatter.minute"] = plt.rcParamsDefault["date.autoformatter.minute"]


# Plot the sample confidence regions of the particles (i.e., sample means plus/minus sample standard deviations).

# In[11]:


plt.rcParams["date.autoformatter.minute"] = "%H:%M:%S"
for x_var_index in range(variable_structure.num_x_continuous_vars):
    plt.fill_between(
        timestamps,
        x_continuous_mean_particle[:,x_var_index] - x_continuous_sd_particle[:, x_var_index],
        x_continuous_mean_particle[:,x_var_index] + x_continuous_sd_particle[:, x_var_index],
        color='blue',
        alpha=0.2,
        label='Particle sample CR'
    )
    plt.plot(
        timestamps,
        x_continuous_t[:,x_var_index],
        'g-',
        alpha=1.0,
        label='Actual position')
    plt.gcf().autofmt_xdate()
    plt.axhline(0, color='black', linestyle='dashed', label='Room boundary')
    plt.axhline(room_corners[1][x_var_index % 2], color='black', linestyle='dashed')
    plt.ylim(
        0 - 0.05*np.max(room_corners[1]),
        np.max(room_corners[1]) + 0.05*np.max(room_corners[1]))
    plt.xlabel('$t$')
    plt.ylabel('Position (meters)')
    plt.title(variable_structure.x_continuous_names[x_var_index])
    plt.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0))
    plt.show()
plt.rcParams["date.autoformatter.minute"] = plt.rcParamsDefault["date.autoformatter.minute"]

