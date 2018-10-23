#!/usr/bin/env python
# coding: utf-8

# # Generate $X$ samples

# ## Specify the data run

# In[5]:


io_directory = '/tmp/'
run_identifier = 'aster_data'

num_steps = 100
npart = 190000
#npart = 100000
#use npart = 100000 for default on CPU 
# ## Load the libraries we need

# Load the third-party libraries.

# In[6]:


import numpy as np
import os
import pickle
import time
import threading
# Load our `smclocalize` module.

# In[7]:
import gc

from smclocalize import *


# ## Load the ping data

# In[8]:


entity_ids_data = np.load(
    os.path.join(
        io_directory,
        run_identifier,
        'entity_ids_data.npz'))
child_entity_ids = entity_ids_data['child_entity_ids']
material_entity_ids = entity_ids_data['material_entity_ids']
teacher_entity_ids = entity_ids_data['teacher_entity_ids']
area_entity_ids = entity_ids_data['area_entity_ids']


# In[9]:
#with tf.device.cpu

timestamps_data = np.load(
    os.path.join(
        io_directory,
        run_identifier,
        'timestamps_data.npz'))



if 1: 
  # In[10]:
  
  
  y_data = np.load(
      os.path.join(
          io_directory,
          run_identifier,
          'y_data.npz'))
  
  y_discrete_t = y_data['y_discrete_t']
  y_continuous_t = y_data['y_continuous_t']
  
  
  # ## Define the variable structure for the model
  
  # Using the lists of entity IDs, define an instance of the `SensorVariableStructure` class. This class provides a whole bunch of variables and helper functions for working with the data.
  
  # In[11]:
  
  
  variable_structure = SensorVariableStructure(child_entity_ids,
                                               material_entity_ids,
                                               teacher_entity_ids,
                                               area_entity_ids)
  
  
  # ## Create the model
  
  # ### Load the room geometry data
  
  # In[12]:
  
  
  room_geometry_data = np.load(
      os.path.join(
          io_directory,
          run_identifier,
          'room_geometry_data.npz'))
  
  
  # In[13]:
  
  
  fixed_sensor_positions = room_geometry_data['fixed_sensor_positions']
  
  
  # In[14]:
  
  
  fixed_sensor_positions
  
  
  # In[15]:
  
  
  room_corners = room_geometry_data['room_corners']
  # In[16]:
  
  
  room_corners
  
  
  # ### Load the model parameters
  
  # In[17]:
  
  
  with open(
      os.path.join(
          io_directory,
          run_identifier,
          'model_parameters.pkl'),
      'r+b') as file_handle:
      model_parameters = pickle.load(
          file_handle)
  initial_status_on_probability = model_parameters['initial_status_on_probability']
  status_on_to_off_probability = model_parameters['status_on_to_off_probability']
  status_off_to_on_probability = model_parameters['status_off_to_on_probability']
  ping_success_probability_sensor_on = model_parameters['ping_success_probability_sensor_on']
  ping_success_probability_sensor_off = model_parameters['ping_success_probability_sensor_off']
  moving_sensor_drift_reference = model_parameters['moving_sensor_drift_reference']
  reference_time_delta = model_parameters['reference_time_delta']
  ping_success_probability_zero_distance = model_parameters['ping_success_probability_zero_distance']
  receive_probability_reference_distance = model_parameters['receive_probability_reference_distance']
  reference_distance = model_parameters['reference_distance']
  rssi_untruncated_mean_intercept = model_parameters['rssi_untruncated_mean_intercept']
  rssi_untruncated_mean_slope = model_parameters['rssi_untruncated_mean_slope']
  rssi_untruncated_std_dev = model_parameters['rssi_untruncated_std_dev']
  lower_rssi_cutoff = model_parameters['lower_rssi_cutoff']
  
  
  # ### Initialize the model object
  
  # Create a instance of the `SensorModel` class. This class (which is a child of the more general `SMCModel` class) provides functions which allow us to perform inference, generated simulated data, etc. It assumes particular functional forms for the initial state model, the state transition model, and the sensor response model. It does take arguments for various parameters of these models (see the code for the class), but below we use the defaults.
  
  # Set the number of particles.
  
  # In[18]:
  
    
  num_particles = npart
  
  
  # In[19]:
  
  
  sensor_model = SensorModel(
      variable_structure,
      room_corners,
      fixed_sensor_positions,
      num_particles,
      initial_status_on_probability = initial_status_on_probability,
      status_on_to_off_probability = status_on_to_off_probability,
      status_off_to_on_probability = status_off_to_on_probability,
      ping_success_probability_sensor_on =  ping_success_probability_sensor_on,
      ping_success_probability_sensor_off = ping_success_probability_sensor_off,
      moving_sensor_drift_reference = moving_sensor_drift_reference,
      reference_time_delta = reference_time_delta,
      ping_success_probability_zero_distance = ping_success_probability_zero_distance,
      receive_probability_reference_distance = receive_probability_reference_distance,
      reference_distance = reference_distance,
      rssi_untruncated_mean_intercept = rssi_untruncated_mean_intercept,
      rssi_untruncated_mean_slope = rssi_untruncated_mean_slope,
      rssi_untruncated_std_dev = rssi_untruncated_std_dev,
      lower_rssi_cutoff = lower_rssi_cutoff)
  
  del room_geometry_data
  del room_corners
  del initial_status_on_probability
  del y_data

  gc.collect()
  # ## Sample the posterior distribution $f(\mathbf{X}_t | \mathbf{Y}_0, \ldots, \mathbf{Y}_t)$ of the real data using a particle filter
  
  # Using the helper functions provided by the `SensorModel` class, generate samples of the posterior distribution (i.e., the distribution of $\mathbf{X}$ values given the observed $\mathbf{Y}$ values). Below, we separately generate the samples for each time step to mimic the real-time use case. There is also a helper function called `generate_particle_trajectory()` which generates an entire trajectory of $\mathbf{X}$ samples given an entire trajectory of $\mathbf{Y}$ values (not shown here).
  
  # In[20]:
  
timestamps_tmp = timestamps_data['timestamps']

num_steps_total = len(timestamps_tmp)

for step_start in range(0,num_steps_total,num_steps):

  timestamps = timestamps_tmp[step_start:step_start+num_steps]

  #del timestamps_tmp
  #del timestamps_data

  gc.collect()

  num_timesteps = len(timestamps)
  print("Total of ", num_timesteps, "available...")
  num_timesteps = num_steps
  print("Using ", num_timesteps)


  
  t_test = timestamps
  num_timesteps_test = len(t_test)
  #num_timesteps_test = num_steps
  print("Using ", num_timesteps_test)
  
  # Initialize the variables.
  
  # In[21]:
  
  
  x_discrete_particles_trajectory = np.zeros(
      (num_timesteps_test, num_particles, variable_structure.num_x_discrete_vars),
      dtype = 'int')
  x_continuous_particles_trajectory = np.zeros(
      (num_timesteps_test, num_particles, variable_structure.num_x_continuous_vars),
      dtype = 'float')
  log_weights_trajectory = np.zeros(
      (num_timesteps_test, num_particles),
      dtype = 'float')
  ancestor_indices_trajectory = np.zeros(
      (num_timesteps_test, num_particles),
      dtype = 'int')
  
  
  # Generate the particles for the initial state $\mathbf{X}_0$.
  
  # In[22]:
  
  x_discrete_particles_trajectory[0], x_continuous_particles_trajectory[0], log_weights_trajectory[0] = sensor_model.generate_initial_particles(
      y_discrete_t[0],
      y_continuous_t[0])
  
  
  # Generate the particles for all later times.
  
  # In[23]:
  
  x_disc = x_discrete_particles_trajectory
  x_cont = x_continuous_particles_trajectory
  log_weights = log_weights_trajectory
  anc_indx = ancestor_indices_trajectory
  
  #weights=np.exp(log_weights)
  #normlw=np.sum(weights)
  
  #ancestor_indices = np.int32(np.random.choice(
  #                num_particles,
  #                size=num_particles,
  #                p=weights/normlw))
  
  
  log_weight_threshold = np.log(1/num_particles)
  start_time = time.time()
  # Main thread: create a coordinator.
  #coord = tf.train.Coordinator()
  # Create threads that run 'MyLoop()'
  num_threads=6
  #threads = [threading.Thread(target=MyLoop, args=(coord,)) for i in range(0,num_threads)]
  
  def MyBlockUpdate(coord,num_threads,t_index_current):
      num_timesteps_test = t_index_current + num_threads
      t_index_th = t_index_current
      for t_index in range(t_index_th, num_timesteps_test, num_threads):
          #print(t_index,': {}: {}'.format(t_index, t_test[t_index]))
          #print(t_index,': Number of non-negligible previous weights: {}'.format(np.sum(log_weights_trajectory[t_index - 1] > log_weight_threshold)))
  
          #x_discrete_particles_trajectory[t_index], x_continuous_particles_trajectory[t_index], log_weights_trajectory[t_index], ancestor_indices_trajectory[t_index] = sensor_model.generate_next_particles(
          x_disc, x_cont, log_weights, anc_indx = sensor_model.generate_next_particles(
              x_discrete_particles_trajectory[t_index_th - 1],
              x_continuous_particles_trajectory[t_index_th - 1],
              log_weights_trajectory[t_index_th - 1],
              y_discrete_t[t_index_th],
              y_continuous_t[t_index_th],
              t_test[t_index] - t_test[t_index_th - 1])
  
          x_discrete_particles_trajectory[t_index_th] = x_disc
          x_continuous_particles_trajectory[t_index_th] = x_cont
          log_weights_trajectory[t_index_th] = log_weights
          ancestor_indices_trajectory[t_index_th] = anc_indx
      #coord.join(threads)
  
  
  
  if 0:
    for t_index in range(1, num_timesteps_test, num_threads):
      t_start = time.time()
      threads = [threading.Thread(target=MyBlockUpdate, args=(coord,num_threads,t_index)) for i in range(0,num_threads)]    
      for t in threads:
          t.start()
          #print('t: ', t, "/",num_threads)
          #print(t,': {}: {}'.format(t_index, t_test[t_index]))
          #print(t,': Number of non-negligible previous weights: {}'.format(np.sum(log_weights_trajectory[t_index - 1] > log_weight_threshold)))
      
          #x_discrete_particles_trajectory[t_index], x_continuous_particles_trajectory[t_index], log_weights_trajectory[t_index], ancestor_indices_trajectory[t_index] = sensor_model.generate_next_particles(
          #x_disc, x_cont, log_weights, anc_indx = sensor_model.generate_next_particles(
          #    x_discrete_particles_trajectory[t_index - 1],
          #    x_continuous_particles_trajectory[t_index - 1],
          #    log_weights_trajectory[t_index - 1],
          #    y_discrete_t[t_index],
          #    y_continuous_t[t_index],
          #    t_test[t_index] - t_test[t_index - 1])
  
          #x_discrete_particles_trajectory[t_index] = x_disc
          #x_continuous_particles_trajectory[t_index] = x_cont
          #log_weights_trajectory[t_index] = log_weights
          #ancestor_indices_trajectory[t_index] = anc_indx
      coord.join(threads)
      t_end = time.time()
      print('MT: Number of particles after resampling: {}'.format(len(np.unique(ancestor_indices_trajectory[t_index]))))
      part_per_sec = num_particles/(t_end-t_start)
  
      print('MT: Time per step: ', t_end-t_start)
      print('MT: Particles/sec: ', part_per_sec)
    end_time = time.time()
    tot_time = end_time-start_time
    print('MT: Finished TEST loop over ',num_particles, 'num_particles in ', tot_time, 'seconds...')
  else:
  
    for t_index in range(1, num_timesteps_test):
      print('{}: {}'.format(t_index, t_test[t_index]))
      print('Number of non-negligible previous weights: {}'.format(np.sum(log_weights_trajectory[t_index - 1] > log_weight_threshold)))
      t_start = time.time()
      #x_discrete_particles_trajectory[t_index], x_continuous_particles_trajectory[t_index], log_weights_trajectory[t_index], ancestor_indices_trajectory[t_index] = sensor_model.generate_next_particles(
  
      #x_discrete_particles_trajectory[t_index-1] = x_disc[t_index-1]
      #x_continuous_particles_trajectory[t_index-1] = x_cont[t_index-1]
      #log_weights_trajectory[t_index-1] = log_weights[t_index-1]
      #ancestor_indices_trajectory[t_index-1] = anc_indx[t_index-1]
  
      x_disc[t_index], x_cont[t_index], log_weights[t_index], anc_indx[t_index] = sensor_model.generate_next_particles(
              x_discrete_particles_trajectory[t_index-1],
              x_continuous_particles_trajectory[t_index - 1],
              log_weights_trajectory[t_index - 1],
              y_discrete_t[t_index],
              y_continuous_t[t_index],
              t_test[t_index] - t_test[t_index-1], ancestor_indices_trajectory[t_index-1])
  
      x_discrete_particles_trajectory[t_index] = x_disc[t_index]
      x_continuous_particles_trajectory[t_index] = x_cont[t_index]
      log_weights_trajectory[t_index] = log_weights[t_index]
      ancestor_indices_trajectory[t_index] = anc_indx[t_index]
      
      t_end = time.time()
      print('Number of particles after resampling: {}'.format(len(np.unique(ancestor_indices_trajectory[t_index]))))
      part_per_sec = num_particles/(t_end-t_start)
      print('Particles/sec: ', part_per_sec)
    end_time = time.time()
    tot_time = end_time-start_time
    print('Finished TEST loop over ',num_particles, 'num_particles in ', tot_time, 'seconds...')
  
  # In[24]:
  # In[24]:

  #threads = [threading.Thread(target=MyBlockUpdate, args=(coord,num_threads,)) for i in range(0,num_threads)]
  #coord.join(threads)
  
  num_nonzero_weights = np.sum(np.isfinite(log_weights_trajectory), axis=-1)
  
  
  # In[25]:
  
  
  num_nonnegligible_weights = np.sum(log_weights_trajectory > log_weight_threshold, axis=-1)
  
  
  # In[26]:
  
  
  num_after_resampling = np.array(
      [len(np.unique(ancestor_indices_trajectory[t_index + 1])) for t_index in range(0, num_timesteps_test - 1)])
  
  
  # Calculate the sample means and sample standard deviations of the particles at each time step.
  
  # In[27]:
  
  
  x_discrete_mean_particle = np.average(
      x_discrete_particles_trajectory, 
      axis=1, 
      weights=np.repeat(np.exp(log_weights_trajectory), variable_structure.num_x_discrete_vars).reshape(
          (num_timesteps_test,
           num_particles,
           variable_structure.num_x_discrete_vars)))
  
  
  # In[28]:
  
  
  x_continuous_mean_particle = np.average(
      x_continuous_particles_trajectory, 
      axis=1, 
      weights=np.repeat(np.exp(log_weights_trajectory), variable_structure.num_x_continuous_vars).reshape(
          (num_timesteps_test,
           num_particles,
           variable_structure.num_x_continuous_vars)))
  
  
  # In[29]:
  
  
  x_continuous_squared_mean_particle = np.average(
      np.square(x_continuous_particles_trajectory), 
      axis=1,
      weights=np.repeat(np.exp(log_weights_trajectory), variable_structure.num_x_continuous_vars).reshape(
          (num_timesteps_test,
           num_particles,
           variable_structure.num_x_continuous_vars)))
  
  
  # In[30]:
  
  
  x_continuous_sd_particle = np.sqrt(np.abs(x_continuous_squared_mean_particle - np.square(x_continuous_mean_particle)))
  
  
  # In[31]:
  
  np.savez(
      os.path.join(
          io_directory,
          run_identifier,
          'x_samples_summaries'),
      num_nonzero_weights = num_nonzero_weights,
      num_nonnegligible_weights = num_nonnegligible_weights,
      num_after_resampling = num_after_resampling,
      x_discrete_mean_particle = x_discrete_mean_particle,
      x_continuous_mean_particle = x_continuous_mean_particle,
      x_continuous_sd_particle = x_continuous_sd_particle)
  

# In[ ]:




