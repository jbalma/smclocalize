# What header information do I need here?

import numpy as np
from scipy import special

class SMCModel(object):

    def __init__(self, x_initial_sample, x_bar_x_prev_sample, y_bar_x_sample, y_bar_x_log_pdf, num_x_discrete_vars, num_x_continuous_vars, num_y_discrete_vars, num_y_continuous_vars):
        # Need to check dimensions and types of all arguments
        self.x_initial_sample = x_initial_sample
        self.x_bar_x_prev_sample = x_bar_x_prev_sample
        self.y_bar_x_sample = y_bar_x_sample
        self.y_bar_x_log_pdf = y_bar_x_log_pdf
        self.num_x_discrete_vars = num_x_discrete_vars
        self.num_x_continuous_vars = num_x_continuous_vars
        self.num_y_discrete_vars = num_y_discrete_vars
        self.num_y_continuous_vars = num_y_continuous_vars
        # Would it make sense to define x and y objects which combine the discrete and continuous portions of each type of variable?

    def generate_initial_particles(self, num_particles = 1000):
        # Need to check dimensions and types of all arguments
        x_discrete_particles_initial, x_continuous_particles_initial = self.x_initial_sample(num_particles)
        log_weights_initial = np.repeat(np.log(1.0/num_particles), num_particles)
        return x_discrete_particles_initial, x_continuous_particles_initial, log_weights_initial

    def generate_next_particles(self, x_discrete_particles_current, x_continuous_particles_current, log_weights_current, y_discrete_next, y_continuous_next):
        # Need to check dimensions and types of all arguments
        num_particles = x_discrete_particles_current.shape[0]
        ancestors_next = np.random.choice(num_particles, size=num_particles, p=np.exp(log_weights_current))
        x_discrete_particles_next, x_continuous_particles_next = self.x_bar_x_prev_sample(x_discrete_particles_current[ancestors_next], x_continuous_particles_current[ancestors_next])
        log_weights_next = self.y_bar_x_log_pdf(x_discrete_particles_next, x_continuous_particles_next, np.tile(y_discrete_next, (num_particles, 1)), np.tile(y_continuous_next, (num_particles, 1)))
        log_weights_next = log_weights_next - special.logsumexp(log_weights_next)
        return x_discrete_particles_next, x_continuous_particles_next, log_weights_next, ancestors_next

    def generate_particle_trajectory(self, y_discrete_trajectory, y_continuous_trajectory, num_particles = 1000):
        # Need to check dimensions and types of all arguments
        num_timesteps = y_discrete_trajectory.shape[0]
        x_discrete_particles_trajectory = np.zeros((num_timesteps, num_particles, num_x_discrete_vars), dtype='int')
        x_continuous_particles_trajectory = np.zeros((num_timesteps, num_particles, num_x_continous_vars), dtype='float')
        log_weights_trajectory = np.zeros((num_timesteps, num_particles), dtype='float')
        ancestors_trajectory = np.zeros((num_timesteps, num_particles), dtype='int')
        x_discrete_particles_initial, x_continuous_particles_initial, log_weights_initial = self.generate_initial_particles(num_particles)
        x_discrete_particles_trajectory[0], x_continuous_particles_trajectory[0], log_weights_trajectory[0], ancestors_trajectory[0] = self.generate_next_particles(x_discrete_particles_initial, x_continuous_particles_initial, log_weights_initial, y_discrete_trajectory[0],  y_continuous_trajectory[0])
        for i in range(1, num_timesteps):
            x_discrete_particles_trajectory[i], x_continuous_particles_trajectory[i], log_weights_trajectory[i], ancestors_trajectory[i] = self.generate_next_particles(x_discrete_particles_trajectory[i - 1], x_continuous_particles_trajectory[i - 1], log_weights_trajectory[i - 1], y_discrete_trajectory[i], y_continuous_trajectory[i])
        return x_discrete_particles_trajectory, x_continuous_particles_trajectory, log_weights_trajectory, ancestors_trajectory
