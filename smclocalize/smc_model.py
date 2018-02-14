import numpy as np
from scipy import special
import time

# Define a class for a generic sequential Monte Carlo (AKA state space) model
class SMCModel(object):
    # We need to supply this object with functions representing the various
    # conditional probability distributions which comprise the model as well as
    # the number of discrete and continuous X and Y variables.
    def __init__(
        self,
        x_initial_sample,
        x_bar_x_previous_sample,
        y_bar_x_sample,
        y_bar_x_log_pdf,
        num_x_discrete_vars,
        num_x_continuous_vars,
        num_y_discrete_vars,
        num_y_continuous_vars):
        self.x_initial_sample = x_initial_sample
        self.x_bar_x_previous_sample = x_bar_x_previous_sample
        self.y_bar_x_sample = y_bar_x_sample
        self.y_bar_x_log_pdf = y_bar_x_log_pdf
        self.num_x_discrete_vars = num_x_discrete_vars
        self.num_x_continuous_vars = num_x_continuous_vars
        self.num_y_discrete_vars = num_y_discrete_vars
        self.num_y_continuous_vars = num_y_continuous_vars

    # Generates an initial set of X particles (samples) along with
    # their weights.
    def generate_initial_particles(
        self,
        y_discrete_initial,
        y_continuous_initial,
        num_particles = 1000):
        x_discrete_particles_initial, x_continuous_particles_initial = self.x_initial_sample(num_particles)
        log_weights_initial = self.y_bar_x_log_pdf(
            x_discrete_particles_initial,
            x_continuous_particles_initial,
            np.tile(y_discrete_initial, (num_particles, 1)),
            np.tile(y_continuous_initial, (num_particles, 1)))
        log_weights_initial = log_weights_initial - special.logsumexp(log_weights_initial)
        return x_discrete_particles_initial, x_continuous_particles_initial, log_weights_initial

    # Generate a set of X particles and weights (along with pointers to the
    # particles' ancestors) for the current time step based on the set of
    # particles and weights from the previous time step and a Y value from the
    # current time step.
    def generate_next_particles(
        self,
        x_discrete_particles_previous, x_continuous_particles_previous,
        log_weights_previous,
        y_discrete, y_continuous,
        t_delta):
        num_particles = x_discrete_particles_previous.shape[0]
        if __debug__:
            start = time.clock()
        ancestors = np.random.choice(
            num_particles,
            size = num_particles,
            p = np.exp(log_weights_previous))
        if __debug__:
            after_ancestors = time.clock()
        x_discrete_particles, x_continuous_particles = self.x_bar_x_previous_sample(
            x_discrete_particles_previous[ancestors],
            x_continuous_particles_previous[ancestors],
            t_delta)
        if __debug__:
            after_transition = time.clock()
        log_weights = self.y_bar_x_log_pdf(
            x_discrete_particles,
            x_continuous_particles,
            np.tile(y_discrete, (num_particles, 1)),
            np.tile(y_continuous, (num_particles, 1)))
        if __debug__:
            after_weights = time.clock()
        log_weights = log_weights - special.logsumexp(log_weights)
        if __debug__:
            after_renormalize = time.clock()
        if __debug__:
            print ('[generate_next_particles] Anc: {:.1e} Trans: {:.1e} Wts: {:.1e} Renorm: {:.1e}'.format(
                after_ancestors - start,
                after_transition - after_ancestors,
                after_weights - after_transition,
                after_renormalize - after_weights))
        return x_discrete_particles, x_continuous_particles, log_weights, ancestors

    # Generate an entire trajectory of X particles along with their weights and
    # ancestors based on an entire trajectory of Y values.
    def generate_particle_trajectory(
        self,
        variable_structure,
        t_trajectory,
        y_discrete_trajectory,
        y_continuous_trajectory,
        num_particles = 1000):
        num_timesteps = len(t_trajectory)
        x_discrete_particles_trajectory = np.zeros(
            (num_timesteps, num_particles, self.num_x_discrete_vars),
            dtype='int')
        x_continuous_particles_trajectory = np.zeros(
            (num_timesteps, num_particles, self.num_x_continuous_vars),
            dtype='float')
        log_weights_trajectory = np.zeros(
            (num_timesteps, num_particles),
            dtype='float')
        ancestors_trajectory = np.zeros(
            (num_timesteps, num_particles),
            dtype='int')
        x_discrete_particles_trajectory[0], x_continuous_particles_trajectory[0], log_weights_trajectory[0] = self.generate_initial_particles(
            y_discrete_trajectory[0],
            y_continuous_trajectory[0],
            num_particles)
        for i in range(1, num_timesteps):
            x_discrete_particles_trajectory[i], x_continuous_particles_trajectory[i], log_weights_trajectory[i], ancestors_trajectory[i] = self.generate_next_particles(
                x_discrete_particles_trajectory[i - 1],
                x_continuous_particles_trajectory[i - 1],
                log_weights_trajectory[i - 1],
                y_discrete_trajectory[i],
                y_continuous_trajectory[i],
                t_trajectory[i] - t_trajectory[i - 1])
        return x_discrete_particles_trajectory, x_continuous_particles_trajectory, log_weights_trajectory, ancestors_trajectory

    # Generate the initial time step of a simulated X and Y data set.
    def generate_initial_simulation_timestep(
        self):
        x_discrete_initial, x_continuous_initial = self.x_initial_sample()
        y_discrete_initial, y_continuous_initial = self.y_bar_x_sample(
            x_discrete_initial,
            x_continuous_initial)
        return x_discrete_initial, x_continuous_initial, y_discrete_initial, y_continuous_initial

    # Generate a single time step of simulated X and Y data based on the
    # previous time step and a time delta.
    def generate_next_simulation_timestep(
        self,
        x_discrete_previous,
        x_continuous_previous,
        t_delta):
        x_discrete, x_continuous = self.x_bar_x_previous_sample(
            x_discrete_previous,
            x_continuous_previous,
            t_delta)
        y_discrete, y_continuous = self.y_bar_x_sample(
            x_discrete_previous,
            x_continuous_previous)
        return x_discrete, x_continuous, y_discrete, y_continuous

    # Generate an entire trajectory of simulated X and Y data.
    def generate_simulation_trajectory(
        self,
        t_trajectory):
        num_timesteps_trajectory = len(t_trajectory)
        x_discrete_trajectory = np.zeros(
            (num_timesteps_trajectory, self.num_x_discrete_vars),
            dtype='int')
        x_continuous_trajectory = np.zeros(
            (num_timesteps_trajectory, self.num_x_continuous_vars),
            dtype='float')
        y_discrete_trajectory = np.zeros(
            (num_timesteps_trajectory, self.num_y_discrete_vars),
            dtype='int')
        y_continuous_trajectory = np.zeros(
            (num_timesteps_trajectory, self.num_y_continuous_vars),
            dtype='float')
        x_discrete_trajectory[0], x_continuous_trajectory[0], y_discrete_trajectory[0], y_continuous_trajectory[0] = self.generate_initial_simulation_timestep()
        for t_index in range(1, num_timesteps_trajectory):
            x_discrete_trajectory[t_index], x_continuous_trajectory[t_index], y_discrete_trajectory[t_index], y_continuous_trajectory[t_index] = self.generate_next_simulation_timestep(
                x_discrete_trajectory[t_index - 1],
                x_continuous_trajectory[t_index - 1],
                t_trajectory[t_index] - t_trajectory[t_index - 1])
        return x_discrete_trajectory, x_continuous_trajectory, y_discrete_trajectory, y_continuous_trajectory
