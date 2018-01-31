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


class SensorModel(SMCModel):

    def __init__(
        self,
        num_child_sensors, num_material_sensors, num_teacher_sensors, num_area_sensors,
        num_dimensions, room_corners,
        x_bar_x_prev_sample, y_bar_x_sample, y_bar_x_log_pdf
    ):
        # Need to check dimensions and types of all arguments
        self.num_child_sensors = num_child_sensors
        self.num_material_sensors = num_material_sensors
        self.num_teacher_sensors = num_teacher_sensors
        self.num_area_sensors = num_area_sensors
        self.num_dimensions = num_dimensions
        self.room_corners = room_corners
        self.x_bar_x_prev_sample = x_bar_x_prev_sample
        self.y_bar_x_sample = y_bar_x_sample
        self.y_bar_x_log_pdf = y_bar_x_log_pdf

        self.num_moving_sensors = self.num_child_sensors + self.num_material_sensors + self.num_teacher_sensors
        self.num_fixed_sensors = self.num_area_sensors
        self.num_sensors = self.num_moving_sensors + self.num_fixed_sensors

        # Define a boolean mask which extracts and flattens x variables from a larger data structure
        # Start with a matrix that has a row for every sensor and a column for every spatial dimension
        self.extract_x_variables_mask = np.full((self.num_sensors, self.num_dimensions), True)
        # We don't track the positions of fixed sensors
        self.extract_x_variables_mask[self.num_moving_sensors:,:] = False

        # Define the number of discrete and continuous x variables using this mask
        self.num_x_discrete_vars = 0
        self.num_x_continous_vars = np.sum(self.extract_x_variables_mask)

        # Define a boolean mask which extracts and flattens our y variables from a larger data structure
        # Start with matrix that has every pairwise combination of sensors
        self.extract_y_variables_mask = np.full((self.num_sensors, self.num_sensors), True)
        # Sensors don't send pings to themselves
        np.fill_diagonal(self.extract_y_variables_mask, False)
        # We don't store pings from material sensors to other material sensors
        self.extract_y_variables_mask[
            self.num_child_sensors:(self.num_child_sensors + self.num_material_sensors),
            self.num_child_sensors:(self.num_child_sensors + self.num_material_sensors)
        ] = False
        # We don't store pings from teacher sensors to other teacher sensors
        self.extract_y_variables_mask[
            (self.num_child_sensors + self.num_material_sensors):self.num_moving_sensors,
            (self.num_child_sensors + self.num_material_sensors):self.num_moving_sensors
        ] = False
        # We don't store pings from area sensors to other area sensors
        self.extract_y_variables_mask[
            self.num_moving_sensors:,
            self.num_moving_sensors:
        ] = False
        # We don't store pings from material sensors to area sensors (and vice versa)
        self.extract_y_variables_mask[
            self.num_child_sensors:(self.num_child_sensors + self.num_material_sensors),
            self.num_moving_sensors:
        ] = False
        self.extract_y_variables_mask[
            self.num_moving_sensors:,
            self.num_child_sensors:(self.num_child_sensors + self.num_material_sensors)
        ] = False

        # Define the number of discrete and continuous y variables using this mask
        self.num_y_discrete_vars = np.sum(self.extract_y_variables_mask)
        self.num_y_continous_vars = np.sum(self.extract_y_variables_mask)

    # Define a function which uses the boolean mask defined above to extract and flatten x values from a larger data structure
    def extract_x_variables(self, a):
        return a[..., self.extract_x_variables_mask]

    # Define a function which uses the boolean mask defined above to extract and flatten y values from a larger data structure
    def extract_y_variables(a):
        return a[..., self.extract_y_variables_mask]

    # Define a function which generates samples of the initial x state
    def x_initial_sample(self, num_samples=1):
        x_discrete_initial_sample = np.tile(np.array([]), (num_samples, 1))
        x_continuous_initial_sample = np.squeeze(
            self.extract_x_variables(
                np.random.uniform(
                    low = np.tile(self.room_corners[0], (self.num_sensors, 1)),
                    high = np.tile(self.room_corners[1], (self.num_sensors, 1)),
                    size = (num_samples, self.num_sensors, self.num_dimensions)
                )
            )
        )
        return x_discrete_initial_sample, x_continuous_initial_sample
