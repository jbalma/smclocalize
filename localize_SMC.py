# What header information do I need here?

import numpy as np
from scipy import special
from scipy import stats

# Define a class for a generic sequential Monte Carlo (AKA state space) model
class SMCModel(object):

    # We need to supply this object with functions representing the various
    # conditional probability distributions which comprise the model as well as
    # the structure of the X and Y variables
    def __init__(
        self,
        x_initial_sample, x_bar_x_prev_sample, y_bar_x_sample, y_bar_x_log_pdf,
        num_x_discrete_vars, num_x_continuous_vars,
        num_y_discrete_vars, num_y_continuous_vars
    ):
        # Need to check dimensions and types of all arguments
        self.x_initial_sample = x_initial_sample
        self.x_bar_x_prev_sample = x_bar_x_prev_sample
        self.y_bar_x_sample = y_bar_x_sample
        self.y_bar_x_log_pdf = y_bar_x_log_pdf
        self.num_x_discrete_vars = num_x_discrete_vars
        self.num_x_continuous_vars = num_x_continuous_vars
        self.num_y_discrete_vars = num_y_discrete_vars
        self.num_y_continuous_vars = num_y_continuous_vars
        # Would it make sense to define X and Y objects which combine the
        # discrete and continuous portions of each type of variable?

    # Define a function which generates an initial set of X particles along with
    # their weights
    def generate_initial_particles(self, num_particles = 1000):
        x_discrete_particles_initial, x_continuous_particles_initial = self.x_initial_sample(num_particles)
        log_weights_initial = np.repeat(np.log(1.0/num_particles), num_particles)
        return x_discrete_particles_initial, x_continuous_particles_initial, log_weights_initial

    # Define a function which takes a set of particles and weights from the
    # previous time step and a Y value from the current time step and returns a set
    # of particles and weights for the current timestep along with a list of
    # their ancestor particles
    def generate_next_particles(
        self,
        x_discrete_particles_previous, x_continuous_particles_previous,
        log_weights_previous,
        y_discrete, y_continuous
    ):
        # Need to check dimensions and types of all arguments

        # Infer the number of particles from the dimensions of X_previous
        num_particles = x_discrete_particles_previous.shape[0]
        # Choose an ancestor for each new particle based on the previous weights
        ancestors = np.random.choice(
            num_particles,
            size=num_particles,
            p=np.exp(log_weights_previous)
        )
        # Generate the new particles using the state transition function
        x_discrete_particles, x_continuous_particles = self.x_bar_x_prev_sample(
            x_discrete_particles_previous[ancestors],
            x_continuous_particles_previous[ancestors]
        )
        # Assign weights to the new particles using the observation function
        log_weights = self.y_bar_x_log_pdf(
            x_discrete_particles,
            x_continuous_particles,
            np.tile(y_discrete, (num_particles, 1)),
            np.tile(y_continuous, (num_particles, 1))
        )
        # Normalize the weights
        log_weights = log_weights - special.logsumexp(log_weights)
        return x_discrete_particles, x_continuous_particles, log_weights, ancestors

    # Define a function with takes an entire trajectory of Y values and returns
    # an entire trajectory of X particles along with their weights and ancestors
    def generate_particle_trajectory(
        self,
        y_discrete_trajectory, y_continuous_trajectory,
        num_particles = 1000
    ):
        # Need to check dimensions and types of all arguments

        # Infer the number of timesteps from the dimensions of the Y trajectory
        num_timesteps = y_discrete_trajectory.shape[0]
        # Initialize all of the outputs
        x_discrete_particles_trajectory = np.zeros(
            (num_timesteps, num_particles, num_x_discrete_vars),
            dtype='int'
        )
        x_continuous_particles_trajectory = np.zeros(
            (num_timesteps, num_particles, num_x_continous_vars),
            dtype='float'
        )
        log_weights_trajectory = np.zeros(
            (num_timesteps, num_particles),
            dtype='float'
        )
        ancestors_trajectory = np.zeros(
            (num_timesteps, num_particles),
            dtype='int'
        )
        # Generate an initial set of X particles
        x_discrete_particles_initial, x_continuous_particles_initial, log_weights_initial = self.generate_initial_particles(num_particles)
        # Generate the first step in the X particle trajectory based on the
        # first Y value
        x_discrete_particles_trajectory[0], x_continuous_particles_trajectory[0], log_weights_trajectory[0], ancestors_trajectory[0] = self.generate_next_particles(
            x_discrete_particles_initial,
            x_continuous_particles_initial,
            log_weights_initial,
            y_discrete_trajectory[0],
            y_continuous_trajectory[0]
        )
        # Generate the rest of the X particle trajectory by stepping through the
        # rest of the Y values
        for i in range(1, num_timesteps):
            x_discrete_particles_trajectory[i], x_continuous_particles_trajectory[i], log_weights_trajectory[i], ancestors_trajectory[i] = self.generate_next_particles(
                x_discrete_particles_trajectory[i - 1],
                x_continuous_particles_trajectory[i - 1],
                log_weights_trajectory[i - 1],
                y_discrete_trajectory[i],
                y_continuous_trajectory[i]
            )
        return x_discrete_particles_trajectory, x_continuous_particles_trajectory, log_weights_trajectory, ancestors_trajectory


# Define a class based on the generic sequential Monte Carlo model class which
# represents an instance of our particular sensor model
class SensorModel(SMCModel):

    # We need to supply this object with the classroom configuration (number of
    # each kind of sensor, room dimensions, positions of area sensors), the
    # distance that we expect sensors to move from timestep to timestep, and
    # the probability distributions for the sensor response as functions of
    # inter-sensor distance: a ping success probability function, a function
    # which generates RSSI samples, and an RSSI probability density function
    def __init__(
        self,
        sensor_variable_structure,
        room_corners, fixed_sensor_positions,
        moving_sensor_drift,
        ping_success_probability_function,
        rssi_samples_function, rssi_log_pdf_function
    ):
        # Need to check dimensions and types of all arguments
        self.sensor_variable_structure = sensor_variable_structure
        self.room_corners = room_corners
        self.moving_sensor_drift = moving_sensor_drift
        self.fixed_sensor_positions = fixed_sensor_positions
        self.ping_success_probability_function = ping_success_probability_function
        self.rssi_samples_function = rssi_samples_function
        self.rssi_log_pdf_function = rssi_log_pdf_function

        self.num_child_sensors = self.sensor_variable_structure.num_child_sensors
        self.num_material_sensors = self.sensor_variable_structure.num_material_sensors
        self.num_teacher_sensors = self.sensor_variable_structure.num_teacher_sensors
        self.num_area_sensors = self.sensor_variable_structure.num_area_sensors
        self.num_dimensions = self.sensor_variable_structure.num_dimensions
        self.num_x_discrete_vars = self.sensor_variable_structure.num_x_discrete_vars
        self.num_x_continuous_vars = self.sensor_variable_structure.num_x_continuous_vars
        self.num_y_discrete_vars = self.sensor_variable_structure.num_y_discrete_vars
        self.num_y_continuous_vars = self.sensor_variable_structure.num_y_continuous_vars
        self.num_moving_sensors = self.sensor_variable_structure.num_moving_sensors
        self.num_fixed_sensors = self.sensor_variable_structure.num_fixed_sensors
        self.num_sensors = self.sensor_variable_structure.num_sensors

    # Define a function which generates samples of the initial X state
    def x_initial_sample(self, num_samples=1):
        x_discrete_initial_sample = np.tile(np.array([]), (num_samples, 1))
        x_continuous_initial_sample = np.squeeze(
            self.sensor_variable_structure.extract_x_variables(
                np.random.uniform(
                    low = np.tile(self.room_corners[0], (self.num_sensors, 1)),
                    high = np.tile(self.room_corners[1], (self.num_sensors, 1)),
                    size = (num_samples, self.num_sensors, self.num_dimensions)
                )
            )
        )
        return x_discrete_initial_sample, x_continuous_initial_sample

    # Define a function which generates a sample of the current X state given the
    # previous X state
    def x_bar_x_prev_sample(self, x_discrete_prev, x_continuous_prev):
        x_discrete_bar_x_prev_sample = np.array([])
        x_continuous_bar_x_prev_sample = stats.truncnorm.rvs(
            a=(np.tile(self.room_corners[0], self.num_moving_sensors) - x_continuous_prev)/self.moving_sensor_drift,
            b=(np.tile(self.room_corners[1], self.num_moving_sensors) - x_continuous_prev)/self.moving_sensor_drift,
            loc=x_continuous_prev,
            scale=self.moving_sensor_drift
        )
        return x_discrete_bar_x_prev_sample, x_continuous_bar_x_prev_sample

    # Define a function which takes an X value and returns an array representing
    # the positions of all sensors (including the fixed sensors)
    def sensor_positions(self, x_continuous):
        return np.concatenate(
            (x_continuous.reshape(x_continuous.shape[:-1] + (self.num_moving_sensors, self.num_dimensions)),
            np.broadcast_to(self.fixed_sensor_positions, x_continuous.shape[:-1] + self.fixed_sensor_positions.shape)),
            axis=-2
        )

    # Define a function which takes a array representing the positions of all
    # sensors and returns a vector of inter-sensor distances corresponding to
    # the Y variables
    def distances(self, sensor_positions):
        return self.sensor_variable_structure.extract_y_variables(
            np.linalg.norm(
                np.subtract(
                    sensor_positions[...,np.newaxis,:,:],
                    sensor_positions[...,:,np.newaxis,:]
                ),
                axis = -1
            )
        )

    # Define a function which takes a vector of inter-sensor distances
    # corresponding to the Y variables and returns a corresponding array of
    # ping success probabilities
    def ping_success_probabilities_array(self, distances):
        probabilities = self.ping_success_probability_function(distances)
        return np.stack((probabilities, 1 - probabilities), axis=-1)

    # Define a function which takes a vector of inter-sensor distances
    # corresponding to the Y variables and returns a vector of ping success
    # samples
    def ping_success_samples(self, distances):
        return np.apply_along_axis(
            lambda p_array: np.random.choice(len(p_array), p=p_array),
            axis=-1,
            arr=self.ping_success_probabilities_array(distances)
        )

    # Define a function which takes an X value and returns a sample of the
    # discrete Y variables
    def y_discrete_bar_x_sample(self, x_discrete, x_continuous):
        return self.ping_success_samples(self.distances(self.sensor_positions(x_continuous)))

    # Define a function which takes an X value and returns a sample of the
    # continuous Y variables
    def y_continuous_bar_x_sample(self, x_discrete, x_continuous):
        return self.rssi_samples_function(self.distances(self.sensor_positions(x_continuous)))

    # Define a function which combines the above to take an X value and return a
    # Y sample
    def y_bar_x_sample(self, x_discrete, x_continuous):
        return self.y_discrete_bar_x_sample(x_discrete, x_continuous), self.y_continuous_bar_x_sample(x_discrete, x_continuous)

    # Define a function which takes an X value and a Y value and returns the
    # probability density of that Y value given that X value
    def y_bar_x_log_pdf(self, x_discrete, x_continuous, y_discrete, y_continuous):
        distances_x = self.distances(self.sensor_positions(x_continuous))
        ping_success_probabilities_array_x = self.ping_success_probabilities_array(distances_x)
        discrete_log_probabilities = np.log(
            np.choose(
                y_discrete,
                np.rollaxis(
                    ping_success_probabilities_array_x,
                    axis=-1
                )
            )
        )
        continuous_log_probability_densities = self.rssi_log_pdf_function(
            y_continuous,
            distances_x
        )
        continuous_log_probability_densities[y_discrete == 1] = 0.0
        return np.sum(discrete_log_probabilities, axis=-1) + np.sum(continuous_log_probability_densities, axis=-1)

# Define a class which defines an X and Y variable structure based on the number
# of each type of sensor
class SensorVariableStructure(object):

    # We only need to provide this object with the number of each type of sensor
    def __init__(
        self,
        child_entity_ids, material_entity_ids, teacher_entity_ids, area_entity_ids,
        num_dimensions = 2
    ):
        # Need to check dimensions and types of all arguments
        self.child_entity_ids = child_entity_ids
        self.material_entity_ids = material_entity_ids
        self.teacher_entity_ids = teacher_entity_ids
        self.area_entity_ids = area_entity_ids
        self.num_dimensions = num_dimensions

        self.num_child_sensors = len(child_entity_ids)
        self.num_material_sensors = len(material_entity_ids)
        self.num_teacher_sensors = len(teacher_entity_ids)
        self.num_area_sensors = len(area_entity_ids)

        self.num_moving_sensors = self.num_child_sensors + self.num_material_sensors + self.num_teacher_sensors
        self.num_fixed_sensors = self.num_area_sensors
        self.num_sensors = self.num_moving_sensors + self.num_fixed_sensors

        # Define a Boolean mask which helps us extract and flatten X values from
        # an array representing the positions of all sensors

        # Start with an array that has a row for every sensor and a column for every spatial dimension
        self.extract_x_variables_mask = np.full((self.num_sensors, self.num_dimensions), True)
        # We don't track the positions of fixed sensors
        self.extract_x_variables_mask[self.num_moving_sensors:,:] = False

        # Define the number of discrete and continuous x variables using this mask
        self.num_x_discrete_vars = 0
        self.num_x_continuous_vars = np.sum(self.extract_x_variables_mask)

        # Define a Boolean mask which help us extract and flatten Y values from
        # an array representing every pairwise combination of sensors

        # Start with an array that has every pairwise combination of sensors
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

        # Define the number of discrete and continuous Y variables using this mask
        self.num_y_discrete_vars = np.sum(self.extract_y_variables_mask)
        self.num_y_continuous_vars = np.sum(self.extract_y_variables_mask)

    # Define a function which uses the Boolean mask defined above to extract and
    # flatten X values from a larger data structure
    def extract_x_variables(self, a):
        return a[..., self.extract_x_variables_mask]

    # Define a function which uses the Boolean mask defined above to extract and
    # flatten Y values from a larger data structure
    def extract_y_variables(self, a):
        return a[..., self.extract_y_variables_mask]
