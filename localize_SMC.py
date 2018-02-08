# What header information do I need here?

import numpy as np
from scipy import special
from scipy import stats

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

        self.moving_entity_ids = child_entity_ids + material_entity_ids + teacher_entity_ids
        self.fixed_entity_ids = area_entity_ids
        self.entity_ids = self.moving_entity_ids + self.fixed_entity_ids

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
    def generate_initial_particles(
        self,
        y_discrete_initial,
        y_continuous_initial,
        num_particles = 1000,
        t = np.nan
    ):
        x_discrete_particles_initial, x_continuous_particles_initial = self.x_initial_sample(num_particles, t)
        # Assign weights to the new particles using the observation function
        log_weights_initial = self.y_bar_x_log_pdf(
            x_discrete_particles_initial,
            x_continuous_particles_initial,
            np.tile(y_discrete_initial, (num_particles, 1)),
            np.tile(y_continuous_initial, (num_particles, 1)),
            t
        )
        # Normalize the weights
        log_weights_initial = log_weights_initial - special.logsumexp(log_weights_initial)
        return x_discrete_particles_initial, x_continuous_particles_initial, log_weights_initial

    # Define a function which takes a set of particles and weights from the
    # previous time step and a Y value from the current time step and returns a set
    # of particles and weights for the current timestep along with a list of
    # their ancestor particles
    def generate_next_particles(
        self,
        x_discrete_particles_previous, x_continuous_particles_previous,
        log_weights_previous,
        y_discrete, y_continuous,
        t_previous = np.nan,
        t = np.nan
    ):
        # Need to check dimensions and types of all arguments

        # Infer the number of particles from the dimensions of X_previous
        num_particles = x_discrete_particles_previous.shape[0]
        # Choose an ancestor for each new particle based on the previous weights
        ancestors = np.random.choice(
            num_particles,
            size = num_particles,
            p = np.exp(log_weights_previous)
        )
        # Generate the new particles using the state transition function
        x_discrete_particles, x_continuous_particles = self.x_bar_x_prev_sample(
            x_discrete_particles_previous[ancestors],
            x_continuous_particles_previous[ancestors],
            t_previous,
            t
        )
        # Assign weights to the new particles using the observation function
        log_weights = self.y_bar_x_log_pdf(
            x_discrete_particles,
            x_continuous_particles,
            np.tile(y_discrete, (num_particles, 1)),
            np.tile(y_continuous, (num_particles, 1)),
            t
        )
        # Normalize the weights
        log_weights = log_weights - special.logsumexp(log_weights)
        return x_discrete_particles, x_continuous_particles, log_weights, ancestors

    # Define a function with takes an entire trajectory of Y values and returns
    # an entire trajectory of X particles along with their weights and ancestors
    def generate_particle_trajectory(
        self,
        y_discrete_trajectory, y_continuous_trajectory,
        num_particles = 1000,
        t_trajectory = np.nan,
    ):
        # Need to check dimensions and types of all arguments

        # Infer the number of timesteps from the dimensions of the Y trajectory
        num_timesteps = len(t_trajectory)
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
        x_discrete_particles_trajectory[0], x_continuous_particles_trajectory[0], log_weights_trajectory[0] = self.generate_initial_particles(
            y_discrete_trajectory[0],
            y_continuous_trajectory[0],
            num_particles,
            t_trajectory[0]
        )
        # We should probably populate ancestors_trajectory[0] with NA's
        # Generate the rest of the X particle trajectory by stepping through the
        # rest of the Y values
        for i in range(1, num_timesteps):
            x_discrete_particles_trajectory[i], x_continuous_particles_trajectory[i], log_weights_trajectory[i], ancestors_trajectory[i] = self.generate_next_particles(
                x_discrete_particles_trajectory[i - 1],
                x_continuous_particles_trajectory[i - 1],
                log_weights_trajectory[i - 1],
                y_discrete_trajectory[i],
                y_continuous_trajectory[i],
                t_trajectory[i - 1],
                t_trajectory[i]
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
        room_corners,
        fixed_sensor_positions,
        moving_sensor_drift_reference = 1.0,
        reference_time_delta = np.timedelta64(10, 's'),
        ping_success_probability_zero_distance = 1.0,
        receive_probability_reference_distance = 0.135,
        reference_distance = 10.0,
        rssi_untruncated_mean_intercept = -69.18,
        rssi_untruncated_mean_slope = -20.0,
        rssi_untruncated_std_dev = 5.70,
        lower_rssi_cutoff = -96.0001
    ):
        # Need to check dimensions and types of all arguments
        self.sensor_variable_structure = sensor_variable_structure
        self.room_corners = room_corners
        self.fixed_sensor_positions = fixed_sensor_positions
        self.moving_sensor_drift_reference = moving_sensor_drift_reference
        self.reference_time_delta = reference_time_delta
        self.ping_success_probability_zero_distance = ping_success_probability_zero_distance
        self.receive_probability_reference_distance = receive_probability_reference_distance
        self.reference_distance = reference_distance
        self.rssi_untruncated_mean_intercept = rssi_untruncated_mean_intercept
        self.rssi_untruncated_mean_slope = rssi_untruncated_mean_slope
        self.rssi_untruncated_std_dev = rssi_untruncated_std_dev
        self.lower_rssi_cutoff = lower_rssi_cutoff

        self.scale_factor = self.reference_distance/np.log(self.receive_probability_reference_distance/self.ping_success_probability_zero_distance)

    # Define a function which generates samples of the initial X state
    def x_initial_sample(self, num_samples = 1, t = np.nan):
        x_discrete_initial_sample = np.tile(np.array([]), (num_samples, 1))
        x_continuous_initial_sample = np.squeeze(
            self.sensor_variable_structure.extract_x_variables(
                np.random.uniform(
                    low = np.tile(self.room_corners[0], (self.sensor_variable_structure.num_sensors, 1)),
                    high = np.tile(self.room_corners[1], (self.sensor_variable_structure.num_sensors, 1)),
                    size = (num_samples, self.sensor_variable_structure.num_sensors, self.sensor_variable_structure.num_dimensions)
                )
            )
        )
        return x_discrete_initial_sample, x_continuous_initial_sample

    # Define a function which generates a sample of the current X state given the
    # previous X state
    def x_bar_x_prev_sample(self, x_discrete_prev, x_continuous_prev, t_prev = np.nan, t = np.nan):
        moving_sensor_drift = self.moving_sensor_drift_reference*np.sqrt((t - t_prev)/self.reference_time_delta)
        x_discrete_bar_x_prev_sample = np.array([])
        x_continuous_bar_x_prev_sample = stats.truncnorm.rvs(
            a = (np.tile(self.room_corners[0], self.sensor_variable_structure.num_moving_sensors) - x_continuous_prev)/moving_sensor_drift,
            b = (np.tile(self.room_corners[1], self.sensor_variable_structure.num_moving_sensors) - x_continuous_prev)/moving_sensor_drift,
            loc = x_continuous_prev,
            scale = moving_sensor_drift
        )
        return x_discrete_bar_x_prev_sample, x_continuous_bar_x_prev_sample

    # Define a function which takes an X value and returns an array representing
    # the positions of all sensors (including the fixed sensors)
    def sensor_positions(self, x_continuous):
        return np.concatenate(
            (x_continuous.reshape(x_continuous.shape[:-1] + (self.sensor_variable_structure.num_moving_sensors, self.sensor_variable_structure.num_dimensions)),
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

    def ping_success_probability(self, distances):
        return self.ping_success_probability_zero_distance*np.exp(distances/self.scale_factor)

    # Define a function which takes a vector of inter-sensor distances
    # corresponding to the Y variables and returns a corresponding array of
    # ping success probabilities
    def ping_success_probabilities_array(self, distances):
        probabilities = self.ping_success_probability(distances)
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

    def rssi_untruncated_mean(self, distance):
        return self.rssi_untruncated_mean_intercept + self.rssi_untruncated_mean_slope*np.log10(distance)

    def rssi_truncated_mean(self, distance):
        return stats.truncnorm.stats(
            a = (self.lower_rssi_cutoff - self.rssi_untruncated_mean(distance))/self.rssi_untruncated_std_dev,
            b = np.inf,
            loc = self.rssi_untruncated_mean(distance),
            scale = self.rssi_untruncated_std_dev,
            moments = 'm')

    def rssi_samples(self, distances):
        return stats.truncnorm.rvs(
            a = (self.lower_rssi_cutoff - self.rssi_untruncated_mean(distances))/self.rssi_untruncated_std_dev,
            b = np.inf,
            loc = self.rssi_untruncated_mean(distances),
            scale = self.rssi_untruncated_std_dev)

    def left_truncnorm_logpdf(self, x, untruncated_mean, untruncated_std_dev, left_cutoff):
        logf = np.array(
            np.subtract(stats.norm.logpdf(x, loc=untruncated_mean, scale=untruncated_std_dev),
            np.log(1 - stats.norm.cdf(left_cutoff, loc=untruncated_mean, scale=untruncated_std_dev))))
        logf[x < left_cutoff] = -np.inf
        return logf

    def rssi_log_pdf(self, rssi, distance):
        return self.left_truncnorm_logpdf(
            rssi,
            self.rssi_untruncated_mean(distance),
            self.rssi_untruncated_std_dev,
            self.lower_rssi_cutoff)

    # Define a function which takes an X value and returns a sample of the
    # continuous Y variables
    def y_continuous_bar_x_sample(self, x_discrete, x_continuous):
        return self.rssi_samples(self.distances(self.sensor_positions(x_continuous)))

    # Define a function which combines the above to take an X value and return a
    # Y sample
    def y_bar_x_sample(self, x_discrete, x_continuous, t = np.nan):
        return self.y_discrete_bar_x_sample(x_discrete, x_continuous), self.y_continuous_bar_x_sample(x_discrete, x_continuous)

    # Define a function which takes an X value and a Y value and returns the
    # probability density of that Y value given that X value
    def y_bar_x_log_pdf(self, x_discrete, x_continuous, y_discrete, y_continuous, t = np.nan):
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
        continuous_log_probability_densities = self.rssi_log_pdf(
            y_continuous,
            distances_x
        )
        continuous_log_probability_densities[y_discrete == 1] = 0.0
        return np.sum(discrete_log_probabilities, axis=-1) + np.sum(continuous_log_probability_densities, axis=-1)
