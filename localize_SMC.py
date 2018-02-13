import numpy as np
from scipy import special
from scipy import stats
import math
import time

# Define a class which provides a bunch of tools for working with sensor data,
# based on lists of entity IDs
class SensorVariableStructure(object):
    def __init__(
        self,
        child_entity_ids,
        material_entity_ids,
        teacher_entity_ids,
        area_entity_ids,
        num_dimensions = 2,
        child_entity_string = 'child',
        material_entity_string = 'material',
        teacher_entity_string = 'teacher',
        area_entity_string = 'area'):
        self.child_entity_ids = child_entity_ids
        self.material_entity_ids = material_entity_ids
        self.teacher_entity_ids = teacher_entity_ids
        self.area_entity_ids = area_entity_ids
        self.num_dimensions = num_dimensions

        self.child_entity_id_index = [child_entity_string + '_' + str(child_entity_id) for child_entity_id in child_entity_ids]
        self.material_entity_id_index = [material_entity_string + '_' + str(material_entity_id) for material_entity_id in material_entity_ids]
        self.teacher_entity_id_index = [teacher_entity_string + '_' + str(teacher_entity_id) for teacher_entity_id in teacher_entity_ids]
        self.area_entity_id_index = [area_entity_string + '_' + str(area_entity_id) for area_entity_id in area_entity_ids]
        self.entity_id_index = self.child_entity_id_index + self.material_entity_id_index + self.teacher_entity_id_index + self.area_entity_id_index

        self.num_child_sensors = len(child_entity_ids)
        self.num_material_sensors = len(material_entity_ids)
        self.num_teacher_sensors = len(teacher_entity_ids)
        self.num_area_sensors = len(area_entity_ids)

        self.num_moving_sensors = self.num_child_sensors + self.num_material_sensors + self.num_teacher_sensors
        self.num_fixed_sensors = self.num_area_sensors
        self.num_sensors = self.num_moving_sensors + self.num_fixed_sensors

        # Define a Boolean mask which helps us extract and flatten X values from
        # an array representing sensor positions. Start with an
        # array that has a row for every sensor and a column for every spatial
        # dimension.
        self.extract_x_variables_mask = np.full((self.num_sensors, self.num_dimensions), True)
        # We don't track the positions of fixed sensors.
        self.extract_x_variables_mask[self.num_moving_sensors:,:] = False

        # Define the number of discrete and continuous x variables using this
        # mask. This is the key information needed by the SMC model functions.
        self.num_x_discrete_vars = 0
        self.num_x_continuous_vars = np.sum(self.extract_x_variables_mask)

        # Define a Boolean mask which help us extract and flatten Y values from
        # an array representing every pairwise combination of sensors. Start
        # with an array that has every pairwise combination of sensors.
        self.extract_y_variables_mask = np.full((self.num_sensors, self.num_sensors), True)
        # Sensors don't send pings to themselves
        np.fill_diagonal(self.extract_y_variables_mask, False)
        # We don't store pings from material sensors to other material sensors
        self.extract_y_variables_mask[
            self.num_child_sensors:(self.num_child_sensors + self.num_material_sensors),
            self.num_child_sensors:(self.num_child_sensors + self.num_material_sensors)] = False
        # We don't store pings from teacher sensors to other teacher sensors
        self.extract_y_variables_mask[
            (self.num_child_sensors + self.num_material_sensors):self.num_moving_sensors,
            (self.num_child_sensors + self.num_material_sensors):self.num_moving_sensors] = False
        # We don't store pings from area sensors to other area sensors
        self.extract_y_variables_mask[
            self.num_moving_sensors:,
            self.num_moving_sensors:] = False
        # We don't store pings from material sensors to area sensors (and vice versa)
        self.extract_y_variables_mask[
            self.num_child_sensors:(self.num_child_sensors + self.num_material_sensors),
            self.num_moving_sensors:] = False
        self.extract_y_variables_mask[
            self.num_moving_sensors:,
            self.num_child_sensors:(self.num_child_sensors + self.num_material_sensors)] = False

        # Define the number of discrete and continuous Y variables using this
        # mask.
        self.num_y_discrete_vars = np.sum(self.extract_y_variables_mask)
        self.num_y_continuous_vars = np.sum(self.extract_y_variables_mask)

        # Define names for the sensor variables and their values.
        self.child_sensor_names = ['Child sensor {}'.format(id) for id in child_entity_ids]
        self.material_sensor_names = ['Material sensor {}'.format(id) for id in material_entity_ids]
        self.teacher_sensor_names = ['Teacher sensor {}'.format(id) for id in teacher_entity_ids]
        self.area_sensor_names = ['Area sensor {}'.format(id) for id in area_entity_ids]

        self.moving_sensor_names = self.child_sensor_names + self.material_sensor_names + self.teacher_sensor_names
        self.fixed_sensor_names = self.area_sensor_names
        self.sensor_names = self.moving_sensor_names + self.fixed_sensor_names

        self.dimension_names_all = ['$l$', '$w$', '$h']
        self.dimension_names = self.dimension_names_all[:self.num_dimensions]

        self.x_discrete_names = []
        self.sensor_position_name_matrix = [[
            '{} {} position'.format(sensor_name, dimension_name)
            for dimension_name in self.dimension_names]
            for sensor_name in self.sensor_names]
        self.x_continuous_names = self.extract_x_variables(np.array(self.sensor_position_name_matrix)).tolist()

        self.y_discrete_name_matrix = [[
            'Status of ping from {} to {}'.format(sending_sensor_name, receiving_sensor_name)
            for receiving_sensor_name in self.sensor_names]
            for sending_sensor_name in self.sensor_names]
        self.y_continuous_name_matrix = [[
            'RSSI of ping from {} to {}'.format(sending_sensor_name, receiving_sensor_name)
            for receiving_sensor_name in self.sensor_names]
            for sending_sensor_name in self.sensor_names]
        self.y_discrete_names = self.extract_y_variables(np.array(self.y_discrete_name_matrix)).tolist()
        self.y_continuous_names = self.extract_y_variables(np.array(self.y_continuous_name_matrix)).tolist()

        self.ping_status_names = ['Received', 'Not received']
        self.num_ping_statuses = len(self.ping_status_names)

    # Define functions which use the Boolean masks above to extract and
    # flatten X and Y values from larger data arrays.
    def extract_x_variables(self, a):
        return a[..., self.extract_x_variables_mask]
    def extract_y_variables(self, a):
        return a[..., self.extract_y_variables_mask]

    # Parse a dataframe containing a single time step of ping data
    def sensor_data_parse_one_timestep(self, dataframe):
        y_discrete_all_sensors = np.ones(
            (self.num_sensors, self.num_sensors),
            dtype='int')
        y_continuous_all_sensors = np.zeros(
            (self.num_sensors, self.num_sensors),
            dtype='float')
        for row in range(len(dataframe)):
            y_discrete_all_sensors[
                self.entity_id_index.index(dataframe.iloc[row]['remote_type'] + '_' + str(dataframe.iloc[row]['remote_id'])),
                self.entity_id_index.index(dataframe.iloc[row]['local_type'] + '_' + str(dataframe.iloc[row]['local_id']))] = 0
            y_continuous_all_sensors[
                self.entity_id_index.index(dataframe.iloc[row]['remote_type'] + '_' + str(dataframe.iloc[row]['remote_id'])),
                self.entity_id_index.index(dataframe.iloc[row]['local_type'] + '_' + str(dataframe.iloc[row]['local_id']))] = dataframe.iloc[row]['rssi']
        return self.extract_y_variables(y_discrete_all_sensors), self.extract_y_variables(y_continuous_all_sensors)

    # Parse a dataframe containing multiple time steps of ping data
    def sensor_data_parse_multiple_timesteps(self, dataframe):
        timestamps = np.sort(dataframe['observed_at'].unique())
        num_timesteps = len(timestamps)
        y_discrete_t = np.ones(
            (num_timesteps, self.num_y_discrete_vars),
            dtype='int')
        y_continuous_t = np.zeros(
            (num_timesteps, self.num_y_continuous_vars),
            dtype='float')
        for t_index in range(num_timesteps):
            y_discrete_t[t_index], y_continuous_t[t_index] = self.sensor_data_parse_one_timestep(
            dataframe[dataframe['observed_at'] == timestamps[t_index]])
        return y_discrete_t, y_continuous_t, timestamps

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
            print '[generate_next_particles] Anc: {:.1e} Trans: {:.1e} Wts: {:.1e} Renorm: {:.1e}'.format(
                after_ancestors - start,
                after_transition - after_ancestors,
                after_weights - after_transition,
                after_renormalize - after_weights)
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

# Define a class based on the generic sequential Monte Carlo model class which
# represents an instance of our particular sensor model
class SensorModel(SMCModel):
    # We need to supply this object with the sensor variable structure (an
    # instance of the SensorVariableStructure class above), the physical
    # classroom configuration (room corners and positions of fixed sensors), and
    # various parameters for the state transition function and sensor response
    # functions.
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
        lower_rssi_cutoff = -96.0001):
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

        self.num_x_discrete_vars = self.sensor_variable_structure.num_x_discrete_vars
        self.num_x_continuous_vars = self.sensor_variable_structure.num_x_continuous_vars
        self.num_y_discrete_vars = self.sensor_variable_structure.num_y_discrete_vars
        self.num_y_continuous_vars = self.sensor_variable_structure.num_y_continuous_vars

        self.scale_factor = self.reference_distance/np.log(self.receive_probability_reference_distance/self.ping_success_probability_zero_distance)
        self.norm_exponent_factor = -1/(2*self.rssi_untruncated_std_dev**2)

    # Define the function which generate samples of the initial X state. This
    # function is needed by the SMC model functions above.
    def x_initial_sample(self, num_samples = 1):
        x_discrete_initial_sample = np.tile(np.array([]), (num_samples, 1))
        x_continuous_initial_sample = np.squeeze(
            self.sensor_variable_structure.extract_x_variables(
                np.random.uniform(
                    low = np.tile(self.room_corners[0], (self.sensor_variable_structure.num_sensors, 1)),
                    high = np.tile(self.room_corners[1], (self.sensor_variable_structure.num_sensors, 1)),
                    size = (num_samples, self.sensor_variable_structure.num_sensors, self.sensor_variable_structure.num_dimensions))))
        return x_discrete_initial_sample, x_continuous_initial_sample

    # Define the function which generates a sample of the current X state given
    # the previous X state. This function is needed by the SMC model functions
    # above.
    def x_bar_x_previous_sample(self, x_discrete_previous, x_continuous_previous, t_delta):
        moving_sensor_drift = self.moving_sensor_drift_reference*np.sqrt(t_delta/self.reference_time_delta)
        x_discrete_bar_x_previous_sample = np.array([])
        x_continuous_bar_x_previous_sample = stats.truncnorm.rvs(
            a = (np.tile(self.room_corners[0], self.sensor_variable_structure.num_moving_sensors) - x_continuous_previous)/moving_sensor_drift,
            b = (np.tile(self.room_corners[1], self.sensor_variable_structure.num_moving_sensors) - x_continuous_previous)/moving_sensor_drift,
            loc = x_continuous_previous,
            scale = moving_sensor_drift)
        return x_discrete_bar_x_previous_sample, x_continuous_bar_x_previous_sample

    # Generate an array of the positions of all sensors (including the fixed
    # sensors) based on an X value (or an array of X values).
    def sensor_positions(self, x_continuous):
        return np.concatenate(
            (x_continuous.reshape(x_continuous.shape[:-1] + (self.sensor_variable_structure.num_moving_sensors, self.sensor_variable_structure.num_dimensions)),
            np.broadcast_to(self.fixed_sensor_positions, x_continuous.shape[:-1] + self.fixed_sensor_positions.shape)),
            axis=-2)

    # Generate an array of of inter-sensor distances (with the same structure as
    # Y values) based on an array of sensor positions.
    def distances(self, sensor_positions):
        return self.sensor_variable_structure.extract_y_variables(
            np.linalg.norm(
                np.subtract(
                    sensor_positions[...,np.newaxis,:,:],
                    sensor_positions[...,:,np.newaxis,:]),
                axis = -1))

    # Generate the probability of a ping being received (or an array of such
    # probabilities) given the distance between the sending and receiving
    # sensors (or an array of such distances).
    def ping_success_probability(self, distances):
        return self.ping_success_probability_zero_distance*np.exp(distances/self.scale_factor)

    # Generate an array which combines the probabilities calculated above along
    # with their complements (i.e., the probabilities of the pings *not* being
    # received). This structure is needed for several functions below.
    def ping_success_probabilities_array(self, distances):
        probabilities = self.ping_success_probability(distances)
        return np.stack((probabilities, 1 - probabilities), axis=-1)

    # Generate an array of ping success samples given an array of inter-sensor
    # distances.
    def ping_success_samples(self, distances):
        return np.apply_along_axis(
            lambda p_array: np.random.choice(len(p_array), p=p_array),
            axis=-1,
            arr=self.ping_success_probabilities_array(distances))

    # Generate a sample discrete Y value (or an array of such samples) based on
    # a value of X (or an array of X values).
    def y_discrete_bar_x_sample(self, x_discrete, x_continuous):
        return self.ping_success_samples(self.distances(self.sensor_positions(x_continuous)))

    # Generate the mean of the underlying distribution of measured RSSI values
    # (or an array of such means) given the distance between two senors (or an
    # array of such distances).
    def rssi_untruncated_mean(self, distance):
        return self.rssi_untruncated_mean_intercept + self.rssi_untruncated_mean_slope*np.log10(distance)

    # Generate the mean of the distribution which results when we truncate the
    # distribution above.
    def rssi_truncated_mean(self, distance):
        return stats.truncnorm.stats(
            a = (self.lower_rssi_cutoff - self.rssi_untruncated_mean(distance))/self.rssi_untruncated_std_dev,
            b = np.inf,
            loc = self.rssi_untruncated_mean(distance),
            scale = self.rssi_untruncated_std_dev,
            moments = 'm')

    # Generate a sample RSSI value (or an array of such samples) given the
    # distance between two sensors (or an array of such distances).
    def rssi_samples(self, distances):
        return stats.truncnorm.rvs(
            a = (self.lower_rssi_cutoff - self.rssi_untruncated_mean(distances))/self.rssi_untruncated_std_dev,
            b = np.inf,
            loc = self.rssi_untruncated_mean(distances),
            scale = self.rssi_untruncated_std_dev)

    # Generate the (log of the) probability density (or an array of such
    # probability densities) for a given RSSI value (or an array of such values)
    # given the distance between two sensors (or an array of such distances). We
    # implement a truncated normal distribution by combining Numpy functions
    # rather than by using the Scipy function because the latter appears to have
    # a bug in its broadcasting logic.
    def rssi_log_pdf(self, rssi, distance):
        if __debug__:
            start=time.clock()
        rssi_scaled = np.subtract(rssi, self.rssi_untruncated_mean(distance))/self.rssi_untruncated_std_dev
        if __debug__:
            after_rssi_scale = time.clock()
        a_scaled = np.subtract(self.lower_rssi_cutoff, self.rssi_untruncated_mean(distance))/self.rssi_untruncated_std_dev
        if __debug__:
            after_a_scale = time.clock()
        log_sigma = np.log(self.rssi_untruncated_std_dev)
        logf = stats.norm._logpdf(rssi_scaled) - log_sigma - stats.norm._logsf(a_scaled)
        if __debug__:
            after_dists = time.clock()
        logf[rssi < self.lower_rssi_cutoff] = -np.inf
        if __debug__:
            after_truncate = time.clock()
        if __debug__:
            print '[rssi_log_pdf] rssi_scale: {:.1e} a_scale: {:.1e} dists: {:.1e} trunc: {:.1e}'.format(
                after_rssi_scale - start,
                after_a_scale - after_rssi_scale,
                after_dists - after_a_scale,
                after_truncate - after_dists)
        return logf


    # Using the various helper functions above, generate a sample continuous Y
    # value (or an array of such samples) given an X value (or an array of such
    # values)
    def y_continuous_bar_x_sample(self, x_discrete, x_continuous):
        return self.rssi_samples(self.distances(self.sensor_positions(x_continuous)))

    # Define a function which combines the above to produce a sample Y value (or
    # an array of such samples) given an X value (or an array of such values).
    # This function is needed by the SMC model functions above.
    def y_bar_x_sample(self, x_discrete, x_continuous):
        return self.y_discrete_bar_x_sample(x_discrete, x_continuous), self.y_continuous_bar_x_sample(x_discrete, x_continuous)

    # Define a function which takes an X value and a Y value and returns the
    # probability density of that Y value given that X value. This function is
    # needed by the SMC model functions above.
    def y_bar_x_log_pdf(
        self,
        x_discrete,
        x_continuous,
        y_discrete,
        y_continuous):
        if __debug__:
            start = time.clock()
        distances_x = self.distances(self.sensor_positions(x_continuous))
        if __debug__:
            after_distance = time.clock()
        ping_success_probabilities_array_x = self.ping_success_probabilities_array(distances_x)
        if __debug__:
            after_probarray = time.clock()
        discrete_log_probabilities = np.log(
            np.choose(
                y_discrete,
                np.rollaxis(
                    ping_success_probabilities_array_x,
                    axis=-1)))
        if __debug__:
            after_discrete = time.clock()
        continuous_log_probability_densities = self.rssi_log_pdf(
            y_continuous,
            distances_x)
        if __debug__:
            after_continuous = time.clock()
        continuous_log_probability_densities[y_discrete == 1] = 0.0
        if __debug__:
            print '[y_bar_x_log_pdf] Dist: {:.1e} ProbArray: {:.1e} Discrete: {:.1e} Cont: {:.1e}'.format(
                after_distance - start,
                after_probarray - after_distance,
                after_discrete - after_probarray,
                after_continuous - after_discrete)
        return np.sum(discrete_log_probabilities, axis=-1) + np.sum(continuous_log_probability_densities, axis=-1)
