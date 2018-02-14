import numpy as np
from scipy import stats
import time

from .smc_model import *

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
            print ('[rssi_log_pdf] rssi_scale: {:.1e} a_scale: {:.1e} dists: {:.1e} trunc: {:.1e}'.format(
                after_rssi_scale - start,
                after_a_scale - after_rssi_scale,
                after_dists - after_a_scale,
                after_truncate - after_dists))
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
            print ('[y_bar_x_log_pdf] Dist: {:.1e} ProbArray: {:.1e} Discrete: {:.1e} Cont: {:.1e}'.format(
                after_distance - start,
                after_probarray - after_distance,
                after_discrete - after_probarray,
                after_continuous - after_discrete))
        return np.sum(discrete_log_probabilities, axis=-1) + np.sum(continuous_log_probability_densities, axis=-1)
