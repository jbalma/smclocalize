import numpy as np
from scipy import stats
import tensorflow as tf

from .smc_model import *

# Define a class (based on the generic sequential Monte Carlo model class) which
# implements our specific sensor localization model
class SensorModel(SMCModel):
    # We need to supply this object with the sensor variable structure (an
    # instance of the SensorVariableStructure class), the number of
    # particles that should be used in doing inference (required by the parent
    # class), the physical classroom configuration (room corners and positions
    # of fixed sensors), and various parameters for the state transition
    # function and sensor response functions.
    def __init__(
        self,
        sensor_variable_structure,
        room_corners,
        fixed_sensor_positions,
        num_particles = 10000,
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
        self.num_particles = num_particles
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

        self.reference_time_delta_seconds = reference_time_delta/np.timedelta64(1, 's')
        self.scale_factor = self.reference_distance/np.log(self.receive_probability_reference_distance/self.ping_success_probability_zero_distance)
        self.norm_exponent_factor = -1/(2*self.rssi_untruncated_std_dev**2)

        # Call the constructor for the parent class to initialize variables and
        # create the computational graph that we will use in all calculations
        super().__init__(
            self.sensor_variable_structure.num_x_discrete_vars,
            self.sensor_variable_structure.num_x_continuous_vars,
            self.sensor_variable_structure.num_y_discrete_vars,
            self.sensor_variable_structure.num_y_continuous_vars,
            self.num_particles)

        # Create a separate testing graph which is only used in testing the
        # various functions below (this should ultimately be moved to its own
        # class)
        self.sensor_model_testing_graph = tf.Graph() # Need to move this to its own class
        with self.sensor_model_testing_graph.as_default():

            self.num_samples_test_input_tensor = tf.placeholder(tf.int32)
            self.x_discrete_previous_test_input_tensor = tf.placeholder(tf.int32)
            self.x_continuous_previous_test_input_tensor = tf.placeholder(tf.float32)
            self.t_delta_seconds_test_input_tensor = tf.placeholder(tf.float32)
            self.x_discrete_test_input_tensor = tf.placeholder(tf.int32)
            self.x_continuous_test_input_tensor = tf.placeholder(tf.float32)
            self.y_discrete_test_input_tensor = tf.placeholder(tf.int32)
            self.y_continuous_test_input_tensor = tf.placeholder(tf.float32)
            self.distances_test_input_tensor = tf.placeholder(tf.float32)

            self.x_discrete_initial_test_output_tensor, self.x_continuous_initial_test_output_tensor = self.create_x_initial_sample_tensor(self.num_samples_test_input_tensor)

            self.x_discrete_test_output_tensor, self.x_continuous_test_output_tensor = self.create_x_bar_x_previous_sample_tensor(
                self.x_discrete_previous_test_input_tensor,
                self.x_continuous_previous_test_input_tensor,
                self.t_delta_seconds_test_input_tensor)

            self.distances_test_output_tensor = self.create_distances_tensor(
                self.x_continuous_test_input_tensor)

            self.ping_success_probabilities_test_output_tensor = self.create_ping_success_probabilities_tensor(self.distances_test_input_tensor)

            self.ping_failure_probabilities_test_output_tensor = tf.constant(1.0, dtype=tf.float32) - self.ping_success_probabilities_test_output_tensor
            self.ping_success_probabilities_array_test_output_tensor = tf.stack(
                [self.ping_success_probabilities_test_output_tensor, self.ping_failure_probabilities_test_output_tensor],
                axis=-1)

            self.rssi_untruncated_mean_test_output_tensor = self.create_rssi_untruncated_mean_tensor(self.distances_test_input_tensor)

            self.y_discrete_bar_x_sample_test_output_tensor, self.y_continuous_bar_x_sample_test_output_tensor = self.create_y_bar_x_sample_tensor(
                self.x_discrete_test_input_tensor,
                self.x_continuous_test_input_tensor)

            self.y_bar_x_log_pdf_test_output_tensor = self.create_y_bar_x_log_pdf_tensor(
                self.x_discrete_test_input_tensor,
                self.x_continuous_test_input_tensor,
                self.y_discrete_test_input_tensor,
                self.y_continuous_test_input_tensor)

        self.sensor_model_testing_session = tf.Session(graph = self.sensor_model_testing_graph)

    # These functions implement the probability distributions which define our
    # specific sensor localization model. These functions are used by the parent
    # class in building the computational graph

    def create_x_initial_sample_tensor(self, num_samples_tensor = tf.constant(1, tf.int32)):
        # Convert sensor model values to tensors
        num_x_discrete_vars_tensor = tf.constant(
            self.num_x_discrete_vars,
            tf.int32)
        num_x_continuous_vars_tensor = tf.constant(
            self.num_x_continuous_vars,
            tf.int32)
        num_moving_sensors_tensor = tf.constant(
            self.sensor_variable_structure.num_moving_sensors,
            tf.int32)
        num_dimensions_tensor = tf.constant(
            self.sensor_variable_structure.num_dimensions,
            tf.int32)
        room_corners_tensor = tf.constant(
            self.room_corners,
            tf.float32)
        # Generate the sample of the discrete X variables
        x_discrete_initial_tensor = tf.squeeze(
            tf.zeros([num_samples_tensor, num_x_discrete_vars_tensor]))
        # Generate the sample of the continuous X variables
        initial_positions_distribution = tf.distributions.Uniform(
            low = room_corners_tensor[0],
            high = room_corners_tensor[1])
        initial_positions_tensor = initial_positions_distribution.sample(
            sample_shape = [num_samples_tensor, num_moving_sensors_tensor])
        x_continuous_initial_tensor = tf.cast(
            tf.squeeze(
                tf.reshape(
                    initial_positions_tensor,
                    [num_samples_tensor, num_x_continuous_vars_tensor])),
            tf.float32)
        return x_discrete_initial_tensor, x_continuous_initial_tensor

    def create_x_bar_x_previous_sample_tensor(
        self,
        x_discrete_previous_tensor,
        x_continuous_previous_tensor,
        t_delta_seconds_tensor):
        # Convert sensor model values to tensors
        room_corners_tensor = tf.constant(
            self.room_corners,
            tf.float32)
        num_moving_sensors_tensor = tf.constant(
            self.sensor_variable_structure.num_moving_sensors,
            tf.int32)
        moving_sensor_drift_reference_tensor = tf.constant(
            self.moving_sensor_drift_reference,
            tf.float32)
        reference_time_delta_seconds_tensor = tf.constant(
            self.reference_time_delta_seconds,
            tf.float32)
        # Generate the sample of the discrete X variables
        x_discrete_tensor = x_discrete_previous_tensor
        # Generate the sample of the continuous X variables
        moving_sensor_drift_tensor = tf.multiply(
            moving_sensor_drift_reference_tensor,
            tf.sqrt(tf.divide(
                t_delta_seconds_tensor,
                reference_time_delta_seconds_tensor)))
        x_continuous_tensor = tf.cast(
            tf.reshape(
                tf.py_func(
                    stats.truncnorm.rvs,
                    [tf.divide(
                        tf.subtract(
                            tf.tile(room_corners_tensor[0], [num_moving_sensors_tensor]),
                            x_continuous_previous_tensor),
                        moving_sensor_drift_tensor),
                    tf.divide(
                        tf.subtract(
                            tf.tile(room_corners_tensor[1], [num_moving_sensors_tensor]),
                            x_continuous_previous_tensor),
                        moving_sensor_drift_tensor),
                    x_continuous_previous_tensor,
                    moving_sensor_drift_tensor],
                    tf.float64),
                tf.shape(x_continuous_previous_tensor)),
            tf.float32)
        return x_discrete_tensor, x_continuous_tensor

    def create_y_bar_x_log_pdf_tensor(
        self,
        x_discrete_tensor,
        x_continuous_tensor,
        y_discrete_tensor,
        y_continuous_tensor):
        # Convert sensor model values to tensors
        rssi_untruncated_std_dev_tensor = tf.constant(
            self.rssi_untruncated_std_dev,
            dtype=tf.float32)
        lower_rssi_cutoff_tensor = tf.constant(
            self.lower_rssi_cutoff,
            dtype=tf.float32)
        # Calculate inter-sensor distances
        distances_tensor = self.create_distances_tensor(x_continuous_tensor)
        # Calculate discrete probabilities
        ping_success_probabilities_tensor = self.create_ping_success_probabilities_tensor(distances_tensor)
        ping_failure_probabilities_tensor = tf.subtract(
            tf.constant(1.0, dtype=tf.float32),
            ping_success_probabilities_tensor)
        y_discrete_broadcast_tensor = tf.add(
            y_discrete_tensor,
            tf.zeros_like(ping_success_probabilities_tensor, tf.int32))
        discrete_probabilities_tensor = tf.where(
            tf.equal(
                y_discrete_broadcast_tensor,
                tf.constant(0, dtype=tf.int32)),
            ping_success_probabilities_tensor,
            ping_failure_probabilities_tensor)
        discrete_log_probabilities_tensor = tf.log(discrete_probabilities_tensor)
        # Calculate continuous probability densities
        y_continuous_broadcast_tensor = tf.add(
            y_continuous_tensor,
            tf.zeros_like(distances_tensor))
        rssi_untruncated_mean_tensor = self.create_rssi_untruncated_mean_tensor(distances_tensor)
        rssi_scaled_tensor = tf.divide(
            tf.subtract(
                y_continuous_broadcast_tensor,
                rssi_untruncated_mean_tensor),
            rssi_untruncated_std_dev_tensor)
        a_scaled_tensor = tf.divide(
            tf.subtract(
                lower_rssi_cutoff_tensor,
                rssi_untruncated_mean_tensor),
            rssi_untruncated_std_dev_tensor)
        untruncated_normal_distribution = tf.distributions.Normal(
            loc=tf.constant(0.0, dtype=tf.float32),
            scale = tf.constant(1.0, dtype=tf.float32))
        logf_untruncated_tensor = tf.subtract(
            untruncated_normal_distribution.log_prob(rssi_scaled_tensor),
            tf.add(
                tf.log(rssi_untruncated_std_dev_tensor),
                untruncated_normal_distribution.log_survival_function(a_scaled_tensor)))
        logf_tensor = tf.where(
            tf.less(
                y_continuous_broadcast_tensor,
                lower_rssi_cutoff_tensor),
            tf.fill(
                tf.shape(logf_untruncated_tensor),
                np.float32(-np.inf)),
            logf_untruncated_tensor)
        continuous_log_probability_densities_tensor = tf.where(
            tf.equal(
                y_discrete_broadcast_tensor,
                tf.constant(1, dtype=tf.int32)),
            tf.fill(tf.shape(logf_tensor), np.float32(0.0)),
            logf_tensor)
        # Combine discrete probabilities and continuous probability densities
        y_bar_x_log_pdf_tensor = tf.add(
            tf.reduce_sum(discrete_log_probabilities_tensor, axis = -1),
            tf.reduce_sum(continuous_log_probability_densities_tensor, axis = -1))
        return y_bar_x_log_pdf_tensor

    def create_y_bar_x_sample_tensor(
        self,
        x_discrete_tensor,
        x_continuous_tensor):
        # Convert sensor model values to tensors
        num_y_continuous_vars_tensor = tf.constant(
            self.num_y_continuous_vars,
            tf.int32)
        lower_rssi_cutoff_tensor = tf.constant(
            self.lower_rssi_cutoff,
            tf.float32)
        rssi_untruncated_std_dev_tensor = tf.constant(
            self.rssi_untruncated_std_dev,
            tf.float32)
        # Calculate inter-sensor distances
        distances_tensor = self.create_distances_tensor(x_continuous_tensor)
        # Generate the sample of the discrete Y variables
        ping_success_probabilities_tensor = self.create_ping_success_probabilities_tensor(
            distances_tensor)
        ping_success_logits_tensor = tf.log(
            tf.stack(
                [ping_success_probabilities_tensor, 1 - ping_success_probabilities_tensor],
                axis= -1))
        ping_success_logits_tensor_reshaped = tf.reshape(
            ping_success_logits_tensor,
            [-1, num_y_continuous_vars_tensor, 2])
        y_discrete_bar_x_sample_tensor = tf.cast(
            tf.squeeze(
                tf.map_fn(
                    lambda logits_tensor: tf.transpose(
                        tf.multinomial(logits_tensor, num_samples = 1)),
                    ping_success_logits_tensor_reshaped,
                    dtype=tf.int64)),
            tf.int32)
        # Generate the sample of the continuous Y variables
        rssi_untruncated_mean_tensor = self.create_rssi_untruncated_mean_tensor(distances_tensor)
        y_continuous_bar_x_sample_tensor = tf.cast(
            tf.py_func(
                stats.truncnorm.rvs,
                [tf.divide(
                    tf.subtract(
                        lower_rssi_cutoff_tensor,
                        rssi_untruncated_mean_tensor),
                    rssi_untruncated_std_dev_tensor),
                np.float32(np.inf),
                rssi_untruncated_mean_tensor,
                rssi_untruncated_std_dev_tensor],
                tf.float64),
            tf.float32)
        return y_discrete_bar_x_sample_tensor, y_continuous_bar_x_sample_tensor

    # These functions do not appear in the parent class. They are used by the
    # functions above

    def create_distances_tensor(self, x_continuous_tensor):
        # Convert sensor model values to tensors
        num_moving_sensors_tensor = tf.constant(
            self.sensor_variable_structure.num_moving_sensors,
            tf.int32)
        num_dimensions_tensor = tf.constant(
            self.sensor_variable_structure.num_dimensions,
            tf.int32)
        fixed_sensor_positions_tensor = tf.constant(
            self.fixed_sensor_positions,
            dtype=tf.float32)
        extract_y_variables_mask_tensor = tf.constant(
            self.sensor_variable_structure.extract_y_variables_mask,
            tf.bool)
        # Calculate matrix of inter-sensor distances
        x_continuous_samples_shape_tensor = tf.shape(x_continuous_tensor)[:-1]
        moving_sensor_positions_tensor = tf.reshape(
            x_continuous_tensor,
            tf.concat(
                [x_continuous_samples_shape_tensor, [num_moving_sensors_tensor, num_dimensions_tensor]],
                axis=0))
        fixed_sensor_positions_shape_tensor = tf.shape(fixed_sensor_positions_tensor)
        fixed_sensor_positions_broadcast_tensor = tf.add(
            fixed_sensor_positions_tensor,
            tf.zeros(
                tf.concat(
                    [x_continuous_samples_shape_tensor, fixed_sensor_positions_shape_tensor],
                    axis=0),
            dtype=tf.float32))
        all_sensor_positions_tensor = tf.concat(
            [moving_sensor_positions_tensor, fixed_sensor_positions_broadcast_tensor],
            axis = -2)
        distance_matrix_tensor = tf.norm(
            tf.subtract(
                tf.expand_dims(all_sensor_positions_tensor, axis = -3),
                tf.expand_dims(all_sensor_positions_tensor, axis=-2)),
            axis=-1)
        # Extract and flatten the inter-sensor distances that correspond to
        # continuous Y variables
        distance_matrix_indices_tensor = tf.range(tf.rank(distance_matrix_tensor))
        distance_matrix_roll_forward_tensor = tf.transpose(
            distance_matrix_tensor,
            tf.concat(
                [distance_matrix_indices_tensor[-2:], distance_matrix_indices_tensor[:-2]],
                axis=0))
        distances_roll_forward_tensor = tf.boolean_mask(
            distance_matrix_roll_forward_tensor,
            extract_y_variables_mask_tensor)
        distances_roll_forward_indices_tensor = tf.range(tf.rank(distances_roll_forward_tensor))
        distances_tensor = tf.transpose(
            distances_roll_forward_tensor,
            tf.concat(
                [distances_roll_forward_indices_tensor[1:], [0]],
                axis=0))
        return distances_tensor

    def create_ping_success_probabilities_tensor(
        self,
        distances_tensor):
        # Convert sensor model values to tensors
        ping_success_probability_zero_distance_tensor = tf.constant(
            self.ping_success_probability_zero_distance,
            dtype=tf.float32)
        scale_factor_tensor = tf.constant(
            self.scale_factor,
            dtype=tf.float32)
        # Calculate ping success probabilties
        ping_success_probabilities_tensor = tf.multiply(
            ping_success_probability_zero_distance_tensor,
            tf.exp(
                tf.divide(
                    distances_tensor,
                    scale_factor_tensor)))
        return ping_success_probabilities_tensor

    def create_rssi_untruncated_mean_tensor(
        self,
        distances_tensor):
        # Convert sensor model values to tensors
        rssi_untruncated_mean_intercept_tensor = tf.constant(
            self.rssi_untruncated_mean_intercept,
            dtype=tf.float32)
        rssi_untruncated_mean_slope_tensor = tf.constant(
            self.rssi_untruncated_mean_slope,
            dtype=tf.float32)
        # Calculate untruncted means of RSSI distributions
        rssi_untruncated_mean_tensor = tf.add(
            rssi_untruncated_mean_intercept_tensor,
            tf.multiply(
                rssi_untruncated_mean_slope_tensor,
                tf.divide(
                    tf.log(distances_tensor),
                    tf.constant(np.log(10.0), dtype=tf.float32))))
        return rssi_untruncated_mean_tensor

    # These functions are just used for testing the functions above

    def x_initial_sample_test(self, num_samples_test_input = 1):
        return self.sensor_model_testing_session.run(
            [self.x_discrete_initial_test_output_tensor, self.x_continuous_initial_test_output_tensor],
            feed_dict = {
                self.num_samples_test_input_tensor: num_samples_test_input})

    def x_bar_x_previous_sample_test(self, x_discrete_previous_test_input, x_continuous_previous_test_input, t_delta_test_input):
        t_delta_seconds_test_input = t_delta_test_input/np.timedelta64(1, 's')
        return self.sensor_model_testing_session.run(
            [self.x_discrete_test_output_tensor, self.x_continuous_test_output_tensor],
            feed_dict = {
                self.x_discrete_previous_test_input_tensor: x_discrete_previous_test_input,
                self.x_continuous_previous_test_input_tensor: x_continuous_previous_test_input,
                self.t_delta_seconds_test_input_tensor: t_delta_seconds_test_input})

    def distances_test(self, x_continuous_test_input):
        return self.sensor_model_testing_session.run(
            self.distances_test_output_tensor,
            feed_dict = {
                self.x_continuous_test_input_tensor: x_continuous_test_input})

    def ping_success_probabilities_test(self, distances_test_input):
        return self.sensor_model_testing_session.run(
            self.ping_success_probabilities_test_output_tensor,
            feed_dict = {
                self.distances_test_input_tensor: distances_test_input})

    def ping_success_probabilities_array_test(self, distances_test_input):
        return self.sensor_model_testing_session.run(
            self.ping_success_probabilities_array_test_output_tensor,
            feed_dict = {
                self.distances_test_input_tensor: distances_test_input})

    def rssi_untruncated_mean_test(self, distances_test_input):
        return self.sensor_model_testing_session.run(
            self.rssi_untruncated_mean_test_output_tensor,
            feed_dict = {
                self.distances_test_input_tensor: distances_test_input})

    def rssi_truncated_mean_test(self, distances_test):
        return stats.truncnorm.stats(
            a = (self.lower_rssi_cutoff - self.rssi_untruncated_mean_test(distances_test))/self.rssi_untruncated_std_dev,
            b = np.inf,
            loc = self.rssi_untruncated_mean_test(distances_test),
            scale = self.rssi_untruncated_std_dev,
            moments = 'm')

    def y_bar_x_sample_test(self, x_discrete_test_input, x_continuous_test_input):
        return self.sensor_model_testing_session.run(
            [self.y_discrete_bar_x_sample_test_output_tensor, self.y_continuous_bar_x_sample_test_output_tensor],
            feed_dict = {
                self.x_discrete_test_input_tensor: x_discrete_test_input,
                self.x_continuous_test_input_tensor: x_continuous_test_input})

    def y_bar_x_log_pdf_test(
        self,
        x_discrete_test_input,
        x_continuous_test_input,
        y_discrete_test_input,
        y_continuous_test_input):
        return self.sensor_model_testing_session.run(
            self.y_bar_x_log_pdf_test_output_tensor,
            feed_dict = {
                self.x_discrete_test_input_tensor: x_discrete_test_input,
                self.x_continuous_test_input_tensor: x_continuous_test_input,
                self.y_discrete_test_input_tensor: y_discrete_test_input,
                self.y_continuous_test_input_tensor: y_continuous_test_input})
