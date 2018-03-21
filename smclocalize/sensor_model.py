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
        initial_status_on_probability = 0.9,
        status_on_to_off_probability = 0.01,
        status_off_to_on_probability = 0.01,
        ping_success_probability_sensor_on = 0.999,
        ping_success_probability_sensor_off = 0.001,
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
        self.initial_status_on_probability = initial_status_on_probability
        self.status_on_to_off_probability = status_on_to_off_probability
        self.status_off_to_on_probability = status_off_to_on_probability
        self.ping_success_probability_sensor_on = ping_success_probability_sensor_on
        self.ping_success_probability_sensor_off = ping_success_probability_sensor_off
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
            num_particles)

        # Create a separate testing graph which is only used in testing the
        # various functions below (this should ultimately be moved to its own
        # class)
        self.sensor_model_testing_graph = tf.Graph() # Need to move this to its own class
        with self.sensor_model_testing_graph.as_default():

            self.num_samples_test_input_tensor = tf.placeholder(
                tf.int32,
                name='num_samples_test_input')
            self.x_discrete_previous_test_input_tensor = tf.placeholder(
                tf.int32,
                name='x_discrete_previous_test_input')
            self.x_continuous_previous_test_input_tensor = tf.placeholder(
                tf.float32,
                name='x_continuous_previous_test_input')
            self.t_delta_seconds_test_input_tensor = tf.placeholder(
                tf.float32,
                name='t_delta_seconds_test_input')
            self.x_discrete_test_input_tensor = tf.placeholder(
                tf.int32,
                name='x_discrete_test_input')
            self.x_continuous_test_input_tensor = tf.placeholder(
                tf.float32,
                name='x_continuous_test_input')
            self.y_discrete_test_input_tensor = tf.placeholder(
                tf.int32,
                name='y_discrete_test_input')
            self.y_continuous_test_input_tensor = tf.placeholder(
                tf.float32,
                name='y_continuous_test_input')
            self.distances_test_input_tensor = tf.placeholder(
                tf.float32,
                name='distances_test_input')

            self.x_discrete_initial_test_output_tensor, self.x_continuous_initial_test_output_tensor = self.create_x_initial_sample_tensor(self.num_samples_test_input_tensor)

            self.x_discrete_test_output_tensor, self.x_continuous_test_output_tensor = self.create_x_bar_x_previous_sample_tensor(
                self.x_discrete_previous_test_input_tensor,
                self.x_continuous_previous_test_input_tensor,
                self.t_delta_seconds_test_input_tensor)

            self.distances_test_output_tensor = self.create_distances_tensor(
                self.x_continuous_test_input_tensor)

            self.ping_success_probabilities_test_output_tensor, self.ping_failure_probabilities_test_output_tensor = self.create_ping_success_probabilities_tensor(
                self.x_discrete_test_input_tensor,
                self.distances_test_input_tensor)

            self.ping_success_probabilities_from_sensor_statuses_test_output_tensor = self.create_ping_success_probabilities_from_sensor_statuses_tensor(
                self.x_discrete_test_input_tensor)

            self.ping_success_probabilities_from_distances_test_output_tensor = self.create_ping_success_probabilities_from_distances_tensor(self.distances_test_input_tensor)

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
        with tf.name_scope('create_x_initial_sample'):
            # Convert sensor model values to tensors
            room_corners_tensor = tf.constant(
                self.room_corners,
                tf.float32,
                name='room_corners')
            # Generate the sample of the discrete X variables
            initial_statuses_distribution = tf.distributions.Bernoulli(
                probs =  1 - self.initial_status_on_probability,
                dtype = tf.int32,
                name = 'bernoulli_distribution')
            x_discrete_initial_tensor = tf.squeeze(
                initial_statuses_distribution.sample(
                    sample_shape = [num_samples_tensor, self.num_x_discrete_vars],
                    name = 'calculate_sensor_status_probabilities'),
                name = 'create_x_discrete_initial_tensor')
            # Generate the sample of the continuous X variables
            initial_positions_distribution = tf.distributions.Uniform(
                low = room_corners_tensor[0],
                high = room_corners_tensor[1],
                name='uniform_distribution_across_room')
            initial_positions_tensor = initial_positions_distribution.sample(
                sample_shape = [num_samples_tensor, self.sensor_variable_structure.num_moving_sensors],
                name='create_initial_positions')
            x_continuous_initial_tensor = tf.cast(
                tf.squeeze(
                    tf.reshape(
                        initial_positions_tensor,
                        [num_samples_tensor, self.num_x_continuous_vars],
                        name='reshape_initial_positions'),
                    name='squeeze_initial_positions'),
                tf.float32,
                name='create_x_continuous_initial')
        return x_discrete_initial_tensor, x_continuous_initial_tensor

    def create_x_bar_x_previous_sample_tensor(
        self,
        x_discrete_previous_tensor,
        x_continuous_previous_tensor,
        t_delta_seconds_tensor):
        with tf.name_scope('create_x_bar_x_previous_sample'):
            # Convert sensor model values to tensors
            room_corners_tensor = tf.constant(
                self.room_corners,
                tf.float32,
                name='room_corners')
            moving_sensor_drift_reference_tensor = tf.constant(
                self.moving_sensor_drift_reference,
                tf.float32,
                name='moving_sensor_drift_reference')
            reference_time_delta_seconds_tensor = tf.constant(
                self.reference_time_delta_seconds,
                tf.float32,
                name='reference_time_delta_seconds')
            # Generate the sample of the discrete X variables
            x_discrete_previous_shape_tensor = tf.shape(x_discrete_previous_tensor)
            status_on_previous_tensor = tf.logical_not(
                tf.cast(
                    x_discrete_previous_tensor,
                    tf.bool,
                    name='convert_x_discrete_to_boolean'),
                name='create_status_on_previous_tensor')
            status_change_probabilities_tensor = tf.where(
                status_on_previous_tensor,
                tf.fill(
                    x_discrete_previous_shape_tensor,
                    self.status_on_to_off_probability,
                    name='fill_on_to_off_probabilities'),
                tf.fill(
                    x_discrete_previous_shape_tensor,
                    self.status_off_to_on_probability,
                    name='fill_off_to_on_probabilities'),
                name='create_status_change_probabilities_tensor')
            status_change_distribution = tf.distributions.Bernoulli(
                probs = status_change_probabilities_tensor,
                dtype = tf.bool,
                name = 'bernoulli_distribution')
            status_changed_tensor = status_change_distribution.sample(
                name='create_status_changed_tensor')
            status_on_tensor = tf.logical_xor(
                status_on_previous_tensor,
                status_changed_tensor,
                name='create_status_on_tensor')
            x_discrete_tensor = tf.squeeze(
                tf.cast(
                    tf.logical_not(
                        status_on_tensor,
                        name='calc_status_off_tensor'),
                    tf.int32,
                    name='cast_status_off_tensor_to_int'),
                name='create_x_discrete_tensor')
            # Generate the sample of the continuous X variables
            moving_sensor_drift_tensor = tf.multiply(
                moving_sensor_drift_reference_tensor,
                tf.sqrt(
                    tf.divide(
                        t_delta_seconds_tensor,
                        reference_time_delta_seconds_tensor,
                        name='calc_ratio_of_t_delta_and_reference_time'),
                    name='take_sqrt_of_ratio'),
                name='create_moving_sensor_drift')
            room_min_scaled = tf.divide(
                tf.subtract(
                    tf.tile(
                        room_corners_tensor[0],
                        [self.sensor_variable_structure.num_moving_sensors],
                        name='tile_room_min'),
                    x_continuous_previous_tensor,
                    name='subtract_x_prev_from_room_min_tiled'),
                moving_sensor_drift_tensor,
                name='create_room_min_scaled')
            room_max_scaled = tf.divide(
                tf.subtract(
                    tf.tile(
                        room_corners_tensor[1],
                        [self.sensor_variable_structure.num_moving_sensors],
                        name='tile_room_max'),
                    x_continuous_previous_tensor,
                    name='subtract_x_prev_from_room_max_tiled'),
                moving_sensor_drift_tensor,
                name='create_room_max_scaled')
            x_continuous_tensor = tf.cast(
                tf.reshape(
                    tf.py_func(
                        stats.truncnorm.rvs,
                        [room_min_scaled,
                        room_max_scaled,
                        x_continuous_previous_tensor,
                        moving_sensor_drift_tensor],
                        tf.float64,
                        name='create_x_continuous_sample'),
                    tf.shape(x_continuous_previous_tensor),
                    name='reshape_x_continuous_sample'),
                tf.float32,
                name='create_x_continuous')
        return x_discrete_tensor, x_continuous_tensor

    def create_y_bar_x_log_pdf_tensor(
        self,
        x_discrete_tensor,
        x_continuous_tensor,
        y_discrete_tensor,
        y_continuous_tensor):
        with tf.name_scope('create_y_bar_x_log_pdf'):
            # Convert sensor model values to tensors
            rssi_untruncated_std_dev_tensor = tf.constant(
                self.rssi_untruncated_std_dev,
                dtype=tf.float32,
                name='rssi_untruncated_std_dev')
            lower_rssi_cutoff_tensor = tf.constant(
                self.lower_rssi_cutoff,
                dtype=tf.float32,
                name='lower_rssi_cutoff')
            # Calculate inter-sensor distances
            distances_tensor = self.create_distances_tensor(
                x_continuous_tensor)
            # Calculate discrete probabilities
            ping_success_probabilities_tensor, ping_failure_probabilities_tensor = self.create_ping_success_probabilities_tensor(
                x_discrete_tensor,
                distances_tensor)
            y_discrete_broadcast_tensor = tf.add(
                y_discrete_tensor,
                tf.zeros_like(
                    ping_success_probabilities_tensor,
                    tf.int32,
                    name='create_zeros_like_ping_success_probabilities'),
                name='create_y_discrete_broadcast')
            discrete_probabilities_tensor = tf.where(
                tf.equal(
                    y_discrete_broadcast_tensor,
                    tf.constant(0, dtype=tf.int32),
                    name='create_ping_success_boolean'),
                ping_success_probabilities_tensor,
                ping_failure_probabilities_tensor,
                name='create_discrete_probabilities')
            discrete_log_probabilities_tensor = tf.log(
                discrete_probabilities_tensor,
                name='create_discrete_log_probabilities')
            # Calculate continuous probability densities
            y_continuous_broadcast_tensor = tf.add(
                y_continuous_tensor,
                tf.zeros_like(
                    distances_tensor,
                    name='create_zeros_like_distances'),
                name='y_continuous_broadcast')
            rssi_untruncated_mean_tensor = self.create_rssi_untruncated_mean_tensor(distances_tensor)
            rssi_scaled_tensor = tf.divide(
                tf.subtract(
                    y_continuous_broadcast_tensor,
                    rssi_untruncated_mean_tensor,
                    name='subtract_rssi_untruncated_mean_from_y_continuous'),
                rssi_untruncated_std_dev_tensor,
                name='create_rssi_scaled')
            rssi_lower_cutoff_scaled_tensor = tf.divide(
                tf.subtract(
                    lower_rssi_cutoff_tensor,
                    rssi_untruncated_mean_tensor,
                    name='subtract_rssi_untruncated_mean_from_lower_rssi_cutoff'),
                rssi_untruncated_std_dev_tensor,
                name='create_rssi_lower_cutoff_scaled')
            untruncated_normal_distribution = tf.distributions.Normal(
                loc=tf.constant(0.0, dtype=tf.float32),
                scale = tf.constant(1.0, dtype=tf.float32),
                name='standard_normal_distribution')
            logf_untruncated_tensor = tf.subtract(
                untruncated_normal_distribution.log_prob(
                    rssi_scaled_tensor,
                    name='calc_normal_log_pdf'),
                tf.add(
                    tf.log(
                        rssi_untruncated_std_dev_tensor,
                        name='calc_log_sigma_normalization_factor'),
                    untruncated_normal_distribution.log_survival_function(
                        rssi_lower_cutoff_scaled_tensor,
                        name='calc_domain_normalization_factor'),
                    name='calc_normalization_factor'),
                name='create_logf_untruncated')
            logf_tensor = tf.where(
                tf.less(
                    y_continuous_broadcast_tensor,
                    lower_rssi_cutoff_tensor,
                    name='create_y_less_than_rssi_cutoff_boolean'),
                tf.fill(
                    tf.shape(
                        logf_untruncated_tensor,
                        name='extract_logf_untruncated_shape'),
                    np.float32(-np.inf),
                    name='create_negative_infinity_like_logf_untruncated'),
                logf_untruncated_tensor,
                name='create_logf')
            continuous_log_probability_densities_tensor = tf.where(
                tf.equal(
                    y_discrete_broadcast_tensor,
                    tf.constant(1, dtype=tf.int32),
                    name='create_ping_failure_boolean'),
                tf.zeros(
                    tf.shape(
                        logf_tensor,
                        name='extract_logf_shape'),
                    tf.float32,
                    name='create_zeros_like_logf'),
                logf_tensor,
                name='create_continuous_log_probability_densities')
            # Combine discrete probabilities and continuous probability densities
            y_bar_x_log_pdf_tensor = tf.add(
                tf.reduce_sum(
                    discrete_log_probabilities_tensor,
                    axis = -1,
                    name='sum_discrete_log_probabilities'),
                tf.reduce_sum(
                    continuous_log_probability_densities_tensor,
                    axis = -1,
                    name='sum_continuous_log_probability_densities'),
                name='create_y_bar_x_log_pdf')
        return y_bar_x_log_pdf_tensor

    def create_y_bar_x_sample_tensor(
        self,
        x_discrete_tensor,
        x_continuous_tensor):
        with tf.name_scope('create_y_bar_x_sample'):
            # Convert sensor model values to tensors
            lower_rssi_cutoff_tensor = tf.constant(
                self.lower_rssi_cutoff,
                tf.float32,
                name='lower_rssi_cutoff')
            rssi_untruncated_std_dev_tensor = tf.constant(
                self.rssi_untruncated_std_dev,
                tf.float32,
                name='rssi_untruncated_std_dev')
            # Calculate inter-sensor distances
            distances_tensor = self.create_distances_tensor(
                x_continuous_tensor)
            # Generate the sample of the discrete Y variables
            ping_success_probabilities_tensor, ping_failure_probabilities_tensor = self.create_ping_success_probabilities_tensor(
                x_discrete_tensor,
                distances_tensor)
            ping_success_logits_tensor = tf.log(
                tf.stack(
                    [ping_success_probabilities_tensor, ping_failure_probabilities_tensor],
                    axis= -1,
                    name='create_ping_success_probability_array'),
                name='create_ping_success_logits')
            ping_success_logits_reshaped_tensor = tf.reshape(
                ping_success_logits_tensor,
                [-1, self.num_y_continuous_vars, 2],
                name='create_ping_success_logits_reshaped')
            ping_success_samples_tensor = tf.cast(
                tf.squeeze(
                    tf.map_fn(
                        lambda logits_tensor: tf.transpose(
                            tf.multinomial(logits_tensor, num_samples = 1)),
                        ping_success_logits_reshaped_tensor,
                        dtype=tf.int64,
                        name='create_ping_success_samples_before_casting'),
                    name='squeeze_ping_success_samples'),
                tf.int32,
                name='create_ping_success_samples')
            y_discrete_bar_x_sample_tensor = tf.squeeze(
                ping_success_samples_tensor,
                name = 'create_y_discrete_bar_x_sample')
            # Generate the sample of the continuous Y variables
            rssi_untruncated_mean_tensor = self.create_rssi_untruncated_mean_tensor(distances_tensor)
            y_continuous_bar_x_sample_tensor = tf.cast(
                tf.squeeze(
                    tf.reshape(
                        tf.py_func(
                            stats.truncnorm.rvs,
                            [tf.divide(
                                tf.subtract(
                                    lower_rssi_cutoff_tensor,
                                    rssi_untruncated_mean_tensor,
                                    name='subtract_rssi_untruncated_mean_from_lower_cutoff'),
                                rssi_untruncated_std_dev_tensor,
                                name='divide_by_std_dev'),
                            np.float32(np.inf),
                            rssi_untruncated_mean_tensor,
                            rssi_untruncated_std_dev_tensor],
                            tf.float64,
                            name='create_rssi_samples'),
                        tf.shape(distances_tensor),
                        name='reshape_rssi_samples'),
                    name='squeeze_rssi_samples'),
                tf.float32,
                name='create_y_continuous_bar_x_sample')
        return y_discrete_bar_x_sample_tensor, y_continuous_bar_x_sample_tensor

    # These functions do not appear in the parent class. They are used by the
    # functions above

    def create_distances_tensor(self, x_continuous_tensor):
        with tf.name_scope('create_distances'):
            # Convert sensor model values to tensors
            fixed_sensor_positions_tensor = tf.constant(
                self.fixed_sensor_positions,
                dtype=tf.float32,
                name='fixed_sensor_positions')
            # Calculate matrix of inter-sensor distances
            # Add a leading dimension of size 1 if x_continuous_tensor is of
            # rank 1
            x_continuous_reshaped_tensor = tf.reshape(
                x_continuous_tensor,
                [-1, self.num_x_continuous_vars],
                name='create_x_continuous_regularized')
            # Check whether the number of x values in x_continuous_tensor is
            # known at the time of graph creation
            num_x_values_known = (x_continuous_reshaped_tensor.get_shape()[0].value is not None)
            # If the number of x values in x_continuous_tensor is known at the
            # time of graph creation, define it then. Otherwise, calculate it at
            # runtime.
            if num_x_values_known:
                num_x_values = x_continuous_reshaped_tensor.get_shape()[0].value
            else:
                num_x_values = tf.shape(x_continuous_reshaped_tensor)[0]
            moving_sensor_positions_tensor = tf.reshape(
                x_continuous_reshaped_tensor,
                [-1, self.sensor_variable_structure.num_moving_sensors, self.sensor_variable_structure.num_dimensions],
                name='create_moving_sensor_positions')
            # Broadcast fixed_sensor_positions_tensor to the shape we want by
            # constructing a tensor of zeros in the shape we want an adding
            fixed_sensor_positions_broadcast_tensor = tf.add(
                fixed_sensor_positions_tensor,
                tf.zeros(
                    [num_x_values, self.sensor_variable_structure.num_fixed_sensors, self.sensor_variable_structure.num_dimensions],
                    dtype=tf.float32,
                    name='create_fixed_sensor_positions_broadcast_template'),
                name='create_fixed_sensor_positions_broadcast')
            all_sensor_positions_tensor = tf.concat(
                [moving_sensor_positions_tensor, fixed_sensor_positions_broadcast_tensor],
                axis = -2,
                name='create_all_sensor_positions')
            distance_matrix_tensor = tf.norm(
                tf.subtract(
                    tf.expand_dims(
                        all_sensor_positions_tensor,
                        axis = -3,
                        name='create_distance_matrix_copy_1'),
                    tf.expand_dims(
                        all_sensor_positions_tensor,
                        axis=-2,
                        name='create_distance_matrix_copy_2'),
                    name='subtract_sensor_positions_from_themselves'),
                axis=-1,
                name='create_distance_matrix')
            # Extract and flatten the inter-sensor distances that correspond to
            # continuous Y variables
            distance_matrix_rolled_forward_tensor = tf.transpose(
                distance_matrix_tensor,
                [1, 2, 0],
                name='create_distance_matrix_rolled_forward')
            if num_x_values_known:
                distances_rolled_forward_tensor = tf.boolean_mask(
                    distance_matrix_rolled_forward_tensor,
                    self.sensor_variable_structure.extract_y_variables_mask,
                    name='create_distances_rolled_forward')
                distances_rolled_forward_tensor.set_shape([self.num_y_continuous_vars, num_x_values])
            else:
                distances_rolled_forward_unshaped_tensor = tf.boolean_mask(
                    distance_matrix_rolled_forward_tensor,
                    self.sensor_variable_structure.extract_y_variables_mask,
                    name='extract_and_flatten_distances')
                distances_rolled_forward_tensor = tf.reshape(
                    distances_rolled_forward_unshaped_tensor,
                    [self.num_y_continuous_vars, num_x_values],
                    name='create_distances_rolled_forward')
            distances_tensor = tf.squeeze(
                tf.transpose(
                    distances_rolled_forward_tensor,
                    [1, 0],
                    name='roll_back_indices'),
                name='create_distances')
        return distances_tensor

    def create_ping_success_probabilities_tensor(
        self,
        x_discrete_tensor,
        distances_tensor):
        with tf.name_scope('create_ping_success_probabilities'):
            ping_success_probabilities_from_distances_tensor = self.create_ping_success_probabilities_from_distances_tensor(
                distances_tensor)
            ping_success_probabilities_from_sensor_statuses_tensor = self.create_ping_success_probabilities_from_sensor_statuses_tensor(
                x_discrete_tensor)
            ping_success_probabilities_tensor = tf.multiply(
                ping_success_probabilities_from_distances_tensor,
                ping_success_probabilities_from_sensor_statuses_tensor,
                name='create_ping_success_probabilities')
            ping_failure_probabilities_tensor = tf.subtract(
                tf.constant(1.0, dtype=tf.float32),
                ping_success_probabilities_tensor,
                name='create_ping_failure_probabilities')
            return ping_success_probabilities_tensor, ping_failure_probabilities_tensor

    def create_ping_success_probabilities_from_sensor_statuses_tensor(
        self,
        x_discrete_tensor):
        with tf.name_scope('create_ping_success_probabilities_from_sensor_statuses'):
            # Convert sensor model values to tensors
            ping_success_probability_sensor_on_tensor = tf.constant(
                self.ping_success_probability_sensor_on,
                tf.float32,
                name='ping_success_probability_sensor_on')
            ping_success_probability_sensor_off_tensor = tf.constant(
                self.ping_success_probability_sensor_off,
                tf.float32,
                name='ping_success_probability_sensor_on')
            # Add a leading dimension of size 1 if x_discrete_tensor is of
            # rank 1
            x_discrete_reshaped_tensor = tf.reshape(
                x_discrete_tensor,
                [-1, self.num_x_discrete_vars],
                name='create_x_continuous_reshaped')
            # Check whether the number of x values in x_discrete_tensor is
            # known at the time of graph creation
            num_x_values_known = (x_discrete_reshaped_tensor.get_shape()[0].value is not None)
            # If the number of x values in x_discrete_tensor is known at the
            # time of graph creation, define it then. Otherwise, calculate it at
            # runtime.
            if num_x_values_known:
                num_x_values = x_discrete_reshaped_tensor.get_shape()[0].value
            else:
                num_x_values = tf.shape(x_discrete_reshaped_tensor)[0]
            sensors_on_tensor = tf.logical_not(
                tf.cast(
                    x_discrete_reshaped_tensor,
                    tf.bool,
                    name='convert_x_discrete_regularized_to_boolean'),
                name='create_sensors_on')
            sensor_pairs_on_matrix_tensor = tf.logical_and(
                tf.expand_dims(
                    sensors_on_tensor,
                    axis = -2,
                    name='sensor_pair_on_matrix_copy_1'),
                tf.expand_dims(
                    sensors_on_tensor,
                    axis=-1,
                    name='sensor_pairs_on_matrix_copy_2'),
                name='create_sensor_pairs_on_matrix')
            # Extract and flatten the sensor pair statuses that correspond to Y
            # variables
            sensor_pairs_on_matrix_rolled_forward_tensor = tf.transpose(
                sensor_pairs_on_matrix_tensor,
                [1, 2, 0],
                name='create_sensor_pairs_on_matrix_rolled_forward')
            if num_x_values_known:
                sensor_pairs_on_rolled_forward_tensor = tf.boolean_mask(
                    sensor_pairs_on_matrix_rolled_forward_tensor,
                    self.sensor_variable_structure.extract_y_variables_mask,
                    name='extract_and_flatten_sensor_pairs_on')
                sensor_pairs_on_rolled_forward_tensor.set_shape([self.num_y_discrete_vars, num_x_values])
            else:
                sensor_pairs_on_rolled_forward_unshaped_tensor = tf.boolean_mask(
                    sensor_pairs_on_matrix_rolled_forward_tensor,
                    self.sensor_variable_structure.extract_y_variables_mask,
                    name='extract_and_flatten_sensor_pairs_on')
                sensor_pairs_on_rolled_forward_tensor = tf.reshape(
                    sensor_pairs_on_rolled_forward_unshaped_tensor,
                    [self.num_y_discrete_vars, num_x_values],
                    name='create_sensor_pairs_on_rolled_forward')
            sensor_pairs_on_tensor = tf.transpose(
                sensor_pairs_on_rolled_forward_tensor,
                [1, 0],
                name='create_sensor_pairs_on')
            ping_success_probabilities_from_sensor_statuses_reshaped_tensor = tf.where(
                sensor_pairs_on_tensor,
                tf.fill(
                    [num_x_values, self.num_y_discrete_vars],
                    ping_success_probability_sensor_on_tensor,
                    name='fill_ping_success_probability_sensor_on'),
                tf.fill(
                    [num_x_values, self.num_y_discrete_vars],
                    ping_success_probability_sensor_off_tensor,
                    name='fill_ping_success_probability_sensor_off'),
                name='create_ping_success_probabilities_from_sensor_statuses_reshaped')
            ping_success_probabilities_from_sensor_statuses_tensor = tf.squeeze(
                ping_success_probabilities_from_sensor_statuses_reshaped_tensor,
                name='create_ping_success_probabilities_from_sensor_statuses')
        return ping_success_probabilities_from_sensor_statuses_tensor

    def create_ping_success_probabilities_from_distances_tensor(
        self,
        distances_tensor):
        with tf.name_scope('create_ping_success_probabilities_from_distances'):
            # Convert sensor model values to tensors
            ping_success_probability_zero_distance_tensor = tf.constant(
                self.ping_success_probability_zero_distance,
                dtype=tf.float32,
                name='ping_success_probability_zero_distance')
            scale_factor_tensor = tf.constant(
                self.scale_factor,
                dtype=tf.float32,
                name='scale_factor')
            # Calculate ping success probabilties
            ping_success_probabilities_from_distances_tensor = tf.multiply(
                ping_success_probability_zero_distance_tensor,
                tf.exp(
                    tf.divide(
                        distances_tensor,
                        scale_factor_tensor,
                        name='divide_distances_by_scale_factor'),
                    name='calc_ping_success_probability_scale_factor'),
                name='create_ping_success_probabilities_from_distances')
        return ping_success_probabilities_from_distances_tensor

    def create_rssi_untruncated_mean_tensor(
        self,
        distances_tensor):
        with tf.name_scope('create_rssi_untruncated_mean'):
            # Convert sensor model values to tensors
            rssi_untruncated_mean_intercept_tensor = tf.constant(
                self.rssi_untruncated_mean_intercept,
                dtype=tf.float32,
                name='rssi_untruncated_mean_intercept')
            rssi_untruncated_mean_slope_tensor = tf.constant(
                self.rssi_untruncated_mean_slope,
                dtype=tf.float32,
                name='rssi_untruncated_mean_slope')
            # Calculate untruncted means of RSSI distributions
            rssi_untruncated_mean_tensor = tf.add(
                rssi_untruncated_mean_intercept_tensor,
                tf.multiply(
                    rssi_untruncated_mean_slope_tensor,
                    tf.divide(
                        tf.log(distances_tensor),
                        tf.constant(np.log(10.0), dtype=tf.float32),
                        name='calc_log_10_distances'),
                    name='multiply_log_distances_by_slope'),
                name='create_rssi_untruncated_mean')
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

    def distances_test(
        self,
        x_continuous_test_input):
        return self.sensor_model_testing_session.run(
            self.distances_test_output_tensor,
            feed_dict = {
                self.x_continuous_test_input_tensor: x_continuous_test_input})

    def ping_success_probabilities_test(
        self,
        x_discrete_test_input,
        distances_test_input):
        return self.sensor_model_testing_session.run(
            [self.ping_success_probabilities_test_output_tensor, self.ping_failure_probabilities_test_output_tensor],
            feed_dict = {
                self.x_discrete_test_input_tensor: x_discrete_test_input,
                self.distances_test_input_tensor: distances_test_input})

    def ping_success_probabilities_from_sensor_statuses_test(
        self,
        x_discrete_test_input):
        return self.sensor_model_testing_session.run(
            self.ping_success_probabilities_from_sensor_statuses_test_output_tensor,
            feed_dict = {
                self.x_discrete_test_input_tensor: x_discrete_test_input})

    def ping_success_probabilities_from_distances_test(self, distances_test_input):
        return self.sensor_model_testing_session.run(
            self.ping_success_probabilities_from_distances_test_output_tensor,
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

    def y_bar_x_sample_test(
        self,
        x_discrete_test_input,
        x_continuous_test_input):
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
