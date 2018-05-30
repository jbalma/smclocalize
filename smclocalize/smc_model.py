import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

# Define a class for a generic sequential Monte Carlo (AKA state space) model
class SMCModel(object):
    def __init__(
        self,
        num_x_discrete_vars,
        num_x_continuous_vars,
        num_y_discrete_vars,
        num_y_continuous_vars,
        num_particles = 10000):
        self.num_x_discrete_vars = num_x_discrete_vars
        self.num_x_continuous_vars = num_x_continuous_vars
        self.num_y_discrete_vars = num_y_discrete_vars
        self.num_y_continuous_vars = num_y_continuous_vars
        self.num_particles = num_particles
        # Generate computational graph to support the functions below
        self.smc_model_graph = tf.Graph()
        with self.smc_model_graph.as_default():
            # Generate computational graph to support generate_initial_particles()
            self.y_discrete_initial_tensor = tf.placeholder(
                tf.int32,
                shape = (self.num_y_discrete_vars),
                name='y_discrete_initial_input')
            self.y_continuous_initial_tensor = tf.placeholder(
                tf.float32,
                shape = (self.num_y_continuous_vars),
                name='y_continuous_initial_input')
            self.x_discrete_initial_tensor, self.x_continuous_initial_tensor, self.log_weights_initial_tensor = self.create_initial_particles_tensor(
                self.y_discrete_initial_tensor,
                self.y_continuous_initial_tensor)
            # Generate computational graph to support generate_next_particles()
            self.x_discrete_previous_tensor = tf.placeholder(
                tf.float32,
                shape = (self.num_particles, self.num_x_discrete_vars),
                name='x_discrete_previous_input')
            self.x_continuous_previous_tensor = tf.placeholder(
                tf.float32,
                shape = (self.num_particles, self.num_x_continuous_vars),
                name='x_continuous_previous_input')
            self.ancestor_indices_tensor = tf.placeholder(
                tf.int32,
                shape = (self.num_particles),
                name='ancestor_indices_input')
            self.t_delta_seconds_tensor = tf.placeholder(
                tf.float32,
                shape = (),
                name='t_delta_seconds_input')
            self.y_discrete_tensor = tf.placeholder(
                tf.int32,
                shape = (self.num_y_discrete_vars),
                name='y_discrete_input')
            self.y_continuous_tensor = tf.placeholder(
                tf.float32,
                shape = (self.num_y_continuous_vars),
                name='y_continuous_input')
            self.x_discrete_tensor, self.x_continuous_tensor, self.log_weights_tensor = self.create_next_particles_tensor(
                self.x_discrete_previous_tensor,
                self.x_continuous_previous_tensor,
                self.ancestor_indices_tensor,
                self.y_discrete_tensor,
                self.y_continuous_tensor,
                self.t_delta_seconds_tensor)
            # Generate computational graph to support generate_initial_simulation_timestep()
            self.x_discrete_initial_sim_tensor, self.x_continuous_initial_sim_tensor, self.y_discrete_initial_sim_tensor, self.y_continuous_initial_sim_tensor = self.create_initial_simulation_timestep_tensor()
            # Generate computational graph to support generate_next_simulation_timestep()
            self.x_discrete_previous_sim_tensor = tf.placeholder(
                tf.int32,
                shape = (self.num_x_discrete_vars),
                name='x_discrete_previous_sim_input')
            self.x_continuous_previous_sim_tensor = tf.placeholder(
                tf.float32,
                shape = (self.num_x_continuous_vars),
                name='x_continuous_previous_sim_input')
            self.t_delta_seconds_sim_tensor = tf.placeholder(
                tf.float32,
                shape = (),
                name='t_delta_seconds_sim_input')
            self.x_discrete_sim_tensor, self.x_continuous_sim_tensor, self.y_discrete_sim_tensor, self.y_continuous_sim_tensor = self.create_next_simulation_timestep_tensor(
                self.x_discrete_previous_sim_tensor,
                self.x_continuous_previous_sim_tensor,
                self.t_delta_seconds_sim_tensor)
        # Initialize session with this graph
        self.smc_model_session = tf.Session(graph = self.smc_model_graph)

        # Create a debugging version of this session which invokes tfdbg
        self.smc_model_session_debug = tf_debug.LocalCLIDebugWrapperSession(self.smc_model_session)

    # Functions which should come from the child class. These specify the
    # probability distributions which define the specific model we're working
    # with

    def create_x_initial_sample_tensor(
        self,
        num_samples_tensor = tf.constant(1, tf.int32)):
        raise Exception('create_x_initial_sample_tensor() called from parent SMCModel class. Should be overriden by child class.')

    def create_x_bar_x_previous_sample_tensor(
    self,
    x_discrete_previous_tensor,
    x_continuous_previous_tensor,
    t_delta_seconds_tensor):
        raise Exception('create_x_bar_x_previous_sample_tensor() called from parent SMCModel class. Should be overriden by child class.')

    def create_y_bar_x_log_pdf_tensor(
        self,
        x_discrete_tensor,
        x_continuous_tensor,
        y_discrete_tensor,
        y_continuous_tensor,
        num_x_values_tensor):
        raise Exception('create_y_bar_x_log_pdf_tensor() called from parent SMCModel class. Should be overriden by child class.')

    def create_y_bar_x_sample_tensor(
        self,
        x_discrete_tensor,
        x_continuous_tensor,
        num_x_values_tensor):
        raise Exception('create_y_bar_x_sample_tensor() called from parent SMCModel class. Should be overriden by child class.')

    # Functions which are used in building the computational graph above

    def create_initial_particles_tensor(
        self,
        y_discrete_initial_tensor,
        y_continuous_initial_tensor):
        with tf.name_scope('create_initial_particles'):
            num_particles_tensor = tf.constant(
                self.num_particles,
                tf.int32,
                name='num_particles')
            x_discrete_initial_tensor, x_continuous_initial_tensor = self.create_x_initial_sample_tensor(
                num_particles_tensor)
            log_weights_unnormalized_initial_tensor = self.create_y_bar_x_log_pdf_tensor(
                x_discrete_initial_tensor,
                x_continuous_initial_tensor,
                y_discrete_initial_tensor,
                y_continuous_initial_tensor)
            log_weights_initial_tensor = self.create_normalized_log_weights_tensor(
                log_weights_unnormalized_initial_tensor)
            x_discrete_initial_reshaped_tensor = tf.reshape(
                x_discrete_initial_tensor,
                [self.num_particles, self.num_x_discrete_vars],
                name='create_x_discrete_initial_reshaped')
            x_continuous_initial_reshaped_tensor = tf.reshape(
                x_continuous_initial_tensor,
                [self.num_particles, self.num_x_continuous_vars],
                name='create_x_continuous_initial_reshaped')
            log_weights_initial_reshaped_tensor = tf.reshape(
                log_weights_initial_tensor,
                [self.num_particles],
                name='create_log_weights_initial_reshaped')
        return x_discrete_initial_reshaped_tensor, x_continuous_initial_reshaped_tensor, log_weights_initial_reshaped_tensor

    def create_next_particles_tensor(
        self,
        x_discrete_previous_tensor,
        x_continuous_previous_tensor,
        ancestor_indices_tensor,
        y_discrete_tensor,
        y_continuous_tensor,
        t_delta_seconds_tensor):
        with tf.name_scope('create_next_particles'):
            x_discrete_previous_resampled_tensor, x_continuous_previous_resampled_tensor = self.create_resampled_particles_tensor(
                x_discrete_previous_tensor,
                x_continuous_previous_tensor,
                ancestor_indices_tensor)
            x_discrete_tensor, x_continuous_tensor = self.create_x_bar_x_previous_sample_tensor(
                x_discrete_previous_resampled_tensor,
                x_continuous_previous_resampled_tensor,
                t_delta_seconds_tensor)
            log_weights_unnormalized_tensor = self.create_y_bar_x_log_pdf_tensor(
                x_discrete_tensor,
                x_continuous_tensor,
                y_discrete_tensor,
                y_continuous_tensor)
            log_weights_tensor = self.create_normalized_log_weights_tensor(
                log_weights_unnormalized_tensor)
            ancestor_indices_reshaped_tensor = tf.reshape(
                ancestor_indices_tensor,
                [self.num_particles],
                name='create_ancestor_indices_reshaped')
            x_discrete_reshaped_tensor = tf.reshape(
                x_discrete_tensor,
                [self.num_particles, self.num_x_discrete_vars],
                name='create_x_discrete_reshaped')
            x_continuous_reshaped_tensor = tf.reshape(
                x_continuous_tensor,
                [self.num_particles, self.num_x_continuous_vars],
                name='create_x_continuous_reshaped')
            log_weights_unnormalized_reshaped_tensor = tf.reshape(
                log_weights_unnormalized_tensor,
                [self.num_particles],
                name='create_log_weights_reshaped')
            log_weights_reshaped_tensor = tf.reshape(
                log_weights_tensor,
                [self.num_particles],
                name='create_log_weights_reshaped')
        return x_discrete_reshaped_tensor, x_continuous_reshaped_tensor, log_weights_reshaped_tensor

    def create_initial_simulation_timestep_tensor(
        self):
        with tf.name_scope('create_initial_simulation_timestep'):
            num_samples_tensor = tf.constant(
                1,
                tf.int32,
                name='num_samples')
            x_discrete_initial_tensor, x_continuous_initial_tensor = self.create_x_initial_sample_tensor(
                num_samples_tensor)
            y_discrete_initial_tensor, y_continuous_initial_tensor = self.create_y_bar_x_sample_tensor(
                x_discrete_initial_tensor,
                x_continuous_initial_tensor)
            x_discrete_initial_reshaped_tensor = tf.reshape(
                x_discrete_initial_tensor,
                [self.num_x_discrete_vars],
                name='create_x_discrete_initial_reshaped')
            x_continuous_initial_reshaped_tensor = tf.reshape(
                x_continuous_initial_tensor,
                [self.num_x_continuous_vars],
                name='create_x_continuous_initial_reshaped')
            y_discrete_initial_reshaped_tensor = tf.reshape(
                y_discrete_initial_tensor,
                [self.num_y_discrete_vars],
                name='create_y_discrete_initial_reshaped')
            y_continuous_initial_reshaped_tensor = tf.reshape(
                y_continuous_initial_tensor,
                [self.num_y_continuous_vars],
                name='create_y_continuous_initial_reshaped')
        return x_discrete_initial_reshaped_tensor, x_continuous_initial_reshaped_tensor, y_discrete_initial_reshaped_tensor, y_continuous_initial_reshaped_tensor

    def create_next_simulation_timestep_tensor(
        self,
        x_discrete_previous_tensor,
        x_continuous_previous_tensor,
        t_delta_seconds_tensor):
        with tf.name_scope('create_next_simulation_timestep'):
            x_discrete_tensor, x_continuous_tensor = self.create_x_bar_x_previous_sample_tensor(
                x_discrete_previous_tensor,
                x_continuous_previous_tensor,
                t_delta_seconds_tensor)
            y_discrete_tensor, y_continuous_tensor = self.create_y_bar_x_sample_tensor(
                x_discrete_tensor,
                x_continuous_tensor)
            x_discrete_reshaped_tensor = tf.reshape(
                x_discrete_tensor,
                [self.num_x_discrete_vars],
                name='create_x_discrete_reshaped')
            x_continuous_reshaped_tensor = tf.reshape(
                x_continuous_tensor,
                [self.num_x_continuous_vars],
                name='create_x_continuous_reshaped')
            y_discrete_reshaped_tensor = tf.reshape(
                y_discrete_tensor,
                [self.num_y_discrete_vars],
                name='create_y_discrete_reshaped')
            y_continuous_reshaped_tensor = tf.reshape(
                y_continuous_tensor,
                [self.num_y_continuous_vars],
                name='create_y_continuous_reshaped')
        return x_discrete_reshaped_tensor, x_continuous_reshaped_tensor, y_discrete_reshaped_tensor, y_continuous_reshaped_tensor

    # We are not currently using this function because tf.multinomial was
    # causing memory overflows, but we're keeping it here in case we need it
    # again
    def create_ancestor_indices_tensor(self, log_weights_previous_tensor):
        with tf.name_scope('create_ancestor_indices'):
            ancestor_indices_tensor = tf.multinomial(
                [log_weights_previous_tensor],
                tf.shape(log_weights_previous_tensor)[0],
                name='create_ancestor_indices')
            ancestor_indices_tensor.set_shape([1, self.num_particles])
            ancestor_indices_reshaped_tensor = tf.reshape(
                ancestor_indices_tensor,
                [self.num_particles],
                name='ancestor_indices_reshaped')
        return ancestor_indices_reshaped_tensor

    def create_resampled_particles_tensor(
        self,
        x_discrete_tensor,
        x_continuous_tensor,
        ancestor_indices_tensor):
        with tf.name_scope('create_resampled_particles'):
            x_discrete_resampled_tensor = tf.gather(
                x_discrete_tensor,
                ancestor_indices_tensor,
                name='create_x_discrete_resampled')
            x_continuous_resampled_tensor = tf.gather(
                x_continuous_tensor,
                ancestor_indices_tensor,
                name='create_x_continuous_resampled')
            x_discrete_resampled_reshaped_tensor = tf.reshape(
                x_discrete_resampled_tensor,
                [self.num_particles, self.num_x_discrete_vars],
                name='create_x_discrete_resampled_reshaped')
            x_continuous_resampled_reshaped_tensor = tf.reshape(
                x_continuous_resampled_tensor,
                [self.num_particles, self.num_x_continuous_vars],
                name='create_x_continuous_resampled_reshaped')
        return x_discrete_resampled_reshaped_tensor, x_continuous_resampled_reshaped_tensor

    def create_normalized_log_weights_tensor(self, log_weights_unnormalized_tensor):
        with tf.name_scope('create_normalized_log_weights'):
            log_weights_tensor = tf.subtract(
                log_weights_unnormalized_tensor,
                tf.reduce_logsumexp(
                    log_weights_unnormalized_tensor,
                    name='calculate_normalization_factor'),
                name='create_log_weights')
            log_weights_reshaped_tensor = tf.reshape(
                log_weights_tensor,
                [self.num_particles],
                name='create_log_weights_reshaped')
        return log_weights_reshaped_tensor


    # Functions which provide the interface to this class. These functions run
    # portions of the computational graph above to generate results

    # Generate an initial set of X particles based on an initial Y value
    def generate_initial_particles(
        self,
        y_discrete_initial,
        y_continuous_initial):
        if np.any(np.isnan(y_discrete_initial)):
            raise Exception('Some initial y discrete values are NaN')
        if np.any(np.isnan(y_continuous_initial)):
            raise Exception('Some initial y continuous values are NaN')
        y_discrete_initial_reshaped = np.reshape(
            y_discrete_initial,
            (self.num_y_discrete_vars))
        y_continuous_initial_reshaped = np.reshape(
            y_continuous_initial,
            (self.num_y_continuous_vars))
        x_discrete_initial, x_continuous_initial, log_weights_initial =  self.smc_model_session.run(
            [self.x_discrete_initial_tensor, self.x_continuous_initial_tensor, self.log_weights_initial_tensor],
            feed_dict = {
                self.y_discrete_initial_tensor: y_discrete_initial_reshaped,
                self.y_continuous_initial_tensor : y_continuous_initial_reshaped})
        # x_discrete_initial, x_continuous_initial, log_weights_initial =  self.smc_model_session_debug.run(
        #     [self.x_discrete_initial_tensor, self.x_continuous_initial_tensor, self.log_weights_initial_tensor],
        #     feed_dict = {
        #         self.y_discrete_initial_tensor: y_discrete_initial_reshaped,
        #         self.y_continuous_initial_tensor : y_continuous_initial_reshaped})
        if np.any(np.isnan(x_discrete_initial)):
            raise Exception('Some initial x discrete values are NaN')
        if np.any(np.isnan(x_continuous_initial)):
            raise Exception('Some initial x continuous values are NaN')
        if np.any(np.isnan(log_weights_initial)):
            raise Exception('Some initial log weights are NaN')
        if np.all(np.isneginf(log_weights_initial)):
            raise Exception('All initial log weights are negative infinite')
        return x_discrete_initial, x_continuous_initial, log_weights_initial

    # Generate a set of X particles and weights (along with indices which point
    # to their ancestors) for the current time step based on the set of
    # particles and weights from the previous time step and a Y value from the
    # current time step
    def generate_next_particles(
        self,
        x_discrete_previous,
        x_continuous_previous,
        log_weights_previous,
        y_discrete,
        y_continuous,
        t_delta):
        if np.any(np.isnan(x_discrete_previous)):
            raise Exception('Some previous x discrete values are NaN')
        if np.any(np.isnan(x_continuous_previous)):
            raise Exception('Some previous x continuous values are NaN')
        if np.any(np.isnan(log_weights_previous)):
            raise Exception('Some previous log weights are NaN')
        if np.any(np.isnan(y_discrete)):
            raise Exception('Some y discrete values are NaN')
        if np.any(np.isnan(y_continuous)):
            raise Exception('Some y continuous values are NaN')
        if np.all(np.isneginf(log_weights_previous)):
            raise Exception('All previous log weights are negative infinite')
        if not t_delta > np.timedelta64(0, 's'):
            raise Exception('Time delta is not greater than zero')
        x_discrete_previous_reshaped = np.reshape(
            x_discrete_previous,
            (self.num_particles, self.num_x_discrete_vars))
        x_continuous_previous_reshaped = np.reshape(
            x_continuous_previous,
            (self.num_particles, self.num_x_continuous_vars))
        log_weights_previous_reshaped = np.reshape(
            log_weights_previous,
            (self.num_particles))
        y_discrete_reshaped = np.reshape(
            y_discrete,
            (self.num_y_discrete_vars))
        y_continuous_reshaped = np.reshape(
            y_continuous,
            (self.num_y_continuous_vars))
        t_delta_seconds = t_delta/np.timedelta64(1, 's')
        # Calculate ancestor indices outside of TensorFlow because
        # tf.multinomial() uses too much memory for logit arrays
        weights_previous = np.exp(log_weights_previous_reshaped)
        weights_previous_sum = np.sum(weights_previous)
        if weights_previous_sum == 0:
            raise Exception('Sum of previous weights is zero')
        weights_previous_normalized = weights_previous/weights_previous_sum
        if np.any(np.isnan(weights_previous_normalized)):
            raise Exception('After normalization, some previous weights are NaN')
        ancestor_indices = np.int32(np.random.choice(
            self.num_particles,
            size=self.num_particles,
            p=weights_previous_normalized))
        if np.any(np.isnan(ancestor_indices)):
            raise Exception('Some ancestor indices are NaN')
        x_discrete, x_continuous, log_weights =  self.smc_model_session.run(
            [self.x_discrete_tensor, self.x_continuous_tensor, self.log_weights_tensor],
            feed_dict = {
                self.x_discrete_previous_tensor: x_discrete_previous_reshaped,
                self.x_continuous_previous_tensor : x_continuous_previous_reshaped,
                self.ancestor_indices_tensor: ancestor_indices,
                self.y_discrete_tensor: y_discrete_reshaped,
                self.y_continuous_tensor: y_continuous_reshaped,
                self.t_delta_seconds_tensor: t_delta_seconds})
        if np.any(np.isnan(x_discrete)):
            raise Exception('Some x discrete values are NaN')
        if np.any(np.isnan(x_continuous)):
            raise Exception('Some x continuous values are NaN')
        if np.any(np.isnan(log_weights)):
            print("Dumping input and output objects...")
            with open('error_dump.pkl', 'w+b') as file_handle:
                pickle.dump(
                    {'x_discrete_previous': x_discrete_previous,
                    'x_continuous_previous': x_continuous_previous,
                    'log_weights_previous': log_weights_previous,
                    'y_discrete': y_discrete,
                    'y_continuous': y_continuous,
                    't_delta': t_delta,
                    'x_discrete': x_discrete,
                    'x_continuous': x_continuous,
                    'log_weights': log_weights},
                    file_handle)
            raise Exception('Some log weights are NaN')
        if np.all(np.isneginf(log_weights)):
            raise Exception('All log weights are negative infinite')
        return x_discrete, x_continuous, log_weights, ancestor_indices

    # Generate an entire trajectory of X particles along with their weights and
    # ancestor indices based on an entire trajectory of Y values
    def generate_particle_trajectory(
        self,
        t_trajectory,
        y_discrete_trajectory,
        y_continuous_trajectory):
        t_trajectory_reshaped = np.reshape(
            t_trajectory,
            (-1))
        num_timesteps = len(t_trajectory)
        y_discrete_trajectory_reshaped = np.reshape(
            y_discrete_trajectory,
            (self.num_timesteps, self.num_y_discrete_vars))
        y_continuous_trajectory_reshaped = np.reshape(
            y_continuous_trajectory
            (self.num_timesteps, self.num_y_continuous_vars))
        x_discrete_particles_trajectory = np.zeros(
            (num_timesteps, self.num_particles, self.num_x_discrete_vars),
            dtype='int')
        x_continuous_particles_trajectory = np.zeros(
            (num_timesteps, self.num_particles, self.num_x_continuous_vars),
            dtype='float')
        log_weights_trajectory = np.zeros(
            (num_timesteps, self.num_particles),
            dtype='float')
        ancestor_indices_trajectory = np.zeros(
            (num_timesteps, self.num_particles),
            dtype='int')
        x_discrete_particles_trajectory[0], x_continuous_particles_trajectory[0], log_weights_trajectory[0] = self.generate_initial_particles(
            y_discrete_trajectory_reshaped[0],
            y_continuous_trajectory_reshaped[0])
        for i in range(1, num_timesteps):
            x_discrete_particles_trajectory[i], x_continuous_particles_trajectory[i], log_weights_trajectory[i], ancestor_indices_trajectory[i] = self.generate_next_particles(
                x_discrete_particles_trajectory[i - 1],
                x_continuous_particles_trajectory[i - 1],
                log_weights_trajectory[i - 1],
                y_discrete_trajectory_reshaped[i],
                y_continuous_trajectory_reshaped[i],
                t_trajectory_reshaped[i] - t_trajectory_reshaped[i - 1])
        return x_discrete_particles_trajectory, x_continuous_particles_trajectory, log_weights_trajectory, ancestor_indices_trajectory

    # Generate the initial time step of a simulated X and Y data set
    def generate_initial_simulation_timestep(
        self):
        return self.smc_model_session.run(
            [self.x_discrete_initial_sim_tensor, self.x_continuous_initial_sim_tensor, self.y_discrete_initial_sim_tensor, self.y_continuous_initial_sim_tensor])

    # Generate a single time step of simulated X and Y data based on the
    # previous time step and a time delta
    def generate_next_simulation_timestep(
        self,
        x_discrete_previous,
        x_continuous_previous,
        t_delta):
        x_discrete_previous_reshaped = np.reshape(
            x_discrete_previous,
            (self.num_x_discrete_vars))
        x_continuous_previous_reshaped = np.reshape(
            x_continuous_previous,
            (self.num_x_continuous_vars))
        t_delta_seconds = t_delta/np.timedelta64(1, 's')
        return self.smc_model_session.run(
            [self.x_discrete_sim_tensor, self.x_continuous_sim_tensor, self.y_discrete_sim_tensor, self.y_continuous_sim_tensor],
            feed_dict = {
                self.x_discrete_previous_sim_tensor: x_discrete_previous_reshaped,
                self.x_continuous_previous_sim_tensor: x_continuous_previous_reshaped,
                self.t_delta_seconds_sim_tensor: t_delta_seconds})

    # Generate an entire trajectory of simulated X and Y data
    def generate_simulation_trajectory(
        self,
        t_trajectory):
        t_trajectory_reshaped = np.reshape(
            t_trajectory,
            (-1))
        num_timesteps_trajectory = len(t_trajectory_reshaped)
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
                t_trajectory_reshaped[t_index] - t_trajectory_reshaped[t_index - 1])
        return x_discrete_trajectory, x_continuous_trajectory, y_discrete_trajectory, y_continuous_trajectory

    # These functions are just used for testing the functions above

    def write_computational_graph(
        self,
        tensorflow_log_directory):
        log_file_writer = tf.summary.FileWriter(
            tensorflow_log_directory,
            self.smc_model_graph)
        log_file_writer.close()

    def generate_next_particles_profile(
        self,
        x_discrete_previous,
        x_continuous_previous,
        log_weights_previous,
        y_discrete,
        y_continuous,
        t_delta,
        tensorflow_log_directory):
        x_discrete_previous_reshaped = np.reshape(
            x_discrete_previous,
            (self.num_particles, self.num_x_discrete_vars))
        x_continuous_previous_reshaped = np.reshape(
            x_continuous_previous,
            (self.num_particles, self.num_x_continuous_vars))
        log_weights_previous_reshaped = np.reshape(
            log_weights_previous,
            (self.num_particles))
        y_discrete_reshaped = np.reshape(
            y_discrete,
            (self.num_y_discrete_vars))
        y_continuous_reshaped = np.reshape(
            y_continuous,
            (self.num_y_continuous_vars))
        t_delta_seconds = t_delta/np.timedelta64(1, 's')
        # Calculate ancestor indices outside of TensorFlow because
        # tf.multinomial() uses too much memory for logit arrays
        weights_previous = np.exp(log_weights_previous_reshaped)
        weights_previous_normalized = weights_previous/np.sum(weights_previous)
        ancestor_indices = np.int32(np.random.choice(
            self.num_particles,
            size=self.num_particles,
            p=weights_previous_normalized))
        log_file_writer = tf.summary.FileWriter(
            tensorflow_log_directory,
            self.smc_model_graph)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        x_discrete, x_continuous, log_weights =  self.smc_model_session.run(
            [self.x_discrete_tensor, self.x_continuous_tensor, self.log_weights_tensor],
            feed_dict = {
                self.x_discrete_previous_tensor: x_discrete_previous_reshaped,
                self.x_continuous_previous_tensor : x_continuous_previous_reshaped,
                self.ancestor_indices_tensor: ancestor_indices,
                self.y_discrete_tensor: y_discrete_reshaped,
                self.y_continuous_tensor: y_continuous_reshaped,
                self.t_delta_seconds_tensor: t_delta_seconds},
            options=run_options,
            run_metadata=run_metadata)
        log_file_writer.add_run_metadata(run_metadata, 'generate_next_particles_profile')
        log_file_writer.close()
        return x_discrete, x_continuous, log_weights, ancestor_indices
