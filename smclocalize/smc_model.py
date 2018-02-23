import numpy as np
import tensorflow as tf

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
            self.y_discrete_initial_tensor = tf.placeholder(tf.int32, shape = (self.num_y_discrete_vars))
            self.y_continuous_initial_tensor = tf.placeholder(tf.float32, shape = (self.num_y_continuous_vars))
            self.x_discrete_initial_tensor, self.x_continuous_initial_tensor, self.log_weights_initial_tensor = self.create_initial_particles_tensor(
                self.y_discrete_initial_tensor,
                self.y_continuous_initial_tensor)
            # Generate computational graph to support generate_next_particles()
            self.x_discrete_previous_tensor = tf.placeholder(tf.float32, shape = (self.num_particles, self.num_x_discrete_vars))
            self.x_continuous_previous_tensor = tf.placeholder(tf.float32, shape = (self.num_particles, self.num_x_continuous_vars))
            self.log_weights_previous_tensor = tf.placeholder(tf.float32, shape = (self.num_particles))
            self.t_delta_seconds_tensor = tf.placeholder(tf.float32, shape = ())
            self.y_discrete_tensor = tf.placeholder(tf.int32, shape = (self.num_y_discrete_vars))
            self.y_continuous_tensor = tf.placeholder(tf.float32, shape = (self.num_y_continuous_vars))
            self.x_discrete_tensor, self.x_continuous_tensor, self.log_weights_tensor, self.ancestor_indices_tensor = self.create_next_particles_tensor(
                self.x_discrete_previous_tensor,
                self.x_continuous_previous_tensor,
                self.log_weights_previous_tensor,
                self.y_discrete_tensor,
                self.y_continuous_tensor,
                self.t_delta_seconds_tensor)
            # Generate computational graph to support generate_initial_simulation_timestep()
            self.x_discrete_initial_sim_tensor, self.x_continuous_initial_sim_tensor, self.y_discrete_initial_sim_tensor, self.y_continuous_initial_sim_tensor = self.create_initial_simulation_timestep_tensor()
            # Generate computational graph to support generate_next_simulation_timestep()
            self.x_discrete_previous_sim_tensor = tf.placeholder(tf.int32, shape = (self.num_x_discrete_vars))
            self.x_continuous_previous_sim_tensor = tf.placeholder(tf.float32, shape = (self.num_x_continuous_vars))
            self.t_delta_seconds_sim_tensor = tf.placeholder(tf.float32, shape = ())
            self.x_discrete_sim_tensor, self.x_continuous_sim_tensor, self.y_discrete_sim_tensor, self.y_continuous_sim_tensor = self.create_next_simulation_timestep_tensor(
                self.x_discrete_previous_sim_tensor,
                self.x_continuous_previous_sim_tensor,
                self.t_delta_seconds_sim_tensor)
        # Initialize session with this graph
        self.smc_model_session = tf.Session(graph = self.smc_model_graph)

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
        y_continuous_tensor):
        raise Exception('create_y_bar_x_log_pdf_tensor() called from parent SMCModel class. Should be overriden by child class.')

    def create_y_bar_x_sample_tensor(
        self,
        x_discrete_tensor,
        x_continuous_tensor):
        raise Exception('create_y_bar_x_sample_tensor() called from parent SMCModel class. Should be overriden by child class.')

    # Functions which are used in building the computational graph above

    def create_initial_particles_tensor(
        self,
        y_discrete_initial_tensor,
        y_continuous_initial_tensor):
        num_particles_tensor = tf.constant(self.num_particles, tf.int32)
        x_discrete_initial_tensor, x_continuous_initial_tensor = self.create_x_initial_sample_tensor(num_particles_tensor)
        log_weights_unnormalized_initial_tensor = self.create_y_bar_x_log_pdf_tensor(
            x_discrete_initial_tensor,
            x_continuous_initial_tensor,
            y_discrete_initial_tensor,
            y_continuous_initial_tensor)
        log_weights_initial_tensor = self.create_normalized_log_weights_tensor(
            log_weights_unnormalized_initial_tensor)
        return x_discrete_initial_tensor, x_continuous_initial_tensor, log_weights_initial_tensor

    def create_next_particles_tensor(
        self,
        x_discrete_previous_tensor,
        x_continuous_previous_tensor,
        log_weights_previous_tensor,
        y_discrete_tensor,
        y_continuous_tensor,
        t_delta_seconds_tensor):
        ancestor_indices_tensor = self.create_ancestor_indices_tensor(log_weights_previous_tensor)
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
        return x_discrete_tensor, x_continuous_tensor, log_weights_tensor, ancestor_indices_tensor

    def create_initial_simulation_timestep_tensor(
        self):
        num_samples_tensor = tf.constant(1, tf.int32)
        x_discrete_initial_sim_tensor, x_continuous_initial_sim_tensor = self.create_x_initial_sample_tensor(num_samples_tensor)
        y_discrete_initial_sim_tensor, y_continuous_initial_sim_tensor = self.create_y_bar_x_sample_tensor(
            x_discrete_initial_sim_tensor,
            x_continuous_initial_sim_tensor)
        return x_discrete_initial_sim_tensor, x_continuous_initial_sim_tensor, y_discrete_initial_sim_tensor, y_continuous_initial_sim_tensor

    def create_next_simulation_timestep_tensor(
        self,
        x_discrete_previous_tensor,
        x_continuous_previous_tensor,
        t_delta_seconds_tensor):
        x_discrete_tensor, x_continuous_tensor = self.create_x_bar_x_previous_sample_tensor(
            x_discrete_previous_tensor,
            x_continuous_previous_tensor,
            t_delta_seconds_tensor)
        y_discrete_tensor, y_continuous_tensor = self.create_y_bar_x_sample_tensor(
            x_discrete_tensor,
            x_continuous_tensor)
        return x_discrete_tensor, x_continuous_tensor, y_discrete_tensor, y_continuous_tensor

    def create_ancestor_indices_tensor(self, log_weights_previous_tensor):
        ancestor_indices_tensor = tf.squeeze(
            tf.multinomial(
                [log_weights_previous_tensor],
                tf.shape(log_weights_previous_tensor)[0]))
        return ancestor_indices_tensor

    def create_resampled_particles_tensor(
        self,
        x_discrete_tensor,
        x_continuous_tensor,
        ancestor_indices_tensor):
        x_discrete_resampled_tensor = tf.gather(
            x_discrete_tensor,
            ancestor_indices_tensor)
        x_continuous_resampled_tensor = tf.gather(
            x_continuous_tensor,
            ancestor_indices_tensor)
        return x_discrete_resampled_tensor, x_continuous_resampled_tensor

    def create_normalized_log_weights_tensor(self, log_weights_unnormalized_tensor):
        log_weights_normalized_tensor = tf.subtract(
            log_weights_unnormalized_tensor,
            tf.reduce_logsumexp(log_weights_unnormalized_tensor))
        return log_weights_normalized_tensor


    # Functions which provide the interface to this class. These functions run
    # portions of the computational graph above to generate results

    # Generate an initial set of X particles based on an initial Y value
    def generate_initial_particles(
        self,
        y_discrete_initial,
        y_continuous_initial):
        return self.smc_model_session.run(
            [self.x_discrete_initial_tensor, self.x_continuous_initial_tensor, self.log_weights_initial_tensor],
            feed_dict = {
                self.y_discrete_initial_tensor: y_discrete_initial,
                self.y_continuous_initial_tensor : y_continuous_initial})

    # Generate a set of X particles and weights (along with indices which point
    # to their ancestors) for the current time step based on the set of
    # particles and weights from the previous time step and a Y value from the
    # current time step
    def generate_next_particles(
        self,
        x_discrete_previous, x_continuous_previous,
        log_weights_previous,
        y_discrete, y_continuous,
        t_delta):
        t_delta_seconds = t_delta/np.timedelta64(1, 's')
        return self.smc_model_session.run(
            [self.x_discrete_tensor, self.x_continuous_tensor, self.log_weights_tensor, self.ancestor_indices_tensor],
            feed_dict = {
                self.x_discrete_previous_tensor: x_discrete_previous,
                self.x_continuous_previous_tensor : x_continuous_previous,
                self.log_weights_previous_tensor : log_weights_previous,
                self.y_discrete_tensor: y_discrete,
                self.y_continuous_tensor: y_continuous,
                self.t_delta_seconds_tensor: t_delta_seconds})

    # Generate an entire trajectory of X particles along with their weights and
    # ancestor indices based on an entire trajectory of Y values
    def generate_particle_trajectory(
        self,
        variable_structure,
        t_trajectory,
        y_discrete_trajectory,
        y_continuous_trajectory):
        num_timesteps = len(t_trajectory)
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
            y_discrete_trajectory[0],
            y_continuous_trajectory[0])
        for i in range(1, num_timesteps):
            x_discrete_particles_trajectory[i], x_continuous_particles_trajectory[i], log_weights_trajectory[i], ancestor_indices_trajectory[i] = self.generate_next_particles(
                x_discrete_particles_trajectory[i - 1],
                x_continuous_particles_trajectory[i - 1],
                log_weights_trajectory[i - 1],
                y_discrete_trajectory[i],
                y_continuous_trajectory[i],
                t_trajectory[i] - t_trajectory[i - 1])
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
        x_discrete_previous_sim,
        x_continuous_previous_sim,
        t_delta_sim):
        t_delta_seconds_sim = t_delta_sim/np.timedelta64(1, 's')
        return self.smc_model_session.run(
            [self.x_discrete_sim_tensor, self.x_continuous_sim_tensor, self.y_discrete_sim_tensor, self.y_continuous_sim_tensor],
            feed_dict = {
                self.x_discrete_previous_sim_tensor: x_discrete_previous_sim,
                self.x_continuous_previous_sim_tensor: x_continuous_previous_sim,
                self.t_delta_seconds_sim_tensor: t_delta_seconds_sim})

    # Generate an entire trajectory of simulated X and Y data
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
