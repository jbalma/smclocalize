import numpy as np

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

        child_entity_keys = [child_entity_string + '_' + str(child_entity_id) for child_entity_id in child_entity_ids]
        material_entity_keys = [material_entity_string + '_' + str(material_entity_id) for material_entity_id in material_entity_ids]
        teacher_entity_keys = [teacher_entity_string + '_' + str(teacher_entity_id) for teacher_entity_id in teacher_entity_ids]
        area_entity_keys = [area_entity_string + '_' + str(area_entity_id) for area_entity_id in area_entity_ids]
        sensor_keys = child_entity_keys + material_entity_keys + teacher_entity_keys + area_entity_keys
        self.entity_id_index = {k: idx for idx,k in enumerate(sensor_keys)}

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

        # Define the number of discrete and continuous x variables. This is the
        # key information needed by the SMC model functions.
        self.num_x_discrete_vars = self.num_sensors
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
        # We *do* now store pings from material sensors to area sensors (and vice versa)
        # self.extract_y_variables_mask[
        #     self.num_child_sensors:(self.num_child_sensors + self.num_material_sensors),
        #     self.num_moving_sensors:] = False
        # self.extract_y_variables_mask[
        #     self.num_moving_sensors:,
        #     self.num_child_sensors:(self.num_child_sensors + self.num_material_sensors)] = False

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

        self.x_discrete_names = ['{} status'.format(sensor_name) for sensor_name in self.sensor_names]
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

    # Parse a dataframe containing a single time step of ping data and arrange
    # it in the flat format of the Y variables
    def parse_ping_data(self, dataframe):
        ping_status_matrix, rssi_matrix = self.parse_ping_data_matrix(
            dataframe)
        return self.extract_y_variables(ping_status_matrix), self.extract_y_variables(rssi_matrix)

    # Parse a dataframe containing a single time step of ping data and arrange
    # it in matrix form
    def parse_ping_data_matrix(self, dataframe):
        ping_status_matrix = np.ones(
            (self.num_sensors, self.num_sensors),
            dtype='int')
        rssi_matrix = np.zeros(
            (self.num_sensors, self.num_sensors),
            dtype='float')
        for row in range(len(dataframe)):
            remote_idx = self.entity_id_index.get(dataframe.iloc[row]['remote_type'] + '_' + str(dataframe.iloc[row]['remote_id']))
            local_idx = self.entity_id_index.get(dataframe.iloc[row]['local_type'] + '_' + str(dataframe.iloc[row]['local_id']))
            ping_status_matrix[remote_idx, local_idx] = 0
            rssi_matrix[remote_idx, local_idx] = dataframe.iloc[row]['rssi']
        return ping_status_matrix, rssi_matrix

    # Parse a dataframe containing a single time step of moving sensor position
    # data and arrange it in the flat format of the X variables
    def parse_moving_sensor_position_data(self, dataframe):
        sensor_position_matrix = self.parse_sensor_position_data_matrix(
            dataframe)
        return self.extract_x_variables(sensor_position_matrix)

    # Parse a dataframe containing fixed sensor position data and arrange it in
    # the matrix form required by smc_model
    def parse_fixed_sensor_position_data(self, dataframe):
        sensor_position_matrix = self.parse_sensor_position_data_matrix(
            dataframe)
        return sensor_position_matrix[-self.num_area_sensors:]

    # Parse a dataframe containing sensor position data and arrange it in matrix
    # form
    def parse_sensor_position_data_matrix(self, dataframe):
        sensor_position_matrix = np.full(
            (self.num_sensors, self.num_dimensions),
            np.nan,
            dtype='float')
        for row in range(len(dataframe)):
            entity_idx = self.entity_id_index.get(dataframe.iloc[row]['entity_type'] + '_' + str(dataframe.iloc[row]['entity_id']))
            sensor_position_matrix[entity_idx, 0] = dataframe.iloc[row]['l']
            sensor_position_matrix[entity_idx, 1] = dataframe.iloc[row]['w']
        return sensor_position_matrix
