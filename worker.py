#!/usr/bin/env python3
import numpy as np
from argparse import ArgumentParser
import pandas as pd
import redis
import time, os, sys
from collections import namedtuple
from smclocalize import *
import firebase_admin
from firebase_admin import firestore
import pytz


ModelState = namedtuple("ModelState", "x_discrete_particles x_continuous_particles log_weights")

class ConfigError(ValueError):
    pass


class LocationModelWorker:
    def __init__(self, classroom_id, num_particles, num_spinup_frames):
        self.classroom_id = classroom_id
        self.num_spinup_frames = num_spinup_frames
        self.num_particles = num_particles


        #Room geometry
        # TODO: fetch from firebase
        feet_to_meters = 12*2.54/100
        room_size = np.array([(19.0 + 4.0/12.0 + 15.0/12.0 + 43.0 + 2.0/12.0 + 2.0)*feet_to_meters,
                              (11.0 + 9.0/12.0)*feet_to_meters])
        self.room_corners = np.array([[0.0, 0.0], room_size])

        self.fixed_sensor_positions = np.array ([
            [ 6.8834,  3.2766],
            [ 0.6096,  3.2766],
            [ 1.6096,  4.2766],
            [19.7358,  0.9144],
            [10.7358,  1.9144],
            [ 2.6096,  5.2766],
            [19.7358,  0.9144],
            [10.8458,  0.3048]])

        redis_url = os.getenv('REDIS_URL')
        if redis_url is None:
            raise ConfigError("Missing environment variable REDIS_URL. Please see README.")

        self.redis_handle = redis.Redis.from_url(redis_url)

        firebase_url = os.getenv('FIREBASE_URL')
        if firebase_url is None:
            raise ConfigError("Missing environment variable FIREBASE_URL. Please see README.")

        self.input_queue = 'radio_obs_classroom_%d' % classroom_id

        credentials_certificate = firebase_admin.credentials.Certificate({
            'private_key': os.getenv('FIREBASE_PRIVATE_KEY', False),
            'client_email': os.getenv('FIREBASE_CLIENT_EMAIL', ''),
            'type': 'service_account',
            'token_uri': 'https://accounts.google.com/o/oauth2/token',
            'project_id': os.getenv('FIREBASE_PROJECT_ID', 'sensei-b9fb6'),
            'private_key_id': os.getenv('FIREBASE_PRIVATE_KEY_ID'),
            'client_id': os.getenv('FIREBASE_CLIENT_ID'),
            'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
            'token_uri': 'https://accounts.google.com/o/oauth2/token',
            'auth_provider_x509_cert_url': 'https://www.googleapis.com/oauth2/v1/certs',
            'client_x509_cert_url': os.getenv('FIREBASE_CLIENT_X509_CERT_URL')
            })
        firebase_admin.initialize_app(credentials_certificate, options={
            'databaseURL': firebase_url
        })
        self.firebase = firebase_admin.firestore.client();


    def run(self):
        # Wait until we have 4 frames of data to look at
        dataframes = []
        print("Fetching initial frames from redis on queue %s..." % self.input_queue)
        while len(dataframes) < 4:
            _, data = self.redis_handle.brpop(self.input_queue)
            print("data = %s" % data)
            dataframes.append(pd.read_json(data))

        initial_frames = pd.concat(dataframes, ignore_index = True)

        child_entity_ids = np.union1d(pd.unique(initial_frames[initial_frames.local_type == 'child'].local_id),
                                      pd.unique(initial_frames[initial_frames.remote_type == 'child'].remote_id)).tolist()
        material_entity_ids = np.union1d(pd.unique(initial_frames[initial_frames.local_type == 'material'].local_id),
                                         pd.unique(initial_frames[initial_frames.remote_type == 'material'].remote_id)).tolist()
        teacher_entity_ids = np.union1d(pd.unique(initial_frames[initial_frames.local_type == 'teacher'].local_id),
                                        pd.unique(initial_frames[initial_frames.remote_type == 'teacher'].remote_id)).tolist()
        area_entity_ids = np.union1d(pd.unique(initial_frames[initial_frames.local_type == 'area'].local_id),
                                     pd.unique(initial_frames[initial_frames.remote_type == 'area'].remote_id)).tolist()

        print("child_entity_ids = %s" % child_entity_ids)
        print("material_entity_ids = %s" % material_entity_ids)
        print("teacher_entity_ids = %s" % teacher_entity_ids)
        print("area_entity_ids = %s" % area_entity_ids)

        self.entity_ids = (
            [['child', id] for id in child_entity_ids] +
            [['material', id] for id in material_entity_ids] +
            [['teacher', id] for id in teacher_entity_ids] +
            [['area', id] for id in area_entity_ids])


        self.variable_structure = SensorVariableStructure(child_entity_ids,
                                                          material_entity_ids,
                                                          teacher_entity_ids,
                                                          area_entity_ids)

        print("room_corners = %s" % self.room_corners)

        self.fixed_sensor_positions = self.fixed_sensor_positions[:self.variable_structure.num_fixed_sensors]

        print("fixed_sensor_positions = %s" % self.fixed_sensor_positions)
        self.sensor_model = SensorModel(
            self.variable_structure,
            self.room_corners,
            self.fixed_sensor_positions,
            self.num_particles)

        self.model_state = ()

        # Run the model on the first frames
        previous_frame_valid_time = None
        for i in range(self.num_spinup_frames):
            frame_valid_time = dataframes[i]['observed_at'][0]
            if not previous_frame_valid_time:
                delta_t = 0
            else:
                delta_t = frame_valid_time - previous_frame_valid_time
            locations = self.process_frame(dataframes[i], frame_valid_time, delta_t, i==0)
            previous_frame_valid_time = frame_valid_time

        while True:
            _, data = self.redis_handle.brpop(self.input_queue)
            frame = pd.read_json(data)
            frame_valid_time = frame['observed_at'][0]
            delta_t = frame_valid_time - previous_frame_valid_time
            locations = self.process_frame(frame, frame_valid_time, delta_t, i==0)
            previous_frame_valid_time = frame_valid_time

    def process_frame(self, frame, frame_valid_time, delta_t, initialize = False):
        start_time = time.time()
        print("Processing frame %s" % frame_valid_time)
        y_discrete, y_continuous = self.variable_structure.sensor_data_parse_one_timestep(frame)
        if initialize:
            self.state = ModelState(*self.sensor_model.generate_initial_particles(y_discrete, y_continuous))
        else:
            self.state = ModelState(*self.sensor_model.generate_next_particles(*self.state, y_discrete, y_continuous, delta_t)[:3])

        x_continuous_mean_particle = np.average(
            self.state.x_continuous_particles,
            axis=0,
            weights=np.repeat(np.exp(self.state.log_weights), self.variable_structure.num_x_continuous_vars).reshape(
                (self.num_particles,
                 self.variable_structure.num_x_continuous_vars)))

        x_continuous_squared_mean_particle = np.average(
            np.square(self.state.x_continuous_particles),
            axis=0,
            weights=np.repeat(np.exp(self.state.log_weights), self.variable_structure.num_x_continuous_vars).reshape(
                (self.num_particles,
                 self.variable_structure.num_x_continuous_vars)))

        x_continuous_sd_particle = np.sqrt(np.abs(x_continuous_squared_mean_particle - np.square(x_continuous_mean_particle)))

        locations = x_continuous_mean_particle.reshape(self.variable_structure.num_moving_sensors, self.variable_structure.num_dimensions)
        std_deviations = x_continuous_sd_particle.reshape(self.variable_structure.num_moving_sensors, self.variable_structure.num_dimensions)
        print("*** processing time: %dms" % (int((time.time() - start_time) * 1000)))
        self.publish_to_firebase(locations, std_deviations, frame_valid_time)
        print("="*80)

    def publish_to_firebase(self, locations, std_deviations, timestamp):
        batch = self.firebase.batch()

        for ((entity_type, entity_id), (x,y), (x_stddev, y_stddev)) in zip(self.entity_ids, locations, std_deviations):
            path = 'classrooms/%s/entity_locations/%s-%s-%s' % (self.classroom_id, entity_type, entity_id, timestamp)
            doc_ref = self.firebase.document(path)
            print("%s %d: (%f, %f)" % (entity_type, entity_id, x, y))
            publish_location = {
                'entityType': u'%s' % entity_type,
                'entityId': entity_id,
                'x': x.item(),
                'y': y.item(),
                'xStdDev': x_stddev.item(),
                'yStdDev': y_stddev.item(),
                'timestamp': timestamp
            }
            batch.set(doc_ref, publish_location)

        batch.commit()

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("-c", "--classroom_id", dest="classroom_id", required=True, type=int, help="Classroom ID")
    parser.add_argument("-n", "--num_particles", dest="num_particles", type=int, default=10000, help="Number of particles to use in the model")
    parser.add_argument("-f", "--num_spinup_frames", dest="num_spinup_frames", type=int, default=4, help="Number of frames to wait before assuming sensor list is sufficient.")

    options = parser.parse_args()

    try:
        worker = LocationModelWorker(options.classroom_id, options.num_particles, options.num_spinup_frames)
        worker.run()
    except ConfigError as error:
        sys.exit(error)
