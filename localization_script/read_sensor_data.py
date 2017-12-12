#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:14:09 2017

@author: charles

This script reads in the sensor data in JSON format and integrates with the
position model.
"""

import json, os, sys

import pandas as pd

with open("./input_data/radio_obs_20171127_170000.json") as json_file:
        json_data = json.load(json_file)
        print json_data
        
json_data[0]

json_data[1]
json_data[2]
json_data[3]

json_data[3]['local_id']
json_data[3]['local_type']

df = pd.DataFrame(data = json_data)