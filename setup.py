#!/usr/bin/env python

from distutils.core import setup

setup(name='smclocalize',
      version='0.1',
      description='Sequential Monte Carlo model for sensor localization',
      url='http://github.com/wildflowerschools/smclocalize',
      author='Ted Quinn',
      author_email='ted.quinn@wildflowerschools.org',
      license='MIT',
      packages=['smclocalize'],
      scripts=['bin/smclocalize_worker'],
      install_requires=[
          'numpy>=1.14',
          'scipy>=1.0',
          'pandas>=0.22',
          'tensorflow>=1.5'
      ])
