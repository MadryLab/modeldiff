#!/usr/bin/env python
from setuptools import setup

setup(name="modeldiff",
      version="1.0.0",
      description="ModelDiff: A Framework for Comparing Learning Algorithms",
      long_description="Check our paper https://arxiv.org/abs/2211.12491",
      author="MadryLab",
      author_email='harshay@mit.edu',
      license_files=('LICENSE.txt', ),
      packages=['modeldiff'],
      install_requires=[
       "traker",
       ],
      extras_require={
          'notebooks':
              ["scikit_learn",
               "torchvision",
               "seaborn",
               "wget",
               "scipy",
               "wilds"
               ],},
      include_package_data=True,
      )
