#!/usr/bin/env python
from setuptools import setup

setup(name="modeldiff",
      version="1.0.1",
      description="ModelDiff: A Framework for Comparing Learning Algorithms",
      long_description="Check out our repo (https://github.com/MadryLab/modeldiff) and our paper (https://arxiv.org/abs/2211.12491).",
      author="MadryLab",
      author_email='harshay@mit.edu',
      license_files=('LICENSE.txt', ),
      packages=['modeldiff'],
      install_requires=[
          "traker",
          "numpy"
      ],
      extras_require={
          'notebooks':
              ["scikit_learn",
               "torchvision",
               "seaborn",
               "wget",
               "scipy",
               "wilds",
               "pandas"
               ], },
      include_package_data=True,
      )
