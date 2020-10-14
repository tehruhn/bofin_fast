#!/usr/bin/env python
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "heom.interfacer",
        ["heom/interfacer.pyx"],
        include_dirs=[np.get_include(), '.'],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]),
]


REQUIRES = ['numpy', 'scipy']

setup(name='heom',
      version='0.0.1',
      description='heom',
      author='Neill Lambert, Tarun Raheja, Shahnawaz Ahmed, Alex Pitchford',
      author_email='nwlambert@gmail.com',
      packages = find_packages(include=['heom', 'heom.*']),
      requires = REQUIRES,
      ext_modules = cythonize(extensions)
     )
