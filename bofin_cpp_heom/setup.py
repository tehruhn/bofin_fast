#!/usr/bin/env python
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy as np
from os import walk

extensions = [
    Extension(
        "bofincpp.interfacer",
        sources=["bofincpp/interfacer.pyx", "bofincpp/utilities.cpp"],
        include_dirs=[np.get_include(), '.', 'bofincpp'],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language='c++')
        # sources=['utilities.cpp'])
]

REQUIRES = ["Cython", "qutip", "numpy", "scipy"]

setup(name='bofincpp',
      version='0.0.1',
      description='C++ version for Bosonic and Fermionic hierarchical-equations-of-motion library for QuTiP',
      author='Neill Lambert, Tarun Raheja, Shahnawaz Ahmed, Alex Pitchford',
      author_email='nwlambert@gmail.com',
      packages = find_packages(include=['heom', 'heom.*']),
      requires = REQUIRES,
      ext_modules = cythonize(extensions)
     )
